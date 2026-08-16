//! Deterministic transcendental functions.
//!
//! We do not call `f64::exp` / `f64::ln` from libm anywhere on the critical
//! path. Two reasons, both operational rather than aesthetic:
//!
//! 1. **Replay must be bit-exact.** libm results differ between glibc versions
//!    and between the vectorised and scalar paths the compiler may select. If
//!    the replay harness cannot reproduce a production decision bit for bit,
//!    every post-mortem becomes an argument about floating point instead of an
//!    argument about the strategy.
//! 2. **Tail latency.** glibc's `exp` has data-dependent branches and can take
//!    a slow path. Ours is branch-light and constant-work.
//!
//! `f64::sqrt` *is* used directly: it compiles to a single `sqrtsd`, which
//! IEEE-754 requires to be correctly rounded, so it is already deterministic.

const LN2_HI: f64 = 6.931_471_803_691_238e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
// `1 / ln(2)`, i.e. `log2(e)` — spelled out to the same f64 value
// `core::f64::consts::LOG2_E` rounds to, but named for what it's used for
// here (range-reducing `exp`'s argument), not what it equals.
const INV_LN2: f64 = core::f64::consts::LOG2_E;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// e^x, accurate to ~1e-10 relative over the range we care about.
///
/// Range reduction `x = k·ln2 + r`, degree-9 Taylor on `r`, then scale by `2^k`
/// through direct exponent-field construction.
///
/// `k` is derived from `x`, which the two early returns above already
/// bound to `[-745.0, 709.0]` — so `k` (via `x * INV_LN2`) is bounded to
/// roughly `[-1075, 1024]` before either branch below ever runs, and every
/// arithmetic op on it in this function stays well inside `i64`/`u64`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }

    let k = (x * INV_LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let kf = k as f64;
    // Two-part ln2 keeps the reduced argument accurate for large |x|.
    let r = (x - kf * LN2_HI) - kf * LN2_LO;

    // Horner form of sum r^n / n! for n = 0..=12.
    //
    // |r| <= ln2/2 = 0.3466, so the first dropped term is r^13/13! ~ 1.1e-16 —
    // below f64 epsilon relative to 1. Degree 9 would leave ~7e-12 relative
    // error, which is visible at the 12th significant digit and would make the
    // "does replay match production" test flaky at tight tolerances.
    // Coefficients highest degree (r^12) first, so the fold below evaluates
    // Horner's method outside-in exactly as a hand-nested
    // `c0 + r*(c1 + r*(c2 + ...))` expression would — same operations, same
    // order, so this is bit-for-bit identical to that form. It replaces one
    // here on purpose: a expression nested 12 parentheses deep is not just
    // hard to read, it is slow enough to *format* that `rustfmt` does not
    // return in practical time on this file, which blocked `cargo fmt`
    // across the whole workspace.
    const COEFFS: [f64; 13] = [
        1.0 / 479_001_600.0, // r^12
        1.0 / 39_916_800.0,  // r^11
        1.0 / 3_628_800.0,   // r^10
        1.0 / 362_880.0,     // r^9
        1.0 / 40320.0,       // r^8
        1.0 / 5040.0,        // r^7
        1.0 / 720.0,         // r^6
        1.0 / 120.0,         // r^5
        1.0 / 24.0,          // r^4
        1.0 / 6.0,           // r^3
        1.0 / 2.0,           // r^2
        1.0,                 // r^1
        1.0,                 // r^0
    ];
    let mut p = 0.0;
    for c in COEFFS {
        p = c + r * p;
    }

    // 2^k by exponent-field construction. Bias is 1023.
    //
    // The subnormal range needs two steps. For `k < -1022` the biased exponent
    // `k + 1023` goes non-positive; casting that to `u64` wraps to an enormous
    // value, and the shift then produces a *large negative number* where a tiny
    // positive one belongs. `exp(-720.0)` returned -6.56e303 instead of
    // 2.03e-313 — not an inaccuracy, a sign flip of 316 orders of magnitude.
    //
    // This was reachable. `norm_pdf` calls `exp(-x²/2)`, so any `|x| > 37.6`
    // enters the range, and `twap::fair` computes `norm_pdf(z)` for unbounded
    // `z`. Standardised moneyness of 38 is routine in the TWAP endgame, where
    // remaining standard deviation collapses cubically — that is, precisely
    // where this strategy expects to make its money. A negative pdf there flips
    // the sign of delta, which flips the inventory skew, which makes the engine
    // lean into risk instead of away from it.
    let two_k = if k >= -1022 {
        f64::from_bits(((k as i64 + 1023) as u64) << 52)
    } else {
        // Scale in two stages so neither exponent field goes out of range.
        let head = f64::from_bits((((k + 1022) as i64 + 1023) as u64) << 52);
        head * f64::from_bits(1u64 << 52) // × 2^-1022
    };
    p * two_k
}

/// Natural log, accurate to ~1e-12 relative. Returns NaN for x <= 0.
///
/// `e` is an 11-bit IEEE-754 exponent field masked to `0x7ff` (`[0, 2047]`)
/// minus the `1023` bias, so it is bounded to `[-1023, 1024]` before the
/// recentring step below can add at most `1` to it.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn ln(x: f64) -> f64 {
    if x <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    // Mantissa in [1, 2).
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);

    // Recentre to [sqrt(0.5), sqrt(2)) so the atanh series converges fast.
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }

    // ln(m) = 2 * atanh(s), s = (m-1)/(m+1), |s| <= 0.1716
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let series = s
        * (2.0
            + s2 * (2.0 / 3.0
                + s2 * (2.0 / 5.0
                    + s2 * (2.0 / 7.0
                        + s2 * (2.0 / 9.0 + s2 * (2.0 / 11.0 + s2 * (2.0 / 13.0)))))));

    (e as f64) * LN2_HI + ((e as f64) * LN2_LO + series)
}

/// Standard normal PDF.
#[inline(always)]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * exp(-0.5 * x * x)
}

/// Standard normal CDF via Abramowitz & Stegun 26.2.17.
///
/// Absolute error < 7.5e-8, i.e. under 0.1 parts per million — comfortably
/// finer than the 1-tick (1000 ppm) resolution of the price grid we quote on.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;

    if x.is_nan() {
        return f64::NAN;
    }
    // Beyond +-8 sigma the answer is 0 or 1 to within f64 resolution, and the
    // rational approximation loses its footing. Short-circuit.
    if x > 8.0 {
        return 1.0;
    }
    if x < -8.0 {
        return 0.0;
    }

    let neg = x < 0.0;
    let ax = if neg { -x } else { x };
    let t = 1.0 / (1.0 + P * ax);
    let poly = t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * B5))));
    let upper = norm_pdf(ax) * poly; // = 1 - Phi(ax)

    if neg {
        upper
    } else {
        1.0 - upper
    }
}

/// Inverse standard normal CDF (Acklam's rational approximation, refined by one
/// Halley step). Used by the risk layer to turn confidence levels into sigma
/// multiples for the safety margin.
pub fn norm_ppf(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const PLOW: f64 = 0.024_25;
    const PHIGH: f64 = 1.0 - PLOW;

    let mut x = if p < PLOW {
        let q = (-2.0 * ln(p)).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= PHIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * ln(1.0 - p)).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement takes the ~1e-9 approximation to near machine eps.
    let e = norm_cdf(x) - p;
    let u = e * 2.506_628_274_631_000_5 * exp(x * x / 2.0);
    x -= u / (1.0 + x * u / 2.0);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn exp_matches_known_values() {
        assert!(close(exp(0.0), 1.0, 1e-15));
        assert!(close(exp(1.0), core::f64::consts::E, 1e-12));
        assert!(close(exp(-1.0), 1.0 / core::f64::consts::E, 1e-12));
        assert!(close(exp(10.0), 22_026.465_794_806_718, 1e-6));
        assert!(close(exp(-10.0), 4.539_992_976_248_485e-5, 1e-15));
        assert_eq!(exp(-1000.0).to_bits(), 0.0f64.to_bits());
        assert!(exp(1000.0).is_infinite());
    }

    #[test]
    fn exp_is_never_negative() {
        // e^x > 0 for all real x. The subnormal-range bug violated this, and
        // the violation was silent: no NaN, no infinity, just a large negative
        // number that propagated straight into a probability.
        let mut x = -760.0;
        while x < 720.0 {
            let v = exp(x);
            assert!(v >= 0.0, "exp({x}) = {v}");
            assert!(!v.is_nan(), "exp({x}) is NaN");
            x += 0.37;
        }
    }

    #[test]
    fn exp_is_correct_in_the_subnormal_range() {
        for x in [-708.0f64, -709.0, -715.0, -720.0, -730.0, -740.0, -744.0] {
            let ours = exp(x);
            let theirs = x.exp();
            assert!(ours > 0.0, "exp({x}) = {ours}");
            assert!(
                (ours - theirs).abs() <= 1e-6 * theirs,
                "exp({x}): ours={ours:e} libm={theirs:e}"
            );
        }
        assert_eq!(exp(-800.0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn norm_pdf_stays_non_negative_at_extreme_moneyness() {
        // The path by which the subnormal bug was reachable in production.
        for z in [30.0f64, 37.0, 37.6, 38.0, 38.5, 40.0, 100.0] {
            let p = norm_pdf(z);
            assert!(p >= 0.0 && p.is_finite(), "norm_pdf({z}) = {p}");
            assert!((norm_pdf(-z) - p).abs() < 1e-300);
        }
    }

    #[test]
    fn exp_agrees_with_libm_across_range() {
        // Sanity check against the platform libm. We do not *use* libm, but if we
        // disagree with it by more than 1e-9 relative, our approximation is wrong.
        let mut x = -20.0;
        while x <= 20.0 {
            let ours = exp(x);
            let theirs = x.exp();
            assert!(
                (ours - theirs).abs() <= 1e-9 * theirs.abs().max(1e-300),
                "exp({x}): ours={ours} libm={theirs}"
            );
            x += 0.013;
        }
    }

    #[test]
    fn ln_agrees_with_libm_across_range() {
        let mut x = 1e-6;
        while x < 1e6 {
            let ours = ln(x);
            let theirs = x.ln();
            assert!(
                (ours - theirs).abs() <= 1e-11,
                "ln({x}): ours={ours} libm={theirs}"
            );
            x *= 1.117;
        }
        assert!(ln(-1.0).is_nan());
        assert!(ln(0.0).is_nan());
    }

    #[test]
    fn exp_ln_round_trip() {
        for i in 1..500 {
            let x = i as f64 * 0.37;
            assert!(close(ln(exp(x)), x, 1e-9));
        }
    }

    #[test]
    fn norm_cdf_is_accurate_and_symmetric() {
        // A&S 26.2.17 carries |eps| < 7.5e-8. That is 0.075 ppm — two orders of
        // magnitude finer than the 1000 ppm price grid we quote on, so tolerance
        // here is set to the approximation's guarantee, not to machine epsilon.
        assert!(close(norm_cdf(0.0), 0.5, 1e-9));
        assert!(close(norm_cdf(1.0), 0.841_344_746_068_543, 1e-7));
        assert!(close(norm_cdf(-1.0), 0.158_655_253_931_457, 1e-7));
        assert!(close(norm_cdf(1.959_963_984_540_054), 0.975, 1e-7));
        assert!(close(norm_cdf(3.0), 0.998_650_101_968_370, 1e-7));

        let mut x = -6.0;
        while x <= 6.0 {
            assert!(close(norm_cdf(x) + norm_cdf(-x), 1.0, 1e-7));
            x += 0.05;
        }
    }

    #[test]
    fn norm_cdf_is_monotone() {
        let mut prev = 0.0;
        let mut x = -8.0;
        while x <= 8.0 {
            let v = norm_cdf(x);
            assert!(v >= prev - 1e-12, "not monotone at {x}");
            prev = v;
            x += 0.001;
        }
    }

    #[test]
    fn norm_ppf_inverts_norm_cdf() {
        for i in 1..1000 {
            let p = i as f64 / 1000.0;
            let x = norm_ppf(p);
            assert!(close(norm_cdf(x), p, 1e-7), "ppf/cdf mismatch at p={p}");
        }
    }
}
