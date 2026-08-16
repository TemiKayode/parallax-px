//! Closed-form fair value for a TWAP-settled binary.
//!
//! # Why this module is the centre of gravity
//!
//! Polymarket's crypto 5-minute, 15-minute and 4-hour markets do not settle on
//! the spot print at expiry. They settle on a Chainlink **time-weighted average
//! price** over a window of `W` seconds ending at expiry. Almost every naive
//! model — and, empirically, a good deal of the resting liquidity — prices these
//! as if they were spot-settled with `sqrt(tau)` time decay.
//!
//! That is wrong in a specific, exploitable way, and the error is largest
//! exactly where the money is.
//!
//! # The mathematics
//!
//! Model the underlying over the (short) remaining horizon as arithmetic
//! Brownian motion with volatility `sigma` in price units per sqrt(second).
//! Over five minutes the difference between arithmetic and geometric Brownian
//! motion is immaterial, and arithmetic keeps the average-of-a-path algebra
//! exact rather than approximate.
//!
//! Settlement value is `V = (1/W) * integral of S(u) du over [T-W, T]`.
//!
//! **Before the window opens** (`t <= T-W`, with `a = (T-W) - t`):
//!
//! ```text
//!   E[V]   = S(t)
//!   Var[V] = sigma^2 * (a + W/3)
//! ```
//!
//! The `W/3` term is the variance of the average of a Brownian path over the
//! window: `Var[(1/W) * int_0^W B_s ds] = W/3`. Note it is *one third* of the
//! variance of the terminal value. A TWAP-settled contract is meaningfully less
//! uncertain than a spot-settled one with the same expiry, and anyone pricing it
//! as spot-settled is systematically too close to 50 cents.
//!
//! **Inside the window** (`T-W < t < T`), with `r = T - t` remaining, elapsed
//! fraction `phi = (W - r)/W`, and `A` the average of the prices already
//! observed since `T-W`:
//!
//! ```text
//!   E[V]   = phi*A + (r/W)*S(t)
//!   Var[V] = sigma^2 * r^3 / (3 * W^2)
//! ```
//!
//! The two expressions agree at `r = W`, as they must.
//!
//! # The consequence
//!
//! Variance inside the window decays as **`r^3`**, not `r`. Standard deviation
//! decays as `r^(3/2)`, not `sqrt(r)`. Halfway through a 60-second settlement
//! window the remaining uncertainty is not 71% of what it was — it is 35%.
//! With ten seconds left on a 60-second window it is 0.68%, not 41%.
//!
//! A model using `sqrt(tau)` decay is wrong by a factor of ~60 in standard
//! deviation at that point, and it is wrong in the direction of pricing an
//! all-but-decided market as if it were still a coin flip. The whole reason the
//! rest of this system exists — the depth walker, the inventory penalty, the
//! rate-limit governor — is to convert that pricing difference into filled size
//! before the resting liquidity is repriced.

use px_core::math::{norm_cdf, norm_pdf};

/// Inputs to the fair-value computation. Deliberately a plain struct of `f64`
/// with no references: it is built on the stack in the hot path and never
/// escapes.
#[derive(Clone, Copy, Debug)]
pub struct TwapInputs {
    /// Current reference spot price of the underlying.
    pub spot: f64,
    /// Strike the market is written against.
    pub strike: f64,
    /// Volatility in *price units* per sqrt(second) (i.e. already multiplied by
    /// spot if the estimator is a relative-return estimator).
    pub sigma: f64,
    /// Seconds until expiry. Clamped at zero by the caller.
    pub tau: f64,
    /// Settlement averaging window in seconds. Zero means spot settlement.
    pub window: f64,
    /// Mean of the prices already observed inside the settlement window.
    /// Ignored when `tau > window`.
    pub observed_avg: f64,
    /// Relative 1-sigma uncertainty on `sigma` itself, from the estimator's
    /// effective sample size. Typically 0.1 to 0.3.
    pub sigma_rel_err: f64,
    /// Age of the spot observation in seconds. Non-zero age means the true spot
    /// has had time to drift away from what we last saw, and that drift is a
    /// genuine source of model uncertainty rather than something to ignore.
    pub spot_age: f64,
}

/// Fair value plus everything the sizing and margin logic needs to know about
/// how much to trust it.
#[derive(Clone, Copy, Debug, Default)]
pub struct TwapFair {
    /// P(settlement > strike).
    pub p: f64,
    /// Expected settlement value.
    pub mean: f64,
    /// Standard deviation of settlement value, in price units.
    pub sd: f64,
    /// Standardised moneyness `(mean - strike) / sd`.
    pub z: f64,
    /// d(p)/d(spot): probability change per unit move in the underlying.
    pub d_spot: f64,
    /// d(p)/d(sigma).
    pub d_sigma: f64,
    /// Total 1-sigma uncertainty on `p`, in probability units. This is the
    /// number the safety margin is built from: we require edge to exceed a
    /// multiple of it before committing capital.
    pub sigma_p: f64,
    /// True once the outcome is determined to within f64 resolution.
    pub decided: bool,
}

/// Variance of the settlement TWAP for unit volatility. Split out because the
/// relative-value monitor needs the same shape function to normalise gaps
/// across markets of different duration.
#[inline(always)]
pub fn variance_shape(tau: f64, window: f64) -> f64 {
    if window <= 0.0 {
        // Spot settlement: plain Brownian variance.
        return tau.max(0.0);
    }
    if tau <= 0.0 {
        return 0.0;
    }
    if tau >= window {
        // Window has not opened: a = tau - window seconds of pure diffusion,
        // then the W/3 of the averaging window itself.
        (tau - window) + window / 3.0
    } else {
        // Inside the window. The cubic collapse.
        (tau * tau * tau) / (3.0 * window * window)
    }
}

/// Expected settlement value given the part of the window already observed.
#[inline(always)]
pub fn expected_settlement(spot: f64, tau: f64, window: f64, observed_avg: f64) -> f64 {
    if window <= 0.0 || tau >= window {
        spot
    } else if tau <= 0.0 {
        observed_avg
    } else {
        let phi = (window - tau) / window;
        phi * observed_avg + (tau / window) * spot
    }
}

/// The main entry point. Allocation-free, branch-light, no libm calls.
pub fn fair(inp: &TwapInputs) -> TwapFair {
    let tau = inp.tau.max(0.0);
    let w = inp.window.max(0.0);

    let mean = expected_settlement(inp.spot, tau, w, inp.observed_avg);
    let shape = variance_shape(tau, w);
    let var = inp.sigma * inp.sigma * shape;
    let sd = var.sqrt();

    // Once the remaining standard deviation is a vanishing fraction of the
    // distance to the strike, the outcome is decided. We say so explicitly
    // rather than returning 0.999999 and letting downstream logic guess.
    if sd <= 0.0 || !sd.is_finite() {
        let p = if mean > inp.strike { 1.0 } else { 0.0 };
        return TwapFair {
            p,
            mean,
            sd: 0.0,
            z: if mean > inp.strike {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            },
            d_spot: 0.0,
            d_sigma: 0.0,
            sigma_p: 0.0,
            decided: true,
        };
    }

    let z = (mean - inp.strike) / sd;
    let p = norm_cdf(z);
    let pdf = norm_pdf(z);

    // d(mean)/d(spot) is 1 before the window opens, r/W inside it: as the
    // window fills up, our exposure to further spot moves shrinks linearly even
    // as our exposure to *variance* shrinks cubically.
    let dmean_dspot = if w <= 0.0 || tau >= w { 1.0 } else { tau / w };
    let d_spot = pdf * dmean_dspot / sd;

    // z is homogeneous of degree -1 in sigma, so dz/dsigma = -z/sigma.
    let d_sigma = if inp.sigma > 0.0 {
        pdf * (-z / inp.sigma)
    } else {
        0.0
    };

    // Uncertainty budget. Two independent contributions:
    //   1. Our spot observation is `spot_age` seconds old, so the true spot has
    //      had time to move by sigma*sqrt(age).
    //   2. Our volatility estimate has sampling error.
    let spot_unc = d_spot * inp.sigma * inp.spot_age.max(0.0).sqrt();
    let vol_unc = d_sigma * inp.sigma * inp.sigma_rel_err.max(0.0);
    let sigma_p = (spot_unc * spot_unc + vol_unc * vol_unc).sqrt();

    TwapFair {
        p,
        mean,
        sd,
        z,
        d_spot,
        d_sigma,
        sigma_p,
        decided: z.abs() > 8.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> TwapInputs {
        TwapInputs {
            spot: 65_000.0,
            strike: 65_000.0,
            // ~40% annualised on BTC: 0.40 / sqrt(365*24*3600) = 7.13e-5 per
            // sqrt(sec) in relative terms, times 65000 = ~4.63 $/sqrt(sec).
            sigma: 4.63,
            tau: 300.0,
            window: 60.0,
            observed_avg: 65_000.0,
            sigma_rel_err: 0.15,
            spot_age: 0.0,
        }
    }

    #[test]
    fn at_the_money_is_a_coin_flip() {
        let f = fair(&base());
        assert!((f.p - 0.5).abs() < 1e-6, "p = {}", f.p);
        assert!(f.sd > 0.0);
    }

    #[test]
    fn variance_shape_is_continuous_at_window_open() {
        // The two branches of the formula must agree at tau == W.
        let w = 60.0;
        let just_before = variance_shape(w + 1e-9, w);
        let just_after = variance_shape(w - 1e-9, w);
        assert!((just_before - just_after).abs() < 1e-6);
        assert!((just_before - w / 3.0).abs() < 1e-6);
    }

    #[test]
    fn twap_variance_is_one_third_of_spot_variance_at_window_open() {
        // The headline result: a TWAP-settled contract at the moment its window
        // opens carries one third the variance of a spot-settled one.
        let w = 60.0;
        assert!((variance_shape(w, w) - w / 3.0).abs() < 1e-9);
        assert!((variance_shape(w, 0.0) - w).abs() < 1e-9);
    }

    #[test]
    fn variance_collapses_cubically_inside_the_window() {
        let w = 60.0;
        let at_open = variance_shape(60.0, w);
        let half = variance_shape(30.0, w);
        let ten_left = variance_shape(10.0, w);

        // Halfway through: (1/2)^3 = 1/8 of the variance at open.
        assert!((half / at_open - 0.125).abs() < 1e-9);
        // Standard deviation ratio 0.354, NOT the 0.707 a sqrt(t) model gives.
        assert!(((half / at_open).sqrt() - 0.353_553_39).abs() < 1e-6);

        // Ten seconds left: (1/6)^3 = 1/216.
        assert!((ten_left / at_open - 1.0 / 216.0).abs() < 1e-9);
        // sd ratio 0.068, versus 0.408 for a sqrt(t) model — a factor of 6.
        assert!(((ten_left / at_open).sqrt() - 0.068_041_38).abs() < 1e-6);
    }

    #[test]
    fn mispricing_versus_a_naive_sqrt_tau_model_is_large_and_directional() {
        // Concrete scenario: BTC is $30 above the strike with 10 seconds left on
        // a 60-second window, and the observed part of the window averaged
        // exactly at the strike.
        let mut inp = base();
        inp.spot = 65_030.0;
        inp.tau = 10.0;
        inp.observed_avg = 65_012.0;

        let f = fair(&inp);

        // Correct model: mean = (50/60)*65012 + (10/60)*65030 = 65015.0
        assert!((f.mean - 65_015.0).abs() < 1e-6);
        // sd = 4.63 * sqrt(1000 / (3*3600)) = 4.63 * 0.30429 = 1.409
        assert!((f.sd - 1.4089).abs() < 1e-3, "sd = {}", f.sd);
        // z = 15 / 1.409 = 10.6 -> effectively decided.
        assert!(f.p > 0.999_999, "p = {}", f.p);
        assert!(f.decided);

        // The naive comparison: spot-settled, sqrt(tau) decay, ignoring the
        // window entirely. sd = 4.63*sqrt(10) = 14.64, z = 30/14.64 = 2.05,
        // p = 0.980. It would quote ~98c into a market worth ~100c.
        let naive_sd = inp.sigma * inp.tau.sqrt();
        let naive_p = norm_cdf((inp.spot - inp.strike) / naive_sd);
        assert!((naive_p - 0.9799).abs() < 1e-3, "naive p = {}", naive_p);
        assert!(f.p - naive_p > 0.019);
    }

    #[test]
    fn observed_average_anchors_the_estimate() {
        // With 1 second left the settlement value is 59/60 already locked in.
        let mut inp = base();
        inp.tau = 1.0;
        inp.spot = 66_000.0;
        inp.observed_avg = 64_900.0;
        let f = fair(&inp);
        // mean = (59/60)*64900 + (1/60)*66000 = 64918.33
        assert!((f.mean - 64_918.333).abs() < 1e-2, "mean = {}", f.mean);
        // Despite spot being $1000 above the strike, the contract is worthless:
        // the average cannot recover in one second.
        assert!(f.p < 1e-9, "p = {}", f.p);
    }

    #[test]
    fn delta_has_the_right_sign_and_shrinks_inside_the_window() {
        let mut inp = base();
        inp.tau = 120.0;
        let before = fair(&inp).d_spot;
        inp.tau = 30.0;
        inp.observed_avg = inp.spot;
        let inside = fair(&inp).d_spot;
        assert!(before > 0.0 && inside > 0.0);
        // At the money, delta actually *rises* as sd collapses, because the
        // density concentrates. Both are positive; that is the invariant we
        // depend on for the hedge ratio.
        assert!(inside > before);
    }

    #[test]
    fn delta_matches_a_finite_difference() {
        let mut inp = base();
        inp.spot = 65_020.0;
        inp.tau = 45.0;
        inp.observed_avg = 65_005.0;
        let f = fair(&inp);

        let h = 0.01;
        let mut up = inp;
        up.spot += h;
        let mut dn = inp;
        dn.spot -= h;
        let fd = (fair(&up).p - fair(&dn).p) / (2.0 * h);
        assert!(
            (f.d_spot - fd).abs() < 1e-6,
            "analytic {} vs fd {}",
            f.d_spot,
            fd
        );
    }

    #[test]
    fn vega_matches_a_finite_difference() {
        let mut inp = base();
        inp.spot = 65_040.0;
        inp.tau = 200.0;
        let f = fair(&inp);

        let h = 1e-4;
        let mut up = inp;
        up.sigma += h;
        let mut dn = inp;
        dn.sigma -= h;
        let fd = (fair(&up).p - fair(&dn).p) / (2.0 * h);
        assert!(
            (f.d_sigma - fd).abs() < 1e-6,
            "analytic {} vs fd {}",
            f.d_sigma,
            fd
        );
    }

    #[test]
    fn stale_spot_inflates_model_uncertainty() {
        let mut inp = base();
        inp.spot = 65_030.0;
        let fresh = fair(&inp);
        inp.spot_age = 2.0;
        let stale = fair(&inp);
        assert!(stale.sigma_p > fresh.sigma_p * 1.5);
        // The point estimate is unchanged; only our confidence in it moves.
        assert!((stale.p - fresh.p).abs() < 1e-12);
    }

    #[test]
    // `tau == 0.0` collapses `sd` to exactly `0.0`, which `fair` handles
    // with an explicit `p = if mean > strike { 1.0 } else { 0.0 }` — a
    // literal, not a computed value.
    #[allow(clippy::float_cmp)]
    fn expired_market_is_decided() {
        let mut inp = base();
        inp.tau = 0.0;
        inp.observed_avg = 65_100.0;
        let f = fair(&inp);
        assert!(f.decided);
        assert_eq!(f.p, 1.0);
    }

    #[test]
    fn spot_settlement_falls_back_to_plain_brownian() {
        let mut inp = base();
        inp.window = 0.0;
        inp.tau = 100.0;
        let f = fair(&inp);
        assert!((f.sd - inp.sigma * 10.0).abs() < 1e-9);
    }

    #[test]
    fn probability_is_monotone_in_spot() {
        let mut inp = base();
        inp.tau = 90.0;
        let mut prev = -1.0;
        let mut s = 64_800.0;
        while s < 65_200.0 {
            inp.spot = s;
            let p = fair(&inp).p;
            assert!(p >= prev - 1e-12, "not monotone at spot {s}");
            prev = p;
            s += 1.0;
        }
    }
}
