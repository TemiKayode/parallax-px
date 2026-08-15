//! Volatility estimation and settlement-window accumulation.
//!
//! Both structures here are updated on the *reference feed* tick path (one
//! update per underlying print), not the market tick path. That split matters:
//! a BTC market may see thousands of book deltas between two spot prints, and
//! recomputing volatility on each of them would be pure waste.

use px_core::math::{exp, ln};

/// Time-aware EWMA volatility estimator.
///
/// Reference feeds do not arrive on a fixed grid, so a plain per-sample EWMA
/// would silently weight a burst of ten ticks in one millisecond the same as
/// ten ticks spread over a second. We decay by elapsed *time*, and we estimate
/// the variance *rate* (`r^2 / dt`) rather than the per-sample variance.
#[derive(Clone, Copy, Debug)]
pub struct EwmaVol {
    /// Half-life of the decay, in seconds.
    half_life: f64,
    /// Current estimate of variance per second, in squared relative returns.
    var_rate: f64,
    last_price: f64,
    last_ts: f64,
    /// Effective sample count, saturating. Used for the standard error.
    n_eff: f64,
    initialised: bool,
}

/// Smallest inter-tick gap we will believe. Two prints with an identical
/// timestamp would give `r^2 / 0 = inf`; a 1 ms floor bounds the damage from a
/// duplicated or out-of-order message without discarding real information.
const MIN_DT: f64 = 1e-3;

/// Winsorisation bound. A single return more than this many current standard
/// deviations is clipped before it enters the estimate. Genuine jumps are real
/// and we want to see them, but a decimal-point error in a feed should not be
/// able to move our volatility estimate by an order of magnitude in one tick.
const WINSOR_SD: f64 = 8.0;

impl EwmaVol {
    pub fn new(half_life_s: f64) -> Self {
        EwmaVol {
            half_life: half_life_s,
            var_rate: 0.0,
            last_price: 0.0,
            last_ts: 0.0,
            n_eff: 0.0,
            initialised: false,
        }
    }

    /// Feed a new reference print. `ts` is seconds on the monotonic timebase.
    pub fn update(&mut self, price: f64, ts: f64) {
        if !(price > 0.0) || !price.is_finite() {
            return; // Reject nonsense outright; the feed guard will notice.
        }
        if !self.initialised {
            self.last_price = price;
            self.last_ts = ts;
            self.initialised = true;
            return;
        }

        let dt = (ts - self.last_ts).max(MIN_DT);
        let r = ln(price / self.last_price);
        self.last_price = price;
        self.last_ts = ts;

        if !r.is_finite() {
            return;
        }

        // Winsorise against the current estimate.
        let sd = (self.var_rate * dt).sqrt();
        let r_clipped = if sd > 0.0 && r.abs() > WINSOR_SD * sd {
            WINSOR_SD * sd * r.signum()
        } else {
            r
        };

        let inst = (r_clipped * r_clipped) / dt;
        let decay = exp(-dt * core::f64::consts::LN_2 / self.half_life);

        if self.n_eff == 0.0 {
            self.var_rate = inst;
        } else {
            self.var_rate = decay * self.var_rate + (1.0 - decay) * inst;
        }
        // Effective sample size saturates at the EWMA's own memory length.
        let n_cap = (1.0 + decay) / (1.0 - decay).max(1e-12);
        self.n_eff = (self.n_eff + 1.0).min(n_cap);
    }

    /// Relative volatility, per sqrt(second).
    #[inline(always)]
    pub fn sigma_rel(&self) -> f64 {
        self.var_rate.max(0.0).sqrt()
    }

    /// Absolute volatility in price units, per sqrt(second).
    #[inline(always)]
    pub fn sigma_abs(&self, spot: f64) -> f64 {
        self.sigma_rel() * spot
    }

    /// Relative 1-sigma standard error of the volatility estimate. Feeds
    /// straight into the safety margin: a poorly-determined vol should buy us
    /// less size, not the same size with more hope.
    #[inline(always)]
    pub fn rel_err(&self) -> f64 {
        if self.n_eff < 2.0 {
            1.0
        } else {
            (1.0 / (2.0 * self.n_eff)).sqrt()
        }
    }

    #[inline(always)]
    pub fn is_warm(&self) -> bool {
        self.n_eff >= 30.0
    }

    #[inline(always)]
    pub fn last_price(&self) -> f64 {
        self.last_price
    }

    #[inline(always)]
    pub fn last_ts(&self) -> f64 {
        self.last_ts
    }

    /// Annualised, for humans reading dashboards.
    pub fn sigma_annual(&self) -> f64 {
        self.sigma_rel() * (365.0 * 24.0 * 3600.0f64).sqrt()
    }
}

/// Two-speed volatility with an explicit burst detector.
///
/// The brief asks for "speed and volatility of the recent move" as separate
/// inputs. They are the same quantity measured over two horizons: a fast
/// estimator that reacts to a news shock within seconds, and a slow one that
/// represents the regime. Their ratio is the shock signal.
#[derive(Clone, Copy, Debug)]
pub struct TwoSpeedVol {
    pub fast: EwmaVol,
    pub slow: EwmaVol,
}

impl TwoSpeedVol {
    pub fn new(fast_hl_s: f64, slow_hl_s: f64) -> Self {
        TwoSpeedVol {
            fast: EwmaVol::new(fast_hl_s),
            slow: EwmaVol::new(slow_hl_s),
        }
    }

    #[inline]
    pub fn update(&mut self, price: f64, ts: f64) {
        self.fast.update(price, ts);
        self.slow.update(price, ts);
    }

    /// How much faster the market is moving right now than its recent regime.
    /// 1.0 is calm; 3.0 means a shock is in progress.
    #[inline(always)]
    pub fn burst_ratio(&self) -> f64 {
        let s = self.slow.sigma_rel();
        if s <= 0.0 {
            1.0
        } else {
            (self.fast.sigma_rel() / s).max(1.0)
        }
    }

    /// The volatility we price with: a precision-weighted blend of the two.
    ///
    /// # Why not simply take the larger
    ///
    /// Taking `max(fast, slow)` is the obvious conservative choice and it is
    /// what this code did first. The replay harness showed it to be a losing
    /// one, for a reason worth stating plainly:
    ///
    /// **A conservative volatility is not a conservative price.** Overstating
    /// sigma pulls the fair probability toward 50 cents. Away from the money
    /// that is a systematic *directional* error, not a widening. When the true
    /// value is 72 cents and we say 66 because our sigma is too high, we are
    /// not being cautious — we are quoting an offer at 67 that the rest of the
    /// market is happy to lift all day. Being picked off for six cents is the
    /// same loss whether it came from recklessness or from caution.
    ///
    /// The blend below weights each estimator by its own precision
    /// (`1 / rel_err^2`), which is the minimum-variance combination and is
    /// unbiased. Conservatism belongs in `rel_err`, which widens the safety
    /// margin and stops us quoting at all when the two estimators disagree —
    /// and in `sigma_rel_conservative`, which the inventory penalty uses,
    /// because *there* a high sigma really does mean "hold less".
    #[inline]
    pub fn sigma_rel(&self) -> f64 {
        let sf = self.fast.sigma_rel();
        let ss = self.slow.sigma_rel();
        if sf <= 0.0 {
            return ss;
        }
        if ss <= 0.0 {
            return sf;
        }
        let ef = self.fast.rel_err().max(1e-6);
        let es = self.slow.rel_err().max(1e-6);
        let wf = 1.0 / (ef * ef);
        let ws = 1.0 / (es * es);
        // Combine in variance space, then take the root.
        let var = (wf * sf * sf + ws * ss * ss) / (wf + ws);
        var.max(0.0).sqrt()
    }

    /// The larger of the two. Used where a high volatility genuinely means
    /// "take less risk" — the inventory penalty and the position limits —
    /// rather than "shift the price".
    #[inline(always)]
    pub fn sigma_rel_conservative(&self) -> f64 {
        self.fast.sigma_rel().max(self.slow.sigma_rel())
    }

    #[inline(always)]
    pub fn sigma_abs(&self, spot: f64) -> f64 {
        self.sigma_rel() * spot
    }

    /// Relative standard error, including the disagreement between the two
    /// estimators.
    ///
    /// Sampling error alone understates what we do not know. When the fast
    /// estimator reads three times the slow one, the honest statement is not
    /// "sigma is X plus or minus 10%" — it is that we do not currently know
    /// which regime we are in, and the uncertainty is of order 100%. Folding
    /// the disagreement in makes the safety margin widen during exactly the
    /// episodes where the point estimate is least trustworthy, which stops the
    /// engine quoting through a regime change instead of quoting confidently
    /// and wrongly.
    /// Only *excess* disagreement counts.
    ///
    /// First attempt at this simply added `burst_ratio - 1` to the sampling
    /// error, and it shut the strategy off entirely. The reason is worth
    /// keeping: a fast estimator with a three-second half-life sees on the
    /// order of sixty samples, so its own standard error is around 10-30%. On a
    /// perfectly stationary random walk it will therefore read 1.3x the slow
    /// estimator much of the time *by construction*. Treating that as evidence
    /// of a regime change means declaring a regime change roughly always.
    ///
    /// So the disagreement term is measured against what sampling noise alone
    /// would produce: only the part exceeding two standard errors counts as
    /// information. Below that threshold the two estimators are consistent and
    /// we have learned nothing.
    #[inline]
    pub fn rel_err(&self) -> f64 {
        let fast_err = self.fast.rel_err();
        let slow_err = self.slow.rel_err();
        let sampling = if self.fast.sigma_rel() >= self.slow.sigma_rel() {
            fast_err
        } else {
            slow_err
        };
        // Expected dispersion of the ratio under the null of equal true vols.
        let expected = 2.0 * (fast_err * fast_err + slow_err * slow_err).sqrt();
        let excess = (self.burst_ratio() - 1.0 - expected).max(0.0);
        (sampling * sampling + excess * excess).sqrt()
    }

    #[inline(always)]
    pub fn is_warm(&self) -> bool {
        self.slow.is_warm()
    }
}

/// Integrates the reference price over a market's settlement window so we know
/// how much of the settlement TWAP is already locked in.
///
/// We compute this ourselves from the spot feed rather than reading Chainlink's
/// published rolling TWAP, because those are different quantities: the venue's
/// 60-second feed at time `t` averages `[t-60, t]`, whereas settlement needs the
/// average over `[T-60, T]`. Inside the window they overlap only partially. The
/// published feed is still worth consuming — as a cross-check on our integral
/// (see `drift_vs`), which is how we catch a spot feed that has silently
/// diverged.
#[derive(Clone, Copy, Debug)]
pub struct TwapAccumulator {
    /// Window start on the monotonic timebase, in seconds.
    start: f64,
    /// Window end (expiry), in seconds.
    end: f64,
    /// Integral of price*dt accumulated so far.
    integral: f64,
    last_px: f64,
    last_t: f64,
    open: bool,
}

impl TwapAccumulator {
    pub fn new(expiry_s: f64, window_s: f64) -> Self {
        TwapAccumulator {
            start: expiry_s - window_s,
            end: expiry_s,
            integral: 0.0,
            last_px: 0.0,
            last_t: 0.0,
            open: false,
        }
    }

    /// Feed a reference print. Uses left-endpoint (step) integration, matching
    /// how an oracle that samples a last-traded price actually behaves — a
    /// trapezoidal rule would assume linear interpolation between prints that
    /// the oracle never performs.
    pub fn update(&mut self, price: f64, ts: f64) {
        if ts <= self.start {
            // Not yet in the window; remember the price so that when the window
            // opens we already know the prevailing level.
            self.last_px = price;
            self.last_t = self.start;
            return;
        }
        let t = ts.min(self.end);
        if !self.open {
            self.open = true;
            if self.last_px <= 0.0 {
                self.last_px = price;
            }
            self.last_t = self.last_t.max(self.start);
        }
        if t > self.last_t && self.last_px > 0.0 {
            self.integral += self.last_px * (t - self.last_t);
            self.last_t = t;
        }
        self.last_px = price;
    }

    /// Average of prices observed so far inside the window. Returns the last
    /// known price if the window has not opened, which is the correct prior.
    pub fn observed_avg(&self) -> f64 {
        let elapsed = self.last_t - self.start;
        if elapsed <= 0.0 || self.integral <= 0.0 {
            self.last_px
        } else {
            self.integral / elapsed
        }
    }

    /// Fraction of the settlement window already observed, in `[0, 1]`.
    pub fn elapsed_fraction(&self) -> f64 {
        let w = self.end - self.start;
        if w <= 0.0 {
            1.0
        } else {
            ((self.last_t - self.start) / w).clamp(0.0, 1.0)
        }
    }

    /// Relative difference between our integral and an externally published
    /// TWAP over a comparable window. A persistent non-zero value means our
    /// spot feed and the settlement oracle disagree, which is a reason to stop
    /// quoting rather than a reason to trade the difference.
    pub fn drift_vs(&self, published: f64) -> f64 {
        let ours = self.observed_avg();
        if ours <= 0.0 || published <= 0.0 {
            0.0
        } else {
            (ours - published) / published
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_recovers_a_known_volatility() {
        // Generate a deterministic walk with a known per-step relative move.
        let mut v = EwmaVol::new(30.0);
        let step = 0.0002; // 2 bp per 100 ms
        let dt = 0.1;
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..4000 {
            // Alternating +-, so |r| is exactly `step` every step.
            px *= if i % 2 == 0 {
                1.0 + step
            } else {
                1.0 / (1.0 + step)
            };
            t += dt;
            v.update(px, t);
        }
        // Variance rate should converge to step^2 / dt.
        let expected = (step * step) / dt;
        let got = v.sigma_rel() * v.sigma_rel();
        assert!(
            (got / expected - 1.0).abs() < 0.05,
            "expected var rate {expected}, got {got}"
        );
        assert!(v.is_warm());
    }

    #[test]
    fn ewma_survives_duplicate_timestamps() {
        let mut v = EwmaVol::new(30.0);
        v.update(65_000.0, 1.0);
        for _ in 0..100 {
            v.update(65_001.0, 1.0); // same timestamp, repeatedly
        }
        assert!(v.sigma_rel().is_finite());
        assert!(v.sigma_rel() < 1.0);
    }

    #[test]
    fn ewma_rejects_nonsense_prices() {
        let mut v = EwmaVol::new(30.0);
        v.update(65_000.0, 1.0);
        v.update(65_010.0, 2.0);
        let before = v.sigma_rel();
        v.update(-1.0, 3.0);
        v.update(0.0, 4.0);
        v.update(f64::NAN, 5.0);
        assert_eq!(v.sigma_rel(), before);
    }

    #[test]
    fn winsorisation_bounds_a_fat_finger_print() {
        let mut v = EwmaVol::new(60.0);
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..2000 {
            px *= if i % 2 == 0 { 1.0001 } else { 0.9999 };
            t += 0.1;
            v.update(px, t);
        }
        let before = v.sigma_rel();
        // A print off by a factor of ten.
        t += 0.1;
        v.update(px * 10.0, t);
        let after = v.sigma_rel();
        // Without clipping this single tick would raise sigma by ~100x.
        assert!(after < before * 12.0, "before {before}, after {after}");
    }

    #[test]
    fn rel_err_falls_as_samples_accumulate() {
        let mut v = EwmaVol::new(60.0);
        assert_eq!(v.rel_err(), 1.0);
        let mut px = 65_000.0;
        let mut t = 0.0;
        let mut prev = 1.0;
        for i in 0..500 {
            px *= if i % 2 == 0 { 1.0001 } else { 0.9999 };
            t += 0.1;
            v.update(px, t);
            let e = v.rel_err();
            assert!(e <= prev + 1e-12);
            prev = e;
        }
        assert!(prev < 0.1);
    }

    #[test]
    fn burst_ratio_spikes_on_a_shock_then_decays() {
        let mut v = TwoSpeedVol::new(2.0, 120.0);
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..3000 {
            px *= if i % 2 == 0 { 1.00002 } else { 0.99998 };
            t += 0.05;
            v.update(px, t);
        }
        let calm = v.burst_ratio();
        assert!(calm < 1.5, "calm ratio {calm}");

        // News shock: twenty consecutive large moves.
        for i in 0..20 {
            px *= if i % 2 == 0 { 1.002 } else { 0.9985 };
            t += 0.05;
            v.update(px, t);
        }
        let shocked = v.burst_ratio();
        assert!(shocked > 3.0, "shock ratio {shocked}");

        // The pricing vol rose, but not all the way to the fast estimate.
        assert!(v.sigma_rel() > v.slow.sigma_rel());
        assert!(v.sigma_rel() <= v.sigma_rel_conservative());
    }

    #[test]
    fn the_pricing_volatility_is_a_blend_not_the_maximum() {
        // Regression guard for the bug the replay harness surfaced: pricing off
        // max(fast, slow) biases fair value toward 50 cents and turns a
        // "conservative" estimator into a systematic directional error.
        let mut v = TwoSpeedVol::new(2.0, 120.0);
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..4000 {
            px *= if i % 2 == 0 { 1.00002 } else { 0.99998 };
            t += 0.05;
            v.update(px, t);
        }
        for i in 0..40 {
            px *= if i % 2 == 0 { 1.0015 } else { 0.9988 };
            t += 0.05;
            v.update(px, t);
        }
        let blend = v.sigma_rel();
        let cons = v.sigma_rel_conservative();
        assert!(blend < cons, "blend {blend} should sit below max {cons}");
        assert!(
            blend > v.slow.sigma_rel(),
            "blend should still react upward"
        );
    }

    #[test]
    fn a_stationary_walk_does_not_look_like_a_regime_change() {
        // Regression guard. Sampling noise in the fast estimator must not be
        // mistaken for information; if it is, the safety margin balloons and
        // the engine stops quoting on a perfectly ordinary tape.
        let mut v = TwoSpeedVol::new(3.0, 180.0);
        let mut seed = 20_260_814u64;
        let mut px = 65_000.0;
        let mut t = 0.0;
        let mut worst: f64 = 0.0;
        for i in 0..8000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            px *= 1.0 + 0.00008 * u;
            t += 0.05;
            v.update(px, t);
            if i > 4000 {
                worst = worst.max(v.rel_err());
            }
        }
        assert!(
            worst < 0.6,
            "stationary walk produced rel_err up to {worst}, which would silence quoting"
        );
    }

    #[test]
    fn disagreement_between_estimators_widens_the_error_bar() {
        let mut v = TwoSpeedVol::new(2.0, 120.0);
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..4000 {
            px *= if i % 2 == 0 { 1.00002 } else { 0.99998 };
            t += 0.05;
            v.update(px, t);
        }
        let calm_err = v.rel_err();
        assert!(calm_err < 0.5, "calm rel_err {calm_err}");

        for i in 0..30 {
            px *= if i % 2 == 0 { 1.003 } else { 0.9975 };
            t += 0.05;
            v.update(px, t);
        }
        let shocked_err = v.rel_err();
        assert!(
            shocked_err > calm_err * 3.0,
            "calm {calm_err} shocked {shocked_err}"
        );
    }

    #[test]
    fn twap_accumulator_averages_a_step_function() {
        // Window [240, 300]. Price is 100 for the first 30s, 200 for the next 30s.
        let mut a = TwapAccumulator::new(300.0, 60.0);
        a.update(100.0, 200.0); // before the window
        a.update(100.0, 240.0);
        a.update(200.0, 270.0);
        a.update(200.0, 300.0);
        // Step integration: 100 over [240,270), 200 over [270,300) -> 150.
        assert!(
            (a.observed_avg() - 150.0).abs() < 1e-9,
            "{}",
            a.observed_avg()
        );
        assert!((a.elapsed_fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn twap_accumulator_reports_partial_progress() {
        let mut a = TwapAccumulator::new(300.0, 60.0);
        a.update(100.0, 240.0);
        a.update(100.0, 255.0);
        assert!((a.elapsed_fraction() - 0.25).abs() < 1e-12);
        assert!((a.observed_avg() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn twap_accumulator_before_window_returns_last_price() {
        let mut a = TwapAccumulator::new(300.0, 60.0);
        a.update(64_900.0, 10.0);
        assert_eq!(a.elapsed_fraction(), 0.0);
        assert_eq!(a.observed_avg(), 64_900.0);
    }

    #[test]
    fn twap_accumulator_ignores_prints_after_expiry() {
        let mut a = TwapAccumulator::new(300.0, 60.0);
        a.update(100.0, 240.0);
        a.update(100.0, 300.0);
        let settled = a.observed_avg();
        a.update(999_999.0, 400.0);
        assert!((a.observed_avg() - settled).abs() < 1e-9);
    }

    #[test]
    fn drift_detects_a_diverging_feed() {
        let mut a = TwapAccumulator::new(300.0, 60.0);
        a.update(65_000.0, 240.0);
        a.update(65_000.0, 270.0);
        assert!(a.drift_vs(65_000.0).abs() < 1e-12);
        // Our feed says 65000, the oracle says 65065: 10 bp of disagreement.
        assert!((a.drift_vs(65_065.0) + 0.000_999).abs() < 1e-5);
    }
}
