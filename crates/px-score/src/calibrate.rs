//! Recalibration — turning information into usable probabilities.
//!
//! # What this fixes and what it cannot
//!
//! The Murphy decomposition splits a Brier score into `reliability −
//! resolution + uncertainty`. Resolution is information: does the model
//! separate outcomes at all? Reliability is calibration: when it says 70%, does
//! that happen 70% of the time?
//!
//! **Only one of those is fixable after the fact.** A monotone map applied to
//! the forecasts can drive reliability to nearly zero while leaving resolution
//! untouched, because a monotone map does not change the *ranking* of
//! forecasts. So:
//!
//! * A model with resolution and poor reliability is a calibration bug. This
//!   module fixes it.
//! * A model with no resolution knows nothing, and no post-processing invents
//!   information. This module cannot help, and will honestly report that it
//!   did not.
//!
//! # Isotonic regression, and why not a sigmoid
//!
//! Platt scaling fits a two-parameter logistic. It is well-behaved on small
//! samples but it *assumes* the miscalibration is sigmoidal, which is not a
//! safe assumption to make blind — the observed error shape on this engine's
//! own forecasts is not sigmoidal (see `docs/ACCURACY.md`).
//!
//! Isotonic regression fits an arbitrary monotone step function by pool
//! adjacent violators. It is non-parametric, so it can follow that shape — at
//! the cost of needing more data and of overfitting cheerfully if you let it
//! see the test set.
//!
//! # The discipline that makes this honest
//!
//! Recalibration fitted and evaluated on the same forecasts *always* improves
//! the score, and the improvement is entirely fictitious. Worse, the split must
//! be by **session**, not by forecast: forecasts within one market resolve
//! against one shared outcome, so a random split puts near-duplicate
//! observations on both sides and leaks the answer.
//!
//! `fit` therefore takes only training data and `Scorer` scores only held-out
//! data. The API makes the honest thing the easy thing.

use crate::{Resolved, Scorer};

/// A fitted monotone calibration map.
#[derive(Clone, Debug, Default)]
pub struct Calibrator {
    /// Breakpoints of the fitted step function: (forecast, calibrated).
    /// Sorted ascending by forecast, values non-decreasing.
    knots: Vec<(f64, f64)>,
}

impl Calibrator {
    /// Fit by pool-adjacent-violators on `(forecast, outcome)` pairs.
    ///
    /// PAVA finds the monotone non-decreasing function minimising squared error
    /// to the outcomes — exactly the right objective, since Brier score *is*
    /// squared error.
    pub fn fit(training: &[Resolved]) -> Calibrator {
        let mut pts: Vec<(f64, f64)> = training
            .iter()
            .filter(|r| r.forecast.model_p.is_finite())
            .map(|r| {
                (
                    r.forecast.model_p.clamp(0.0, 1.0),
                    if r.outcome { 1.0 } else { 0.0 },
                )
            })
            .collect();
        if pts.len() < 20 {
            return Calibrator::default(); // identity
        }
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Blocks of (sum_of_outcomes, count, max_forecast_in_block).
        let mut sum: Vec<f64> = Vec::with_capacity(pts.len());
        let mut cnt: Vec<f64> = Vec::with_capacity(pts.len());
        let mut xhi: Vec<f64> = Vec::with_capacity(pts.len());

        for &(x, y) in &pts {
            sum.push(y);
            cnt.push(1.0);
            xhi.push(x);
            // Pool while the previous block's mean exceeds this one's: the
            // violation of monotonicity that gives the algorithm its name.
            while sum.len() >= 2 {
                let n = sum.len();
                let (s1, c1) = match (sum.get(n - 2), cnt.get(n - 2)) {
                    (Some(a), Some(b)) => (*a, *b),
                    _ => break,
                };
                let (s2, c2) = match (sum.get(n - 1), cnt.get(n - 1)) {
                    (Some(a), Some(b)) => (*a, *b),
                    _ => break,
                };
                if s1 / c1 <= s2 / c2 {
                    break;
                }
                let x_last = xhi.get(n - 1).copied().unwrap_or(x);
                sum.truncate(n - 2);
                cnt.truncate(n - 2);
                xhi.truncate(n - 2);
                sum.push(s1 + s2);
                cnt.push(c1 + c2);
                xhi.push(x_last);
            }
        }

        let mut knots = Vec::with_capacity(sum.len());
        for i in 0..sum.len() {
            let (s, c, x) = match (sum.get(i), cnt.get(i), xhi.get(i)) {
                (Some(a), Some(b), Some(c)) => (*a, *b, *c),
                _ => continue,
            };
            if c > 0.0 {
                knots.push((x, s / c));
            }
        }
        Calibrator { knots }
    }

    /// Apply the map. Linear interpolation between knots; clamped outside.
    pub fn apply(&self, p: f64) -> f64 {
        // `f64::clamp` propagates NaN — it does not clamp it. Guarding with
        // `!p.is_finite()` and then *returning the clamp* was therefore no
        // guard at all: NaN went in and NaN came out, straight into a
        // probability. Match the crate's convention and fall back to the
        // uninformative 0.5.
        if !p.is_finite() {
            return 0.5;
        }
        if self.knots.is_empty() {
            return p.clamp(0.0, 1.0);
        }
        let p = p.clamp(0.0, 1.0);
        let first = self.knots.first().copied().unwrap_or((0.0, 0.0));
        let last = self.knots.last().copied().unwrap_or((1.0, 1.0));
        if p <= first.0 {
            return first.1;
        }
        if p >= last.0 {
            return last.1;
        }
        // Binary search for the bracketing pair.
        let mut lo = 0usize;
        let mut hi = self.knots.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            match self.knots.get(mid) {
                Some(&(x, _)) if x <= p => lo = mid,
                _ => hi = mid,
            }
        }
        let (x0, y0) = self.knots.get(lo).copied().unwrap_or(first);
        let (x1, y1) = self.knots.get(hi).copied().unwrap_or(last);
        if (x1 - x0).abs() < 1e-12 {
            return y1;
        }
        y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    }

    /// Rewrite a set of forecasts through the map, leaving the venue baseline
    /// alone so the comparison stays fair.
    pub fn transform(&self, rs: &[Resolved]) -> Vec<Resolved> {
        rs.iter()
            .map(|r| {
                let mut out = *r;
                out.forecast.model_p = self.apply(r.forecast.model_p);
                out
            })
            .collect()
    }

    pub fn knot_count(&self) -> usize {
        self.knots.len()
    }

    pub fn is_identity(&self) -> bool {
        self.knots.is_empty()
    }
}

/// Result of an honest train/test recalibration experiment.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationResult {
    pub train_n: usize,
    pub test_n: usize,
    /// Skill on held-out data, before recalibration.
    pub skill_before: f64,
    /// Skill on held-out data, after.
    pub skill_after: f64,
    pub reliability_before: f64,
    pub reliability_after: f64,
    /// Resolution should be roughly unchanged — a monotone map preserves
    /// ranking. If it moves a lot, something is wrong with the fit.
    pub resolution_before: f64,
    pub resolution_after: f64,
    pub t_after: f64,
}

impl CalibrationResult {
    pub fn verdict(&self) -> &'static str {
        if self.skill_after > 0.0 && self.t_after < -2.0 {
            "RECALIBRATION WORKS — held-out skill is positive and significant"
        } else if self.skill_after > self.skill_before && self.skill_after > 0.0 {
            "PROMISING — positive out of sample, not yet significant"
        } else if self.skill_after > self.skill_before {
            "IMPROVED BUT STILL NEGATIVE — calibration was part of it, not all"
        } else {
            "NO — the problem is not calibration"
        }
    }
}

/// Fit on the first half of the sessions, evaluate on the second.
///
/// `session_of` maps a forecast to its session, so the split never puts two
/// forecasts sharing an outcome on opposite sides. Splitting by forecast
/// instead would leak the answer and report a fictitious improvement.
pub fn train_test_split<F>(all: &[Resolved], session_of: F) -> CalibrationResult
where
    F: Fn(&Resolved) -> u64,
{
    let mut sessions: Vec<u64> = all.iter().map(&session_of).collect();
    sessions.sort_unstable();
    sessions.dedup();
    let cut = sessions.len() / 2;
    let boundary = sessions.get(cut).copied().unwrap_or(u64::MAX);

    let train: Vec<Resolved> = all
        .iter()
        .filter(|r| session_of(r) < boundary)
        .copied()
        .collect();
    let test: Vec<Resolved> = all
        .iter()
        .filter(|r| session_of(r) >= boundary)
        .copied()
        .collect();

    let cal = Calibrator::fit(&train);
    let adjusted = cal.transform(&test);

    let mut before = Scorer::new();
    for r in &test {
        before.record(*r);
    }
    let mut after = Scorer::new();
    for r in &adjusted {
        after.record(*r);
    }
    let b = before.score();
    let a = after.score();

    CalibrationResult {
        train_n: train.len(),
        test_n: test.len(),
        skill_before: b.skill_score,
        skill_after: a.skill_score,
        reliability_before: b.reliability,
        reliability_after: a.reliability,
        resolution_before: b.resolution,
        resolution_after: a.resolution,
        t_after: a.t_stat_clustered,
    }
}

pub fn report(r: &CalibrationResult) -> String {
    format!(
        "Recalibration, held out by session\n\
         \x20 train {} forecasts   test {} forecasts\n\n\
         \x20                 before      after\n\
         \x20 skill vs mid  {:>+8.4}   {:>+8.4}\n\
         \x20 reliability   {:>8.5}   {:>8.5}   (want lower)\n\
         \x20 resolution    {:>8.5}   {:>8.5}   (want unchanged)\n\
         \x20 t-stat after (clustered) {:>+8.2}\n\n\
         \x20 {}\n",
        r.train_n,
        r.test_n,
        r.skill_before,
        r.skill_after,
        r.reliability_before,
        r.reliability_after,
        r.resolution_before,
        r.resolution_after,
        r.t_after,
        r.verdict()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Forecast;

    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn r(model_p: f64, venue_p: f64, outcome: bool, session: u64) -> Resolved {
        Resolved {
            forecast: Forecast {
                t_s: session as f64,
                model_p,
                venue_p,
                horizon_s: 60.0,
                cluster_id: session,
            },
            outcome,
        }
    }

    #[test]
    fn identity_when_starved_of_data() {
        let c = Calibrator::fit(&[]);
        assert!(c.is_identity());
        assert!((c.apply(0.37) - 0.37).abs() < 1e-12);
    }

    #[test]
    fn fitted_map_is_monotone() {
        let mut rng = Lcg(3);
        let train: Vec<Resolved> = (0..4000)
            .map(|i| {
                let p = rng.u();
                r(p, 0.5, rng.u() < p, i)
            })
            .collect();
        let c = Calibrator::fit(&train);
        let mut prev = -1.0;
        let mut x = 0.0;
        while x <= 1.0 {
            let y = c.apply(x);
            assert!(y >= prev - 1e-9, "not monotone at {x}: {y} < {prev}");
            assert!((0.0..=1.0).contains(&y), "out of range at {x}: {y}");
            prev = y;
            x += 0.001;
        }
    }

    #[test]
    fn recovers_a_known_distortion() {
        // Truth occurs at rate p; the model reports a squashed version. The
        // fitted map should undo the squash.
        let mut rng = Lcg(11);
        let train: Vec<Resolved> = (0..20_000)
            .map(|i| {
                let truth = rng.u();
                let reported = truth * truth; // heavily distorted, still monotone
                r(reported, 0.5, rng.u() < truth, i)
            })
            .collect();
        let c = Calibrator::fit(&train);
        // Reported 0.25 corresponds to truth 0.5.
        assert!((c.apply(0.25) - 0.5).abs() < 0.05, "got {}", c.apply(0.25));
        assert!((c.apply(0.81) - 0.9).abs() < 0.05, "got {}", c.apply(0.81));
    }

    #[test]
    fn overconfidence_is_repaired_out_of_sample() {
        // The failure mode actually observed: informative but overconfident.
        let mut rng = Lcg(23);
        let all: Vec<Resolved> = (0..40_000)
            .map(|i| {
                let truth = rng.u();
                let pushed = (0.5 + (truth - 0.5) * 1.9).clamp(0.0, 1.0);
                r(pushed, truth, rng.u() < truth, i / 100)
            })
            .collect();
        let res = train_test_split(&all, |x| x.forecast.t_s as u64 / 100);

        assert!(
            res.reliability_after < res.reliability_before * 0.5,
            "reliability {} -> {}",
            res.reliability_before,
            res.reliability_after
        );
        // A monotone map preserves ranking, so resolution should barely move.
        assert!(
            (res.resolution_after - res.resolution_before).abs() < 0.02,
            "resolution moved {} -> {}",
            res.resolution_before,
            res.resolution_after
        );
        assert!(res.skill_after > res.skill_before);
    }

    #[test]
    fn cannot_invent_information_that_is_not_there() {
        // A constant forecast carries zero resolution. Recalibration should
        // report honestly that it did not help.
        let mut rng = Lcg(31);
        let all: Vec<Resolved> = (0..20_000)
            .map(|i| r(0.5, rng.u(), rng.u() < 0.4, i / 100))
            .collect();
        let res = train_test_split(&all, |x| x.forecast.t_s as u64 / 100);
        assert!(res.resolution_after < 0.01, "invented resolution");
        assert!(res.verdict().starts_with("NO"), "{}", res.verdict());
    }

    #[test]
    fn the_split_is_by_session_not_by_forecast() {
        // Every forecast in a session shares one outcome. A correct split puts
        // all of a session on one side.
        let all: Vec<Resolved> = (0..600)
            .map(|i| {
                let session = i / 100;
                r(0.5, 0.5, session % 2 == 0, session)
            })
            .collect();
        let res = train_test_split(&all, |x| x.forecast.t_s as u64);
        assert_eq!(res.train_n + res.test_n, 600);
        assert_eq!(
            res.train_n % 100,
            0,
            "a session was split across the boundary"
        );
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let _ = train_test_split(&[], |_| 0);
        let one = vec![r(0.5, 0.5, true, 0)];
        let _ = train_test_split(&one, |_| 0);
        let c = Calibrator::fit(&one);
        assert!(c.apply(f64::NAN).is_finite());
    }
}
