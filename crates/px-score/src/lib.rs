//! `px-score` — does the model actually beat the market?
//!
//! # Why this crate exists, and why it comes before everything else
//!
//! The entire premise of this system is one sentence: *our fair value is closer
//! to the truth than the venue's mid.* Every other component — the depth
//! walker, the inventory penalty, the rate-limit governor, the whole execution
//! path — is machinery for converting that claim into money. If the claim is
//! false, none of the machinery matters, and a beautifully engineered system
//! will lose capital smoothly and correctly.
//!
//! The claim is directly testable **without placing a single order**. Log what
//! the model said, log what the venue said, wait for the market to resolve, and
//! score both against the outcome with a proper scoring rule. No execution
//! risk, no capital at risk, no signing, no adapters — just arithmetic on
//! recorded predictions.
//!
//! That makes this the cheapest and most decisive experiment available, and the
//! reason it belongs before connectivity work rather than after: connectivity
//! converts an open question into a funded position, whereas this answers it.
//!
//! # The number that decides it
//!
//! **Brier Skill Score against the venue mid.**
//!
//! ```text
//!   BSS = 1 - BrierScore(model) / BrierScore(venue_mid)
//! ```
//!
//! * `BSS > 0` — the model carries information the market price does not. There
//!   is something here. How much is a separate question.
//! * `BSS <= 0` — the model is worse than reading the price off the screen.
//!   There is no edge, and no amount of latency engineering creates one.
//!
//! A proper scoring rule is essential here: it is uniquely maximised by
//! reporting your true belief, so a model cannot score well by being
//! systematically confident or systematically hedged. Brier is proper. "How
//! often were we directionally right" is not, and will happily reward a model
//! that says 51% every time.
//!
//! # Calibration is not the same as skill
//!
//! The Murphy decomposition splits the Brier score into three parts:
//!
//! ```text
//!   Brier = Reliability - Resolution + Uncertainty
//! ```
//!
//! * **Reliability** (lower better) — when we say 70%, does it happen 70% of
//!   the time? Poor reliability is *miscalibration*, and it is fixable after the
//!   fact by shrinking forecasts toward the base rate.
//! * **Resolution** (higher better) — do our forecasts separate outcomes at all,
//!   or do we say the same thing regardless? This is genuine information, and it
//!   cannot be manufactured by post-processing.
//! * **Uncertainty** — a property of the market, not of us. Nothing to do with
//!   model quality; it is there so the three terms add up.
//!
//! The practical consequence: **poor reliability with good resolution is a
//! calibration bug worth fixing. Good reliability with zero resolution is a
//! model that knows nothing and says so politely.** Only the second is fatal,
//! and a single Brier number cannot tell you which one you have.
//!
//! # Significance
//!
//! A skill score computed over forty forecasts is a rumour. Because each
//! forecast is scored against the *same* outcome under both models, the
//! comparison is naturally paired, and a paired test on the per-forecast score
//! differences is far more powerful than comparing two aggregates. The `t_stat`
//! below is that test. Treat `|t| < 2` as "no evidence either way", regardless
//! of how good the point estimate looks.
//!
//! # Clustering — the assumption that test actually needs
//!
//! `t_stat` treats every `Resolved` forecast as an independent draw. It is
//! not. `px-engine`'s replay harness logs one forecast per second *within* a
//! single simulated market run, and the standard use of this crate pools many
//! such runs together (`px-replay` pools 60 seeds). Forecasts a second apart
//! in the same run share the same spot path and the same fair-value state —
//! they are correlated, not independent — and forecasts from different seeds
//! are not. Feeding 12,000 correlated observations into a formula that
//! assumes 12,000 independent ones does not make the evidence stronger; it
//! makes the standard error smaller by construction, which is a different
//! thing and the wrong thing. A naive `t_stat` of −40 from 60 seeds is not
//! forty times more significant than one seed's worth of evidence — it is
//! roughly *one* seed's worth of evidence wearing the standard error of
//! twelve thousand.
//!
//! `t_stat_clustered` (Cameron & Miller 2015's CR1 sandwich estimator,
//! specialised to a simple mean) fixes this by treating each `cluster_id` —
//! in practice, each seed/run — as the unit of independence, and is the
//! statistic `has_edge()` actually gates on. `t_stat` is kept alongside it,
//! unlabelled as anything but what it is, specifically so the gap between the
//! two is visible rather than hidden: `report()` prints both, and the gap
//! *is* the finding when it is large. When every forecast is its own
//! cluster — genuinely independent data — the two are identical by
//! construction (`clustered_se_degenerates_to_naive_se_when_every_forecast_is_its_own_cluster`
//! below is that identity, checked).

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod calibrate;

use px_core::math::ln;

/// Number of calibration bins across `[0, 1]`.
pub const BINS: usize = 10;

/// Probabilities are clamped this far from 0 and 1 before taking a log. An
/// unclamped log loss is infinite the first time a confident forecast is wrong,
/// which destroys the average and tells you nothing you did not already know.
const EPS: f64 = 1e-6;

/// One prediction, paired with the market's own view at the same instant.
#[derive(Clone, Copy, Debug)]
pub struct Forecast {
    /// When the forecast was made, seconds on the session timebase.
    pub t_s: f64,
    /// The model's fair probability of YES.
    pub model_p: f64,
    /// The venue's implied probability at the same instant — the baseline we
    /// must beat. Usually the mid.
    pub venue_p: f64,
    /// Seconds between the forecast and resolution.
    pub horizon_s: f64,
    /// The unit of statistical independence: every forecast sharing a
    /// `cluster_id` is assumed correlated, and only forecasts in different
    /// clusters are assumed independent. In practice this is the simulated
    /// run or seed a forecast came from — see the module doc's "Clustering"
    /// section. Two forecasts that are actually independent may safely
    /// share a `cluster_id` of 0 each with no other forecasts, or each get
    /// a distinct id; what matters is that correlated forecasts share one.
    pub cluster_id: u64,
}

/// A forecast whose outcome is now known.
#[derive(Clone, Copy, Debug)]
pub struct Resolved {
    pub forecast: Forecast,
    /// Did YES occur?
    pub outcome: bool,
}

impl Resolved {
    #[inline]
    fn o(&self) -> f64 {
        if self.outcome {
            1.0
        } else {
            0.0
        }
    }

    /// Squared error of the model forecast.
    #[inline]
    pub fn model_brier(&self) -> f64 {
        let d = clamp01(self.forecast.model_p) - self.o();
        d * d
    }

    /// Squared error of the venue's price.
    #[inline]
    pub fn venue_brier(&self) -> f64 {
        let d = clamp01(self.forecast.venue_p) - self.o();
        d * d
    }
}

#[inline]
fn clamp01(p: f64) -> f64 {
    if p.is_nan() {
        0.5
    } else {
        p.clamp(0.0, 1.0)
    }
}

#[inline]
fn log_loss_one(p: f64, o: f64) -> f64 {
    let p = clamp01(p).clamp(EPS, 1.0 - EPS);
    -(o * ln(p) + (1.0 - o) * ln(1.0 - p))
}

/// One calibration bucket.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bin {
    /// Forecasts falling in this bucket.
    pub n: usize,
    /// Mean forecast probability within the bucket.
    pub mean_forecast: f64,
    /// Observed frequency of YES within the bucket.
    pub observed: f64,
}

impl Bin {
    /// How far off calibration is here. Positive means we were overconfident in
    /// YES: we said more than happened.
    #[inline]
    pub fn error(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mean_forecast - self.observed
        }
    }
}

/// The verdict.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scorecard {
    pub n: usize,
    /// Base rate of YES across all resolved forecasts.
    pub base_rate: f64,

    pub model_brier: f64,
    pub venue_brier: f64,
    /// `1 - model/venue`. **Positive means the model beats the market.**
    pub skill_score: f64,

    pub model_log_loss: f64,
    pub venue_log_loss: f64,

    // --- Murphy decomposition of the model's Brier score ---
    /// Miscalibration. Lower is better; fixable by recalibration.
    pub reliability: f64,
    /// Genuine discriminating information. Higher is better; not fakeable.
    pub resolution: f64,
    /// Irreducible difficulty of the question. A property of the market.
    pub uncertainty: f64,

    /// Mean of the per-forecast paired difference (model − venue). Negative is
    /// good: our squared error is smaller.
    pub mean_diff: f64,
    /// Standard error of that mean, assuming every forecast is an
    /// independent observation. That assumption is almost always false for
    /// pooled replay data — see `se_diff_clustered`, and the module doc's
    /// "Clustering" section, before trusting this one.
    pub se_diff: f64,
    /// Paired t-statistic computed from `se_diff`. Kept for comparison, not
    /// as the significance test to act on — `t_stat_clustered` is.
    pub t_stat: f64,

    /// Number of distinct `cluster_id`s observed. Cluster-robust inference
    /// needs *enough* clusters to trust — `has_edge()` requires at least 20.
    pub n_clusters: usize,
    /// Cluster-robust standard error of `mean_diff` (Cameron & Miller 2015,
    /// CR1, specialised to a simple mean): each `cluster_id` is treated as
    /// one unit of independence rather than each forecast. This is the
    /// standard error that reflects how much *independent* evidence was
    /// actually collected.
    pub se_diff_clustered: f64,
    /// Paired t-statistic computed from `se_diff_clustered`. **Below −2 is
    /// evidence of a real edge.** Above −2, including any positive value, is
    /// not. This is the statistic `has_edge()` gates on.
    pub t_stat_clustered: f64,

    pub bins: [Bin; BINS],
}

impl Scorecard {
    /// Is there evidence of an edge, at roughly the 95% level?
    ///
    /// Deliberately conjunctive: the point estimate must favour the model,
    /// *and* the cluster-robust paired test must clear the threshold, *and*
    /// there must be enough forecasts *and* enough independent clusters for
    /// any of that to mean anything. A good-looking skill score on thirty
    /// observations — or on twelve thousand observations drawn from three
    /// seeds — is the most common way to talk yourself into a losing
    /// strategy.
    #[inline]
    pub fn has_edge(&self) -> bool {
        self.n >= 200
            && self.n_clusters >= 20
            && self.skill_score > 0.0
            && self.t_stat_clustered < -2.0
    }

    /// A one-line verdict fit for a log or a dashboard.
    pub fn verdict(&self) -> &'static str {
        if self.n < 200 {
            "INSUFFICIENT DATA — need 200+ resolved forecasts"
        } else if self.n_clusters < 20 {
            "INSUFFICIENT CLUSTERS — need 20+ independent seeds/runs, not just forecasts"
        } else if self.t_stat_clustered < -2.0 && self.skill_score > 0.0 {
            "EDGE — model beats venue mid, significant (cluster-robust)"
        } else if self.skill_score > 0.0 {
            "INCONCLUSIVE — model ahead but within noise, once clustering is accounted for"
        } else if self.resolution > self.reliability {
            "NO EDGE — but informative and miscalibrated; try recalibrating"
        } else {
            "NO EDGE — model is not better than reading the price"
        }
    }
}

/// Accumulates forecasts and scores them.
#[derive(Clone, Debug, Default)]
pub struct Scorer {
    resolved: Vec<Resolved>,
}

impl Scorer {
    pub fn new() -> Self {
        Scorer::default()
    }

    /// Record a forecast whose outcome is known.
    pub fn record(&mut self, r: Resolved) {
        if r.forecast.model_p.is_finite() && r.forecast.venue_p.is_finite() {
            self.resolved.push(r);
        }
    }

    pub fn len(&self) -> usize {
        self.resolved.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }

    pub fn as_slice(&self) -> &[Resolved] {
        &self.resolved
    }

    /// Compute the scorecard.
    pub fn score(&self) -> Scorecard {
        let n = self.resolved.len();
        if n == 0 {
            return Scorecard::default();
        }
        let nf = n as f64;

        let mut sum_o = 0.0;
        let mut sum_mb = 0.0;
        let mut sum_vb = 0.0;
        let mut sum_mll = 0.0;
        let mut sum_vll = 0.0;
        let mut sum_d = 0.0;

        // Per-bin accumulators for the Murphy decomposition.
        let mut bin_n = [0usize; BINS];
        let mut bin_p = [0.0f64; BINS];
        let mut bin_o = [0.0f64; BINS];

        for r in &self.resolved {
            let o = r.o();
            let mp = clamp01(r.forecast.model_p);
            let vp = clamp01(r.forecast.venue_p);

            sum_o += o;
            let mb = r.model_brier();
            let vb = r.venue_brier();
            sum_mb += mb;
            sum_vb += vb;
            sum_d += mb - vb;
            sum_mll += log_loss_one(mp, o);
            sum_vll += log_loss_one(vp, o);

            let k = bin_index(mp);
            if let (Some(bn), Some(bp), Some(bo)) =
                (bin_n.get_mut(k), bin_p.get_mut(k), bin_o.get_mut(k))
            {
                *bn += 1;
                *bp += mp;
                *bo += o;
            }
        }

        let base_rate = sum_o / nf;
        let model_brier = sum_mb / nf;
        let venue_brier = sum_vb / nf;
        let mean_diff = sum_d / nf;

        // Paired variance of the per-forecast score differences.
        let mut ss = 0.0;
        for r in &self.resolved {
            let d = r.model_brier() - r.venue_brier() - mean_diff;
            ss += d * d;
        }
        let se_diff = if n > 1 {
            (ss / ((nf - 1.0) * nf)).sqrt()
        } else {
            0.0
        };
        // Zero variance in the paired differences is not "no evidence" — it is
        // the *most* evidence available. Every single forecast differed by the
        // same amount, which is as consistent a result as it is possible to
        // observe.
        //
        // Returning 0.0 here (the first version) mapped the strongest possible
        // signal onto the value that reads as "no signal". It happens to fail
        // in the safe direction — understating an edge rather than inventing
        // one — but it is the same shape of mistake as a NaN that silently
        // becomes zero, and it would have hidden a genuinely clairvoyant model.
        let t_stat = if se_diff > 0.0 {
            mean_diff / se_diff
        } else if mean_diff == 0.0 {
            0.0 // genuinely identical forecasts: no difference to test
        } else {
            f64::INFINITY * mean_diff.signum()
        };

        // Cluster-robust standard error (Cameron & Miller 2015, CR1,
        // specialised to a simple mean — see the module doc's "Clustering"
        // section for why `se_diff` above is the wrong statistic to act on
        // for pooled replay data). `BTreeMap`, not `HashMap`: this crate's
        // sibling `px-core` treats bit-exact reproducibility as a
        // correctness property, not a nicety, and a hash map's iteration
        // order is randomised per-process — summing floating-point cluster
        // contributions in a random order would make this score
        // non-reproducible run to run for no algorithmic reason.
        let mut clusters: std::collections::BTreeMap<u64, f64> = Default::default();
        for r in &self.resolved {
            let d = r.model_brier() - r.venue_brier() - mean_diff;
            *clusters.entry(r.forecast.cluster_id).or_insert(0.0) += d;
        }
        let n_clusters = clusters.len();
        let se_diff_clustered = if n_clusters > 1 {
            let sum_sq: f64 = clusters.values().map(|s| s * s).sum();
            let cr0 = sum_sq / (nf * nf);
            let correction = n_clusters as f64 / (n_clusters as f64 - 1.0);
            (cr0 * correction).sqrt()
        } else {
            // A single cluster carries no information about between-cluster
            // variance — there is nothing to estimate it from. `NaN`, not
            // `0.0`: this is "undefined", not "zero uncertainty", and
            // `has_edge()`'s separate `n_clusters >= 20` gate is what
            // actually protects callers from treating this as evidence.
            f64::NAN
        };
        let t_stat_clustered = if se_diff_clustered > 0.0 {
            mean_diff / se_diff_clustered
        } else if se_diff_clustered == 0.0 {
            if mean_diff == 0.0 {
                0.0
            } else {
                f64::INFINITY * mean_diff.signum()
            }
        } else {
            f64::NAN // se_diff_clustered was NaN (n_clusters <= 1)
        };

        // Murphy decomposition: Brier = Reliability - Resolution + Uncertainty.
        let uncertainty = base_rate * (1.0 - base_rate);
        let mut reliability = 0.0;
        let mut resolution = 0.0;
        let mut bins = [Bin::default(); BINS];
        for k in 0..BINS {
            let (nk, pk, ok) = match (bin_n.get(k), bin_p.get(k), bin_o.get(k)) {
                (Some(a), Some(b), Some(c)) => (*a, *b, *c),
                _ => continue,
            };
            if nk == 0 {
                continue;
            }
            let nkf = nk as f64;
            let mean_forecast = pk / nkf;
            let observed = ok / nkf;
            let rel = mean_forecast - observed;
            let res = observed - base_rate;
            reliability += nkf * rel * rel;
            resolution += nkf * res * res;
            if let Some(slot) = bins.get_mut(k) {
                *slot = Bin {
                    n: nk,
                    mean_forecast,
                    observed,
                };
            }
        }
        reliability /= nf;
        resolution /= nf;

        let venue_b = if venue_brier > 0.0 {
            venue_brier
        } else {
            f64::NAN
        };
        let skill_score = if venue_b.is_nan() {
            0.0
        } else {
            1.0 - model_brier / venue_b
        };

        Scorecard {
            n,
            base_rate,
            model_brier,
            venue_brier,
            skill_score,
            model_log_loss: sum_mll / nf,
            venue_log_loss: sum_vll / nf,
            reliability,
            resolution,
            uncertainty,
            mean_diff,
            se_diff,
            t_stat,
            n_clusters,
            se_diff_clustered,
            t_stat_clustered,
            bins,
        }
    }
}

#[inline]
fn bin_index(p: f64) -> usize {
    let k = (clamp01(p) * BINS as f64) as usize;
    if k >= BINS {
        BINS - 1
    } else {
        k
    }
}

/// Render a scorecard as a report.
pub fn report(s: &Scorecard) -> String {
    let mut out = String::new();
    out.push_str(&format!("Forecast scorecard  (n = {})\n", s.n));
    out.push_str(&format!("  base rate of YES      {:>8.4}\n", s.base_rate));
    out.push('\n');
    out.push_str(&format!(
        "  Brier  model {:>8.5}   venue {:>8.5}   (lower is better)\n",
        s.model_brier, s.venue_brier
    ));
    out.push_str(&format!(
        "  LogL   model {:>8.5}   venue {:>8.5}\n",
        s.model_log_loss, s.venue_log_loss
    ));
    out.push_str(&format!(
        "  SKILL SCORE vs venue mid  {:>+8.4}   (>0 = model beats the market)\n",
        s.skill_score
    ));
    out.push_str(&format!(
        "  paired t-stat (naive)     {:>+8.2}   assumes every forecast is independent — it is not\n",
        s.t_stat
    ));
    out.push_str(&format!(
        "  paired t-stat (clustered) {:>+8.2}   {} clusters — THIS is the one to trust, < -2 = significant\n",
        s.t_stat_clustered, s.n_clusters
    ));
    out.push('\n');
    out.push_str("  Murphy decomposition of the model's Brier score\n");
    out.push_str(&format!(
        "    reliability {:>8.5}  (miscalibration — fixable)\n",
        s.reliability
    ));
    out.push_str(&format!(
        "    resolution  {:>8.5}  (real information — not fakeable)\n",
        s.resolution
    ));
    out.push_str(&format!(
        "    uncertainty {:>8.5}  (the market's difficulty, not ours)\n",
        s.uncertainty
    ));
    out.push('\n');
    out.push_str("  Calibration\n");
    out.push_str("    forecast    n   said   happened    error\n");
    for (k, b) in s.bins.iter().enumerate() {
        if b.n == 0 {
            continue;
        }
        let lo = k as f64 / BINS as f64;
        out.push_str(&format!(
            "    {:.1}-{:.1}  {:>5}  {:>5.3}     {:>6.3}   {:>+6.3}\n",
            lo,
            lo + 1.0 / BINS as f64,
            b.n,
            b.mean_forecast,
            b.observed,
            b.error()
        ));
    }
    out.push('\n');
    out.push_str(&format!("  VERDICT: {}\n", s.verdict()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG.
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

    /// `cluster_id` defaults to 0 — fine for tests with a handful of
    /// hand-written calls where "every forecast independent" is not the
    /// point being tested. Loops that stand in for "many independent
    /// observations" use `fc_c` below instead, one cluster per iteration.
    fn fc(model_p: f64, venue_p: f64, outcome: bool) -> Resolved {
        fc_c(model_p, venue_p, outcome, 0)
    }

    fn fc_c(model_p: f64, venue_p: f64, outcome: bool, cluster_id: u64) -> Resolved {
        Resolved {
            forecast: Forecast {
                t_s: 0.0,
                model_p,
                venue_p,
                horizon_s: 60.0,
                cluster_id,
            },
            outcome,
        }
    }

    #[test]
    fn empty_scorer_is_safe() {
        let s = Scorer::new().score();
        assert_eq!(s.n, 0);
        assert_eq!(s.n_clusters, 0);
        assert!(s.verdict().starts_with("INSUFFICIENT"));
        assert!(!s.has_edge());
    }

    #[test]
    fn brier_matches_the_definition() {
        // Forecast 0.7, outcome YES -> (0.7-1)^2 = 0.09
        assert!((fc(0.7, 0.5, true).model_brier() - 0.09).abs() < 1e-12);
        // Forecast 0.7, outcome NO  -> 0.49
        assert!((fc(0.7, 0.5, false).model_brier() - 0.49).abs() < 1e-12);
        // Perfect forecast scores zero.
        assert!(fc(1.0, 0.5, true).model_brier() < 1e-12);
    }

    #[test]
    fn a_perfect_model_beats_a_coin_flip_venue() {
        let mut s = Scorer::new();
        let mut rng = Lcg(1);
        for i in 0..1000u64 {
            let o = rng.u() < 0.5;
            // Model is clairvoyant; venue always says 50%. Each draw is a
            // genuinely fresh, independent coin flip, so each gets its own
            // cluster — this is the "clustered SE == naive SE" regime.
            s.record(fc_c(if o { 1.0 } else { 0.0 }, 0.5, o, i));
        }
        let c = s.score();
        assert_eq!(c.n_clusters, 1000);
        assert!(c.model_brier < 1e-9);
        assert!((c.venue_brier - 0.25).abs() < 1e-9);
        assert!((c.skill_score - 1.0).abs() < 1e-9);
        assert!(c.t_stat < -2.0);
        assert!(c.t_stat_clustered < -2.0);
        assert!(c.has_edge());
        assert!(c.verdict().starts_with("EDGE"));
    }

    #[test]
    fn a_constant_difference_is_maximal_evidence_not_zero() {
        // Regression: zero variance in the paired differences used to return
        // t = 0, mapping the strongest possible signal onto the value that
        // reads as "no signal". Checked for both the naive and clustered
        // statistic — clustering must not reintroduce the bug it does not
        // otherwise touch.
        let mut better = Scorer::new();
        let mut worse = Scorer::new();
        for i in 0..500u64 {
            let o = i % 3 == 0;
            better.record(fc_c(if o { 1.0 } else { 0.0 }, 0.5, o, i));
            worse.record(fc_c(0.5, if o { 1.0 } else { 0.0 }, o, i));
        }
        let b = better.score();
        let w = worse.score();
        assert_eq!(b.se_diff, 0.0);
        assert_eq!(b.se_diff_clustered, 0.0);
        assert!(b.t_stat.is_infinite() && b.t_stat < 0.0, "t {}", b.t_stat);
        assert!(
            b.t_stat_clustered.is_infinite() && b.t_stat_clustered < 0.0,
            "t_clustered {}",
            b.t_stat_clustered
        );
        assert!(b.has_edge());
        // And the mirror case must be unambiguously bad, not ambiguous.
        assert!(w.t_stat.is_infinite() && w.t_stat > 0.0);
        assert!(w.t_stat_clustered.is_infinite() && w.t_stat_clustered > 0.0);
        assert!(!w.has_edge());
    }

    #[test]
    fn a_model_that_just_echoes_the_venue_has_no_skill() {
        // The null result this crate exists to detect. Copying the market price
        // scores exactly as well as the market price, and BSS is zero.
        let mut s = Scorer::new();
        let mut rng = Lcg(7);
        for i in 0..2000u64 {
            let p = 0.2 + 0.6 * rng.u();
            let o = rng.u() < p;
            s.record(fc_c(p, p, o, i));
        }
        let c = s.score();
        assert!(c.skill_score.abs() < 1e-9, "skill {}", c.skill_score);
        assert_eq!(c.t_stat, 0.0);
        assert_eq!(c.t_stat_clustered, 0.0);
        assert!(!c.has_edge());
    }

    #[test]
    fn a_worse_model_scores_negative_skill() {
        let mut s = Scorer::new();
        let mut rng = Lcg(11);
        for i in 0..2000u64 {
            let p = 0.2 + 0.6 * rng.u();
            let o = rng.u() < p;
            // Model is the truth pushed toward the wrong side.
            let bad = clamp01(p + if o { -0.25 } else { 0.25 });
            s.record(fc_c(bad, p, o, i));
        }
        let c = s.score();
        assert!(c.skill_score < 0.0, "skill {}", c.skill_score);
        assert!(c.t_stat > 2.0, "t {}", c.t_stat);
        assert!(
            c.t_stat_clustered > 2.0,
            "t_clustered {}",
            c.t_stat_clustered
        );
        assert!(!c.has_edge());
        assert!(c.verdict().starts_with("NO EDGE"));
    }

    #[test]
    fn murphy_decomposition_adds_up() {
        // Brier == Reliability - Resolution + Uncertainty, exactly, when the
        // bin means are used. This identity is the test: if it does not hold,
        // one of the three terms is computed wrong.
        let mut s = Scorer::new();
        let mut rng = Lcg(23);
        for i in 0..5000u64 {
            let p = rng.u();
            let o = rng.u() < p;
            s.record(fc_c(p, 0.5, o, i));
        }
        let c = s.score();
        let recomposed = c.reliability - c.resolution + c.uncertainty;
        // Binning introduces a small within-bin discretisation error; the
        // identity is exact only in the limit of narrow bins.
        assert!(
            (recomposed - c.model_brier).abs() < 5e-3,
            "brier {} vs recomposed {}",
            c.model_brier,
            recomposed
        );
    }

    #[test]
    fn a_well_calibrated_model_has_low_reliability_error() {
        let mut s = Scorer::new();
        let mut rng = Lcg(31);
        for i in 0..20_000u64 {
            let p = rng.u();
            let o = rng.u() < p; // outcomes genuinely occur at rate p
            s.record(fc_c(p, 0.5, o, i));
        }
        let c = s.score();
        assert!(c.reliability < 0.005, "reliability {}", c.reliability);
        assert!(c.resolution > 0.05, "resolution {}", c.resolution);
        // Calibration bins should track the diagonal.
        for b in c.bins.iter().filter(|b| b.n > 100) {
            assert!(b.error().abs() < 0.05, "bin off by {}", b.error());
        }
    }

    #[test]
    fn overconfidence_shows_up_as_reliability_not_resolution() {
        // The distinction that matters: an overconfident model is *informative*
        // and miscalibrated. Resolution stays high, reliability degrades, and
        // the fix is post-processing rather than a new model.
        let mut calibrated = Scorer::new();
        let mut overconf = Scorer::new();
        let mut rng = Lcg(41);
        for i in 0..20_000u64 {
            let p = rng.u();
            let o = rng.u() < p;
            calibrated.record(fc_c(p, 0.5, o, i));
            // Push away from 0.5 — same ranking, more extreme.
            let pushed = clamp01(0.5 + (p - 0.5) * 1.6);
            overconf.record(fc_c(pushed, 0.5, o, i));
        }
        let a = calibrated.score();
        let b = overconf.score();
        assert!(b.reliability > a.reliability * 5.0);
        assert!(b.resolution > a.resolution * 0.8, "resolution collapsed");
        assert!(b.verdict().contains("recalibrat") || b.skill_score > 0.0);
    }

    #[test]
    fn a_constant_forecast_has_zero_resolution() {
        // Says the same thing regardless of the question — perfectly calibrated
        // at the base rate, and completely uninformative.
        let mut s = Scorer::new();
        let mut rng = Lcg(53);
        for i in 0..5000u64 {
            let o = rng.u() < 0.3;
            s.record(fc_c(0.3, 0.5, o, i));
        }
        let c = s.score();
        assert!(c.resolution < 1e-6, "resolution {}", c.resolution);
        assert!(c.reliability < 1e-3, "reliability {}", c.reliability);
    }

    #[test]
    fn significance_requires_sample_size() {
        // A tiny sample with a flattering point estimate must not read as edge.
        let mut s = Scorer::new();
        for i in 0..40u64 {
            let o = i % 2 == 0;
            s.record(fc_c(if o { 0.9 } else { 0.1 }, 0.5, o, i));
        }
        let c = s.score();
        assert!(c.skill_score > 0.0, "point estimate should look good");
        assert!(!c.has_edge(), "40 observations must not qualify");
        assert!(c.verdict().starts_with("INSUFFICIENT"));
    }

    #[test]
    fn significance_requires_enough_clusters_not_just_forecasts() {
        // The bug this whole feature exists to prevent: pooling thousands of
        // forecasts from a *handful* of correlated runs must not read as
        // edge just because n is large. 8000 forecasts, but only 8 clusters
        // — well past the n >= 200 bar, nowhere near the n_clusters >= 20
        // bar.
        let mut s = Scorer::new();
        let mut rng = Lcg(101);
        for cluster in 0..8u64 {
            // Each cluster gets its own small, consistent edge, but the
            // *within*-cluster draws are highly repetitive (a stand-in for
            // "many correlated per-second forecasts from one simulated run").
            let bias = 0.05 + 0.01 * cluster as f64;
            for _ in 0..1000 {
                let truth = 0.5 + bias;
                let o = rng.u() < truth;
                let venue = 0.5;
                let model = truth;
                s.record(fc_c(model, venue, o, cluster));
            }
        }
        let c = s.score();
        assert!(c.n >= 200);
        assert_eq!(c.n_clusters, 8);
        assert!(!c.has_edge(), "8 clusters must not qualify regardless of n");
        assert!(c.verdict().starts_with("INSUFFICIENT CLUSTERS"));
    }

    #[test]
    fn clustering_shrinks_significance_that_naive_pooling_overstates() {
        // The actual failure mode: within-cluster observations that are
        // *identical* (perfect correlation, the limit case of "consecutive
        // ticks are almost perfectly correlated") make the naive t-stat
        // arbitrarily large as more copies are added, while the clustered
        // t-stat — correctly seeing only `n_clusters` independent pieces of
        // evidence — does not move once the per-cluster values are fixed.
        fn build(copies_per_cluster: u64) -> Scorecard {
            let mut s = Scorer::new();
            // 30 clusters, each with a small, *different* true edge so the
            // between-cluster variance is real and nonzero.
            for cluster in 0..30u64 {
                let d = -0.01 - 0.001 * cluster as f64; // model - venue, always negative
                for _ in 0..copies_per_cluster {
                    // Encode the desired per-forecast Brier difference `d`
                    // directly via venue_p = 0.5 fixed, model_p chosen so
                    // model_brier - venue_brier == d for a fixed outcome.
                    // outcome = YES, venue_brier = 0.25 always.
                    let model_brier = 0.25 + d;
                    let model_p = 1.0 - model_brier.max(0.0).sqrt();
                    s.record(fc_c(model_p, 0.5, true, cluster));
                }
            }
            s.score()
        }

        let few = build(1); // one observation per cluster: n == n_clusters
        let many = build(50); // fifty *identical* copies per cluster

        assert_eq!(few.n_clusters, 30);
        assert_eq!(many.n_clusters, 30);
        assert!(many.n > few.n);

        // The clustered t-stat is materially unchanged by duplicating
        // perfectly correlated observations...
        assert!(
            (few.t_stat_clustered - many.t_stat_clustered).abs() < 1.0,
            "clustered t moved too much: {} -> {}",
            few.t_stat_clustered,
            many.t_stat_clustered
        );
        // ...while the naive t-stat, which treats every copy as new
        // evidence, grows roughly with sqrt(copies) and ends up far larger
        // in magnitude than the clustered one on the same data.
        assert!(
            many.t_stat.abs() > many.t_stat_clustered.abs() * 3.0,
            "naive {} vs clustered {} did not diverge as expected",
            many.t_stat,
            many.t_stat_clustered
        );
    }

    #[test]
    fn clustered_se_degenerates_to_naive_se_when_every_forecast_is_its_own_cluster() {
        // The identity documented at the top of this file: with G == n,
        // CR1's finite-cluster correction reduces exactly to the ordinary
        // standard error of the mean.
        let mut s = Scorer::new();
        let mut rng = Lcg(131);
        for i in 0..3000u64 {
            let truth = rng.u();
            let o = rng.u() < truth;
            let venue = clamp01(truth + (rng.u() - 0.5) * 0.3);
            let model = clamp01(truth + (rng.u() - 0.5) * 0.2);
            s.record(fc_c(model, venue, o, i));
        }
        let c = s.score();
        assert_eq!(c.n_clusters, c.n);
        assert!(
            (c.se_diff - c.se_diff_clustered).abs() < 1e-9,
            "naive {} vs clustered {}",
            c.se_diff,
            c.se_diff_clustered
        );
        assert!(
            (c.t_stat - c.t_stat_clustered).abs() < 1e-6,
            "naive {} vs clustered {}",
            c.t_stat,
            c.t_stat_clustered
        );
    }

    #[test]
    fn a_single_cluster_gives_an_undefined_not_zero_clustered_se() {
        // One cluster has no between-cluster variance to measure at all —
        // this must read as "undefined", not as "no uncertainty".
        let mut s = Scorer::new();
        for i in 0..300u64 {
            let o = i % 2 == 0;
            s.record(fc_c(if o { 0.9 } else { 0.1 }, 0.5, o, 0)); // all cluster 0
        }
        let c = s.score();
        assert_eq!(c.n_clusters, 1);
        assert!(c.se_diff_clustered.is_nan(), "{}", c.se_diff_clustered);
        assert!(c.t_stat_clustered.is_nan(), "{}", c.t_stat_clustered);
        assert!(!c.has_edge());
        assert!(c.verdict().starts_with("INSUFFICIENT CLUSTERS"));
    }

    #[test]
    fn a_small_real_edge_is_detected_with_enough_data() {
        let mut s = Scorer::new();
        let mut rng = Lcg(67);
        for i in 0..20_000u64 {
            let truth = rng.u();
            let o = rng.u() < truth;
            // Venue is the truth plus noise; model is the truth plus less noise.
            let venue = clamp01(truth + (rng.u() - 0.5) * 0.20);
            let model = clamp01(truth + (rng.u() - 0.5) * 0.14);
            s.record(fc_c(model, venue, o, i));
        }
        let c = s.score();
        assert!(c.skill_score > 0.0, "skill {}", c.skill_score);
        assert!(c.t_stat < -2.0, "t {}", c.t_stat);
        assert!(
            c.t_stat_clustered < -2.0,
            "t_clustered {}",
            c.t_stat_clustered
        );
        assert!(c.has_edge());
    }

    #[test]
    fn log_loss_survives_a_confidently_wrong_forecast() {
        let mut s = Scorer::new();
        s.record(fc(1.0, 0.5, false)); // maximally wrong
        let c = s.score();
        assert!(c.model_log_loss.is_finite());
        assert!(c.model_log_loss > 10.0);
    }

    #[test]
    fn nan_forecasts_are_rejected_not_scored() {
        let mut s = Scorer::new();
        s.record(fc(f64::NAN, 0.5, true));
        s.record(fc(0.5, f64::NAN, true));
        s.record(fc(f64::INFINITY, 0.5, true));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn report_renders_without_panicking() {
        let mut s = Scorer::new();
        let mut rng = Lcg(97);
        for i in 0..500u64 {
            let p = rng.u();
            s.record(fc_c(p, 0.5, rng.u() < p, i));
        }
        let text = report(&s.score());
        assert!(text.contains("SKILL SCORE"));
        assert!(text.contains("VERDICT"));
        assert!(text.contains("Calibration"));
        assert!(text.contains("clustered"));
    }
}
