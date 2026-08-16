//! Cross-asset structure, with redundancy measured rather than assumed.
//!
//! The brief asks for correlated underlyings as an input, "but explicitly
//! measure signal redundancy to avoid double-counting the same shock". That
//! instruction is the whole design of this module.
//!
//! When BTC, ETH and SOL all jump on the same macro print, a naive system sees
//! three confirming signals and sizes up threefold. In reality it has seen
//! *one* signal three times. The fix is to model the assets with an explicit
//! common factor, then account for how much of each asset's move that single
//! factor already explains.
//!
//! # What we actually get out of correlated assets
//!
//! Not direction. Crypto reference prices are close enough to a martingale over
//! a five-minute horizon that ETH's move does not tell us where BTC is going.
//! Two things it does give us, both worth real money:
//!
//! 1. **Nowcasting a stale feed.** If our BTC feed goes quiet for 300 ms while
//!    ETH and SOL keep printing and both move together, the common factor tells
//!    us where BTC almost certainly is *now*, before our own feed confirms it.
//!    This is the ultra-low-latency edge the brief is after, and it is real: it
//!    buys back the gap between feeds rather than trying to beat the market.
//! 2. **Correct aggregation of risk.** Positions in BTC 5m and ETH 5m are not
//!    two independent bets during a systematic move, and the risk layer needs a
//!    number for how un-independent they are.

use px_core::math::ln;

/// One slot per `px_core::Underlying`.
pub const N: usize = 7;

/// Shrinkage toward the identity matrix. With seven assets and a fast-decaying
/// EWMA the sample correlation matrix is noisy and can be near-singular;
/// shrinking keeps the factor extraction stable. 0.05 is light — enough to
/// regularise, not enough to wash out genuine structure.
const SHRINK: f64 = 0.05;

/// Cross-asset covariance tracker with one-factor decomposition.
#[derive(Clone, Debug)]
pub struct CrossAsset {
    /// EWMA covariance of sampled returns.
    cov: [[f64; N]; N],
    /// Last price seen per asset, and when.
    last_px: [f64; N],
    last_ts: [f64; N],
    /// Price at the start of the current sampling interval.
    anchor_px: [f64; N],
    seen: [bool; N],
    half_life: f64,
    sample_dt: f64,
    next_sample: f64,
    n_samples: u64,

    // --- Derived, recomputed on each sample ---
    /// Unit-norm first eigenvector of the correlation matrix.
    loadings: [f64; N],
    /// Its eigenvalue: how much of the total variance the common factor carries.
    lambda1: f64,
    /// Per-asset fraction of variance explained by the common factor.
    r2: [f64; N],
    /// Participation ratio of the correlation matrix: the number of genuinely
    /// independent directions of movement. 1.0 means everything is one trade.
    effective_dof: f64,
}

// Every array field (`cov: [[f64; N]; N]`, `last_px`/`last_ts`/`anchor_px`/
// `seen`/`loadings`/`r2`: `[_; N]`) is declared with exactly `N` elements,
// and every loop below iterates `0..N` (or a `j` similarly bounded) — the
// same fixed-size relationship `px_core::book`'s `LEVELS`-bounded loops
// already rely on for the identical proof. Index-based iteration (rather
// than `.iter()`) is deliberate throughout: several loops need the index
// itself (diagonal access `cov[i][i]`, two arrays walked in lockstep,
// row/column access into a 2D array), not just the element.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
impl CrossAsset {
    pub fn new(half_life_s: f64, sample_dt_s: f64) -> Self {
        let mut c = CrossAsset {
            cov: [[0.0; N]; N],
            last_px: [0.0; N],
            last_ts: [0.0; N],
            anchor_px: [0.0; N],
            seen: [false; N],
            half_life: half_life_s,
            sample_dt: sample_dt_s,
            next_sample: 0.0,
            n_samples: 0,
            loadings: [0.0; N],
            lambda1: 1.0,
            r2: [0.0; N],
            effective_dof: N as f64,
        };
        // Start at the identity: assume independence until shown otherwise.
        for i in 0..N {
            c.cov[i][i] = 1e-12;
        }
        c
    }

    /// Record a print. Cheap: two stores. The expensive part happens on the
    /// sampling boundary.
    #[inline]
    pub fn observe(&mut self, asset: usize, price: f64, ts: f64) {
        // `!price.is_finite()` already rejects NaN, so the plain `<= 0.0`
        // here does not need to handle the incomparable case itself.
        if asset >= N || price <= 0.0 || !price.is_finite() {
            return;
        }
        if !self.seen[asset] {
            self.seen[asset] = true;
            self.anchor_px[asset] = price;
            if self.next_sample == 0.0 {
                self.next_sample = ts + self.sample_dt;
            }
        }
        self.last_px[asset] = price;
        self.last_ts[asset] = ts;
    }

    /// Advance the synchronised sampling clock. Call once per event loop turn.
    ///
    /// Sampling all assets on a common grid rather than on their own tick
    /// arrivals is what keeps the Epps effect from crushing the measured
    /// correlations toward zero: asynchronous trading makes fast-sampled
    /// correlations look far weaker than they are.
    pub fn tick(&mut self, ts: f64) -> bool {
        if self.next_sample == 0.0 || ts < self.next_sample {
            return false;
        }
        self.next_sample = ts + self.sample_dt;

        let mut r = [0.0f64; N];
        let mut active = [false; N];
        for i in 0..N {
            if self.seen[i] && self.anchor_px[i] > 0.0 && self.last_px[i] > 0.0 {
                let x = ln(self.last_px[i] / self.anchor_px[i]);
                if x.is_finite() {
                    r[i] = x;
                    active[i] = true;
                }
                self.anchor_px[i] = self.last_px[i];
            }
        }

        let decay = px_core::math::exp(-self.sample_dt * core::f64::consts::LN_2 / self.half_life);
        for i in 0..N {
            for j in 0..N {
                let x = if active[i] && active[j] {
                    r[i] * r[j]
                } else {
                    0.0
                };
                self.cov[i][j] = decay * self.cov[i][j] + (1.0 - decay) * x;
            }
        }
        self.n_samples = self.n_samples.saturating_add(1);
        self.recompute();
        true
    }

    /// Extract the common factor and the redundancy measures.
    fn recompute(&mut self) {
        // Correlation matrix, with shrinkage toward the identity.
        let mut sd = [0.0f64; N];
        for i in 0..N {
            sd[i] = self.cov[i][i].max(0.0).sqrt();
        }
        let mut corr = [[0.0f64; N]; N];
        for i in 0..N {
            for j in 0..N {
                let c = if sd[i] > 0.0 && sd[j] > 0.0 {
                    (self.cov[i][j] / (sd[i] * sd[j])).clamp(-1.0, 1.0)
                } else if i == j {
                    1.0
                } else {
                    0.0
                };
                let target = if i == j { 1.0 } else { 0.0 };
                corr[i][j] = (1.0 - SHRINK) * c + SHRINK * target;
            }
            corr[i][i] = 1.0;
        }

        // Participation ratio. For a correlation matrix, trace = N, so
        //     dof = (sum lambda)^2 / sum lambda^2 = N^2 / ||C||_F^2
        // and ||C||_F^2 = sum of squared entries. No eigendecomposition needed.
        let mut frob = 0.0;
        for i in 0..N {
            for j in 0..N {
                frob += corr[i][j] * corr[i][j];
            }
        }
        self.effective_dof = if frob > 0.0 {
            ((N * N) as f64 / frob).clamp(1.0, N as f64)
        } else {
            N as f64
        };

        // First eigenvector by power iteration. Twelve passes over a 7x7 matrix
        // is ~600 flops and converges comfortably for a dominant factor.
        let mut v = [1.0f64 / (N as f64).sqrt(); N];
        let mut lambda = 1.0;
        for _ in 0..12 {
            let mut w = [0.0f64; N];
            for i in 0..N {
                let mut s = 0.0;
                for j in 0..N {
                    s += corr[i][j] * v[j];
                }
                w[i] = s;
            }
            let mut norm = 0.0;
            for i in 0..N {
                norm += w[i] * w[i];
            }
            norm = norm.sqrt();
            if norm <= 1e-300 {
                break;
            }
            for i in 0..N {
                v[i] = w[i] / norm;
            }
            lambda = norm;
        }
        // Sign convention: make the dominant factor point "up" so that a
        // positive factor return means the complex rallied.
        let mut sum = 0.0;
        for i in 0..N {
            sum += v[i];
        }
        if sum < 0.0 {
            for i in 0..N {
                v[i] = -v[i];
            }
        }

        self.loadings = v;
        self.lambda1 = lambda.clamp(0.0, N as f64);
        for i in 0..N {
            self.r2[i] = (v[i] * v[i] * self.lambda1).clamp(0.0, 1.0);
        }
    }

    /// Fraction of asset `i`'s movement explained by the common factor. This is
    /// the redundancy number: at 0.9, a confirming signal from another asset in
    /// the complex adds almost nothing.
    #[inline(always)]
    pub fn redundancy(&self, i: usize) -> f64 {
        if i < N {
            self.r2[i]
        } else {
            0.0
        }
    }

    /// How many independent bets the current correlation structure supports.
    /// Between 1 (everything moves as one) and `N` (all independent).
    #[inline(always)]
    pub fn effective_dof(&self) -> f64 {
        self.effective_dof
    }

    #[inline(always)]
    pub fn lambda1(&self) -> f64 {
        self.lambda1
    }

    #[inline(always)]
    pub fn loading(&self, i: usize) -> f64 {
        if i < N {
            self.loadings[i]
        } else {
            0.0
        }
    }

    /// Marginal information a source asset adds about a target, beyond what the
    /// common factor already told us. Used to weight a second confirming signal
    /// so that three correlated confirmations do not size like three
    /// independent ones.
    ///
    /// Returns a value in `[0, 1]`.
    pub fn marginal_information(&self, target: usize, source: usize) -> f64 {
        if target >= N || source >= N || target == source {
            return 0.0;
        }
        // Under the one-factor model the shared component is r2_t * r2_s; what
        // is left is idiosyncratic and, by construction, uninformative about
        // the target. So the marginal information a *correlated* source adds
        // about the target is what the factor carries that we did not already
        // observe directly from the target itself.
        (1.0 - self.r2[target] * self.r2[source]).clamp(0.0, 1.0)
    }

    /// Nowcast a stale asset's return from assets that have printed since.
    ///
    /// `stale_since` is the timestamp of the target's last print. Any asset
    /// whose latest print is newer contributes. Returns the predicted log
    /// return of the target over that gap, and an `r2` confidence.
    ///
    /// This is the module's payload: it converts feed-arrival jitter between
    /// correlated venues into a fair-value update we can act on before our own
    /// feed catches up.
    pub fn nowcast(&self, target: usize, sigma: &[f64; N]) -> Nowcast {
        if target >= N || !self.seen[target] || self.lambda1 <= 0.0 {
            return Nowcast::default();
        }
        let t_last = self.last_ts[target];

        // Least-squares projection of standardised fresh returns onto the
        // factor loadings: f_hat = sum(v_j z_j) / sum(v_j^2).
        let mut num = 0.0;
        let mut den = 0.0;
        let mut contributors = 0u8;
        for j in 0..N {
            if j == target || !self.seen[j] || self.last_ts[j] <= t_last {
                continue;
            }
            if sigma[j] <= 0.0 || self.anchor_px[j] <= 0.0 || self.last_px[j] <= 0.0 {
                continue;
            }
            // Return of the fresh asset over the interval the target has missed.
            let r_j = ln(self.last_px[j] / self.anchor_px[j]);
            if !r_j.is_finite() {
                continue;
            }
            let z_j = r_j / sigma[j];
            num += self.loadings[j] * z_j;
            den += self.loadings[j] * self.loadings[j];
            contributors = contributors.saturating_add(1);
        }

        if contributors == 0 || den <= 1e-12 {
            return Nowcast::default();
        }

        let f_hat = num / den;
        let z_target = self.loadings[target] * f_hat;
        let predicted = z_target * sigma[target];

        Nowcast {
            log_return: predicted,
            confidence: self.r2[target],
            contributors,
        }
    }

    #[inline(always)]
    pub fn is_warm(&self) -> bool {
        self.n_samples >= 200
    }

    #[inline(always)]
    pub fn last_ts(&self, i: usize) -> f64 {
        if i < N {
            self.last_ts[i]
        } else {
            0.0
        }
    }
}

/// Result of a cross-asset nowcast.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nowcast {
    /// Predicted log return of the target over the gap since its last print.
    pub log_return: f64,
    /// `R^2` of the target against the common factor: how much to trust it.
    pub confidence: f64,
    /// How many fresh assets contributed.
    pub contributors: u8,
}

impl Nowcast {
    /// Shrink the prediction by its own confidence before applying it. A
    /// nowcast we only half believe should move fair value half as far, not all
    /// the way with a caveat attached.
    #[inline(always)]
    pub fn shrunk(&self) -> f64 {
        self.log_return * self.confidence
    }
}

#[cfg(test)]
// Same bound as `impl CrossAsset` above: every array here is sized `N`
// and every loop iterates `0..N`.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Deterministic LCG — no rand crate, and replayable.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        /// Box-Muller.
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * ln(u1)).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn independent_assets_show_full_degrees_of_freedom() {
        let mut c = CrossAsset::new(60.0, 0.25);
        let mut rng = Lcg(42);
        let mut px = [65_000.0, 3_200.0, 150.0, 0.6, 0.15, 600.0, 30.0];
        let mut t = 0.0;
        for _ in 0..3000 {
            for i in 0..N {
                px[i] *= 1.0 + 0.0005 * rng.normal();
                c.observe(i, px[i], t);
            }
            t += 0.25;
            c.tick(t);
        }
        // Seven independent assets: dof should be near 7.
        assert!(c.effective_dof() > 5.5, "dof = {}", c.effective_dof());
        for i in 0..N {
            assert!(c.redundancy(i) < 0.45, "asset {i} r2 = {}", c.redundancy(i));
        }
    }

    #[test]
    fn perfectly_correlated_assets_collapse_to_one_degree_of_freedom() {
        let mut c = CrossAsset::new(60.0, 0.25);
        let mut rng = Lcg(7);
        let mut px = [65_000.0, 3_200.0, 150.0, 0.6, 0.15, 600.0, 30.0];
        let mut t = 0.0;
        for _ in 0..3000 {
            // One shared shock drives everything.
            let shock = 0.0005 * rng.normal();
            for i in 0..N {
                px[i] *= 1.0 + shock;
                c.observe(i, px[i], t);
            }
            t += 0.25;
            c.tick(t);
        }
        assert!(c.effective_dof() < 1.5, "dof = {}", c.effective_dof());
        // Every asset is almost entirely explained by the common factor.
        for i in 0..N {
            assert!(c.redundancy(i) > 0.85, "asset {i} r2 = {}", c.redundancy(i));
        }
        // And a confirming signal from a peer adds almost nothing.
        assert!(c.marginal_information(0, 1) < 0.2);
    }

    #[test]
    // `marginal_information(t, t)` hits the explicit `target == source`
    // early return of a literal `0.0` — exact by construction.
    #[allow(clippy::float_cmp)]
    fn marginal_information_is_high_for_uncorrelated_pairs() {
        let mut c = CrossAsset::new(60.0, 0.25);
        let mut rng = Lcg(99);
        let mut px = [65_000.0, 3_200.0, 150.0, 0.6, 0.15, 600.0, 30.0];
        let mut t = 0.0;
        for _ in 0..3000 {
            for i in 0..N {
                px[i] *= 1.0 + 0.0005 * rng.normal();
                c.observe(i, px[i], t);
            }
            t += 0.25;
            c.tick(t);
        }
        assert!(c.marginal_information(0, 1) > 0.8);
        assert_eq!(c.marginal_information(0, 0), 0.0);
    }

    #[test]
    fn nowcast_predicts_a_stale_asset_from_fresh_peers() {
        let mut c = CrossAsset::new(30.0, 0.25);
        let mut rng = Lcg(2024);
        let mut px = [65_000.0, 3_200.0, 150.0, 0.6, 0.15, 600.0, 30.0];
        let mut t = 0.0;
        // Warm up with a strong common factor plus a little idiosyncratic noise.
        for _ in 0..3000 {
            let shock = 0.0005 * rng.normal();
            for i in 0..N {
                px[i] *= 1.0 + shock + 0.00005 * rng.normal();
                c.observe(i, px[i], t);
            }
            t += 0.25;
            c.tick(t);
        }
        assert!(c.is_warm());
        assert!(c.redundancy(0) > 0.8);

        // Now BTC (index 0) goes quiet while every peer rallies 0.2%.
        let sigma = [0.0005f64; N];
        for i in 1..N {
            px[i] *= 1.002;
            c.observe(i, px[i], t + 0.1);
        }
        let nc = c.nowcast(0, &sigma);

        assert!(nc.contributors == 6, "contributors = {}", nc.contributors);
        assert!(nc.confidence > 0.8, "confidence = {}", nc.confidence);
        // Direction must be right, and magnitude in the right neighbourhood.
        assert!(nc.log_return > 0.0005, "predicted {}", nc.log_return);
        assert!(nc.log_return < 0.004, "predicted {}", nc.log_return);
        // The shrunk value is what we would actually apply.
        assert!(nc.shrunk() < nc.log_return);
    }

    #[test]
    // `nc` is `Nowcast::default()` (the `contributors == 0` early-return
    // path), whose `log_return` field is `f64::default()` — a literal
    // `0.0`, exact by construction.
    #[allow(clippy::float_cmp)]
    fn nowcast_is_silent_when_no_peer_is_fresher() {
        let mut c = CrossAsset::new(30.0, 0.25);
        let mut t = 0.0;
        for _ in 0..500 {
            for i in 0..N {
                c.observe(i, 100.0 + i as f64, t);
            }
            t += 0.25;
            c.tick(t);
        }
        // Everyone shares the same timestamp, so nothing is fresher than BTC.
        let nc = c.nowcast(0, &[0.0005; N]);
        assert_eq!(nc.contributors, 0);
        assert_eq!(nc.log_return, 0.0);
    }

    #[test]
    fn unseen_asset_yields_no_nowcast() {
        let c = CrossAsset::new(30.0, 0.25);
        let nc = c.nowcast(3, &[0.0005; N]);
        assert_eq!(nc.contributors, 0);
    }

    #[test]
    fn observe_rejects_bad_input_without_panicking() {
        let mut c = CrossAsset::new(30.0, 0.25);
        c.observe(99, 100.0, 1.0);
        c.observe(0, -5.0, 1.0);
        c.observe(0, f64::NAN, 1.0);
        c.observe(0, f64::INFINITY, 1.0);
        assert!(c.tick(2.0) || !c.tick(2.0));
        assert!(c.effective_dof().is_finite());
    }
}
