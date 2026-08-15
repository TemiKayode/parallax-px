//! Multi-market relative-value monitor.
//!
//! The brief's normalised score is
//!
//! ```text
//!   Relative Score = (Current Gap - Typical Gap) / Historical Gap Volatility
//! ```
//!
//! The reason to normalise is that raw gaps are not comparable across markets.
//! A 2-cent gap on a BTC 5-minute contract twenty seconds from expiry is an
//! enormous dislocation; the same 2 cents on a 4-hour contract is noise. A
//! market that habitually trades a cent rich has a *typical* gap of one cent,
//! and pricing that as edge would mean paying for the same non-information all
//! day.
//!
//! Both corrections are what the z-score does: subtract the habitual bias,
//! divide by the market's own gap volatility.

use px_core::{MarketId, Px};

/// Welford-style EWMA tracker for the fair-minus-mid gap of a single market.
#[derive(Clone, Copy, Debug)]
pub struct GapTracker {
    mean: f64,
    var: f64,
    alpha: f64,
    n: u64,
}

impl GapTracker {
    pub fn new(alpha: f64) -> Self {
        GapTracker {
            mean: 0.0,
            var: 0.0,
            alpha: alpha.clamp(1e-5, 1.0),
            n: 0,
        }
    }

    /// Record a gap observation, in micro-dollars.
    pub fn observe(&mut self, gap: f64) {
        if !gap.is_finite() {
            return;
        }
        if self.n == 0 {
            self.mean = gap;
            self.var = 0.0;
        } else {
            let d = gap - self.mean;
            self.mean += self.alpha * d;
            // EWMA of squared deviation, using the pre-update residual: this is
            // the standard incremental form and stays positive by construction.
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * d * d);
        }
        self.n = self.n.saturating_add(1);
    }

    #[inline(always)]
    pub fn typical(&self) -> f64 {
        self.mean
    }

    #[inline(always)]
    pub fn sd(&self) -> f64 {
        self.var.max(0.0).sqrt()
    }

    /// Normalised dislocation. Returns 0 until the tracker has enough history:
    /// an unwarmed tracker reports a huge z-score for the first observation,
    /// which would be the worst possible time to size up.
    pub fn z(&self, gap: f64) -> f64 {
        if self.n < 100 {
            return 0.0;
        }
        let s = self.sd();
        if s <= 1.0 {
            // Gap volatility below one micro-dollar means the market has not
            // moved. Treat as no signal rather than dividing by ~zero.
            return 0.0;
        }
        ((gap - self.mean) / s).clamp(-20.0, 20.0)
    }

    #[inline(always)]
    pub fn is_warm(&self) -> bool {
        self.n >= 100
    }
}

impl Default for GapTracker {
    fn default() -> Self {
        GapTracker::new(0.01)
    }
}

/// One market's standing in the cross-market comparison.
#[derive(Clone, Copy, Debug, Default)]
pub struct Candidate {
    pub market: MarketId,
    /// Signed dislocation: positive means fair value is above the market's mid,
    /// so the YES side is cheap.
    pub z: f64,
    /// Raw gap, micro-dollars.
    pub gap: f64,
    /// Best realisable net edge per share after fees and margin.
    pub net_edge: i32,
    /// How much size that edge supports.
    pub size: px_core::Qty,
    /// Whether the risk layer will actually let us act.
    pub actionable: bool,
    /// Correlation bucket — markets sharing one are the same bet.
    pub bucket: u8,
}

/// Fixed-capacity ranked view of the market constellation.
///
/// Capacity is a compile-time constant because the hot path must not allocate.
/// Sixty-four covers every crypto duration on every listed underlying with
/// room to spare.
pub const MAX_CANDIDATES: usize = 64;

#[derive(Debug)]
pub struct Ranker {
    items: [Candidate; MAX_CANDIDATES],
    len: usize,
}

impl Default for Ranker {
    fn default() -> Self {
        Ranker {
            items: [Candidate::default(); MAX_CANDIDATES],
            len: 0,
        }
    }
}

impl Ranker {
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn push(&mut self, c: Candidate) -> bool {
        if self.len >= MAX_CANDIDATES {
            return false;
        }
        self.items[self.len] = c;
        self.len += 1;
        true
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Candidate] {
        &self.items[..self.len]
    }

    /// The most acute actionable dislocation, ignoring buckets already used.
    ///
    /// `used_buckets` is a bitmask of correlation buckets we already have
    /// exposure in. BTC 5m and BTC 15m share a bucket, so once we have taken
    /// the best of them we do not also take the second — they are one bet, and
    /// stacking them is how a position limit gets quietly doubled.
    pub fn best(&self, used_buckets: u64) -> Option<&Candidate> {
        let mut best: Option<&Candidate> = None;
        for c in self.as_slice() {
            if !c.actionable {
                continue;
            }
            if used_buckets & (1u64 << (c.bucket & 63)) != 0 {
                continue;
            }
            let better = match best {
                None => true,
                // Rank by total edge dollars, not by z. A spectacular z-score on
                // a market that holds nine shares is not where capital goes.
                Some(b) => (c.net_edge as i64) * c.size.0 > (b.net_edge as i64) * b.size.0,
            };
            if better {
                best = Some(c);
            }
        }
        best
    }

    /// Every actionable candidate whose dislocation exceeds a threshold,
    /// deduplicated to one per correlation bucket (the best in each).
    pub fn best_per_bucket(&self, z_min: f64, out: &mut Vec<Candidate>) {
        out.clear();
        let mut seen: u64 = 0;
        // Two passes: cheap, and keeps the function allocation-free apart from
        // the caller's reused vector.
        for _ in 0..MAX_CANDIDATES {
            let mut pick: Option<Candidate> = None;
            for c in self.as_slice() {
                if !c.actionable || c.z.abs() < z_min {
                    continue;
                }
                if seen & (1u64 << (c.bucket & 63)) != 0 {
                    continue;
                }
                let better = match pick {
                    None => true,
                    Some(p) => (c.net_edge as i64) * c.size.0 > (p.net_edge as i64) * p.size.0,
                };
                if better {
                    pick = Some(*c);
                }
            }
            match pick {
                Some(p) => {
                    seen |= 1u64 << (p.bucket & 63);
                    out.push(p);
                }
                None => break,
            }
        }
    }
}

/// Cost of assembling a complete YES+NO set by taking both books.
///
/// A complete set pays exactly one dollar at settlement regardless of outcome,
/// so if it can be assembled for less than a dollar net of fees, the profit is
/// locked in with no model risk at all. This is the only structure in the
/// system whose edge does not depend on the fair-value model being right.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairCost {
    pub yes_px: Px,
    pub no_px: Px,
    /// Combined cost per set, micro-dollars.
    pub gross: i32,
    /// Combined taker fee per set, micro-dollars.
    pub fees: i32,
    /// `1_000_000 - gross - fees`. Positive means free money.
    pub net_profit: i32,
    pub qty: px_core::Qty,
    pub viable: bool,
}

/// Price a complete set against both order books.
pub fn price_pair(
    yes_book: &px_core::DenseBook,
    no_book: &px_core::DenseBook,
    want: px_core::Qty,
    fees: &px_edge::FeeModel,
) -> PairCost {
    let wy = yes_book.walk_buy_unbounded(want);
    let wn = no_book.walk_buy_unbounded(want);
    let q = wy.filled.min(wn.filled);
    if q.is_zero() {
        return PairCost::default();
    }
    // Re-walk at the achievable size so both legs are priced for the size we
    // can actually get on both. Sizing one leg off the other's depth is how a
    // "risk-free" pair ends up as a naked position.
    let wy = yes_book.walk_buy_unbounded(q);
    let wn = no_book.walk_buy_unbounded(q);

    let gross = wy.avg_px.0 + wn.avg_px.0;
    let fee = (fees.taker_per_share(wy.avg_px) + fees.taker_per_share(wn.avg_px)) as i32;
    let net = 1_000_000 - gross - fee;

    PairCost {
        yes_px: wy.avg_px,
        no_px: wn.avg_px,
        gross,
        fees: fee,
        net_profit: net,
        qty: q,
        viable: net > 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_core::{Category, DenseBook, Qty, Side};

    #[test]
    fn gap_tracker_is_silent_until_warm() {
        let mut g = GapTracker::new(0.05);
        for _ in 0..50 {
            g.observe(1000.0);
        }
        assert!(!g.is_warm());
        assert_eq!(g.z(50_000.0), 0.0);
    }

    #[test]
    fn gap_tracker_normalises_a_habitual_bias_away() {
        // A market that always trades one cent rich has zero dislocation when
        // it is one cent rich.
        let mut g = GapTracker::new(0.02);
        let mut s = 1u64;
        for _ in 0..3000 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2000.0;
            g.observe(10_000.0 + noise);
        }
        assert!(g.is_warm());
        assert!(
            (g.typical() - 10_000.0).abs() < 800.0,
            "typical {}",
            g.typical()
        );
        assert!(g.z(10_000.0).abs() < 0.6, "z {}", g.z(10_000.0));
        // But a genuinely unusual gap still scores.
        assert!(g.z(30_000.0) > 3.0, "z {}", g.z(30_000.0));
        assert!(g.z(-10_000.0) < -3.0);
    }

    #[test]
    fn gap_tracker_ignores_nonsense() {
        let mut g = GapTracker::new(0.05);
        for _ in 0..200 {
            g.observe(1000.0);
        }
        let before = g.typical();
        g.observe(f64::NAN);
        g.observe(f64::INFINITY);
        assert_eq!(g.typical(), before);
    }

    #[test]
    fn zero_volatility_market_reports_no_signal() {
        let mut g = GapTracker::new(0.05);
        for _ in 0..500 {
            g.observe(0.0);
        }
        assert_eq!(g.z(0.0), 0.0);
    }

    fn cand(id: u32, z: f64, edge: i32, shares: i64, bucket: u8) -> Candidate {
        Candidate {
            market: MarketId(id),
            z,
            gap: z * 1000.0,
            net_edge: edge,
            size: Qty::shares(shares),
            actionable: true,
            bucket,
        }
    }

    #[test]
    fn ranker_prefers_total_edge_over_headline_z_score() {
        // The trap the brief warns about: a 9-cent edge on 50 shares versus a
        // 3-cent edge on 400 shares. The second is worth three times as much.
        let mut r = Ranker::default();
        r.push(cand(1, 8.0, 90_000, 50, 0));
        r.push(cand(2, 3.0, 30_000, 400, 1));
        let b = r.best(0).unwrap();
        assert_eq!(b.market, MarketId(2));
    }

    #[test]
    fn ranker_skips_non_actionable_candidates() {
        let mut r = Ranker::default();
        let mut c = cand(1, 9.0, 90_000, 1000, 0);
        c.actionable = false;
        r.push(c);
        r.push(cand(2, 1.0, 5_000, 100, 1));
        assert_eq!(r.best(0).unwrap().market, MarketId(2));
    }

    #[test]
    fn ranker_respects_correlation_buckets() {
        // BTC 5m and BTC 15m share bucket 0: once we hold one, the other is
        // the same bet and must not also be selected.
        let mut r = Ranker::default();
        r.push(cand(1, 5.0, 50_000, 500, 0));
        r.push(cand(2, 4.0, 40_000, 400, 0));
        r.push(cand(3, 2.0, 10_000, 200, 1));

        assert_eq!(r.best(0).unwrap().market, MarketId(1));
        // With bucket 0 already used, we fall through to the ETH market.
        assert_eq!(r.best(1 << 0).unwrap().market, MarketId(3));
        // Both buckets used: nothing left.
        assert!(r.best((1 << 0) | (1 << 1)).is_none());
    }

    #[test]
    fn best_per_bucket_returns_one_per_bucket() {
        let mut r = Ranker::default();
        r.push(cand(1, 5.0, 50_000, 500, 0));
        r.push(cand(2, 4.0, 40_000, 400, 0));
        r.push(cand(3, 3.0, 30_000, 300, 1));
        r.push(cand(4, 0.5, 5_000, 100, 2));

        let mut out = Vec::new();
        r.best_per_bucket(2.0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].market, MarketId(1));
        assert_eq!(out[1].market, MarketId(3));
    }

    #[test]
    fn ranker_capacity_is_bounded() {
        let mut r = Ranker::default();
        for i in 0..(MAX_CANDIDATES + 10) {
            let ok = r.push(cand(i as u32, 1.0, 1000, 10, 0));
            if i >= MAX_CANDIDATES {
                assert!(!ok);
            }
        }
        assert_eq!(r.len(), MAX_CANDIDATES);
    }

    fn one_sided(px: i32, shares: i64) -> DenseBook {
        let mut b = DenseBook::new(10_000);
        b.set_level(Side::Ask, Px(px), Qty::shares(shares));
        b.set_level(Side::Bid, Px(px - 20_000), Qty::shares(shares));
        b
    }

    #[test]
    fn a_genuine_sub_dollar_pair_is_detected() {
        // YES offered at 45c, NO offered at 48c: the set costs 93c. Fees at
        // those prices total ~3.5c, so ~3.5c of locked-in profit per set.
        let y = one_sided(450_000, 500);
        let n = one_sided(480_000, 300);
        let f = px_edge::FeeModel::for_category(Category::Crypto);
        let p = price_pair(&y, &n, Qty::shares(1000), &f);

        assert_eq!(p.qty, Qty::shares(300)); // limited by the thinner leg
        assert_eq!(p.gross, 930_000);
        assert!(p.net_profit > 0, "net {}", p.net_profit);
        assert!(p.viable);
    }

    #[test]
    fn a_pair_that_only_looks_free_before_fees_is_rejected() {
        // 49c + 50c = 99c looks like a cent of free money. The two taker fees
        // at those prices come to about 3.5c. It is a losing trade.
        let y = one_sided(490_000, 500);
        let n = one_sided(500_000, 500);
        let f = px_edge::FeeModel::for_category(Category::Crypto);
        let p = price_pair(&y, &n, Qty::shares(100), &f);
        assert_eq!(p.gross, 990_000);
        assert!(p.fees > 30_000, "fees {}", p.fees);
        assert!(!p.viable, "net {}", p.net_profit);
    }

    #[test]
    fn pair_is_sized_to_the_thinner_leg() {
        let y = one_sided(400_000, 1000);
        let n = one_sided(400_000, 25);
        let f = px_edge::FeeModel::for_category(Category::Crypto);
        let p = price_pair(&y, &n, Qty::shares(1000), &f);
        assert_eq!(p.qty, Qty::shares(25));
        assert!(p.viable);
    }

    #[test]
    fn empty_book_yields_no_pair() {
        let y = DenseBook::new(10_000);
        let n = one_sided(400_000, 100);
        let f = px_edge::FeeModel::for_category(Category::Crypto);
        let p = price_pair(&y, &n, Qty::shares(100), &f);
        assert!(!p.viable);
        assert_eq!(p.qty, Qty::ZERO);
    }
}
