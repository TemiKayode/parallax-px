//! The venue's fee formula, and the single most consequential number in the
//! whole system.
//!
//! Polymarket charges takers
//!
//! ```text
//!   fee = C * feeRate * p * (1 - p)
//! ```
//!
//! where `C` is share count and `p` is the trade price. Makers pay nothing and
//! receive a rebate equal to a category-dependent fraction of the fee their
//! counterparty paid.
//!
//! # Why this reshapes the strategy
//!
//! On a crypto market the rate is 0.07, so a taker crossing at 50 cents pays
//! **1.75 cents per share**. That is not a rounding error to be swept into a
//! "costs" constant — it is larger than the great majority of mispricings that
//! exist in a liquid five-minute book.
//!
//! Run the numbers. To profit as a taker at the money you need more than 1.75c
//! of edge *after* slippage. Meanwhile a maker at the same price pays zero and
//! collects 20% of the taker's 1.75c, or +0.35c per share. The gap between the
//! two sides of the same trade is **2.1 cents**, which is two full ticks on a
//! 1c-tick market.
//!
//! The consequence for the architecture is direct and it is not what the
//! original brief assumed: an aggressive latency-arbitrage bot that crosses the
//! spread whenever its model disagrees with the book will lose money on a
//! majority of its correct calls. The edge lives in *resting* — quoting a fair
//! value the rest of the book has not caught up to, and being adversely
//! selected less often than the fee-plus-rebate differential is worth.
//!
//! Taking is reserved for two cases where it genuinely pays: edges wide enough
//! to clear 1.75c (which do occur, briefly, after a news shock), and inventory
//! reduction where the fee is the price of not carrying risk.

use px_core::{Category, Px, Qty, Usd};

/// Fee and rebate schedule for one market.
#[derive(Clone, Copy, Debug)]
pub struct FeeModel {
    /// `feeRate` in the venue formula.
    pub taker_rate: f64,
    /// Fraction of the counterparty's fee paid back to the resting maker.
    pub maker_rebate: f64,
}

impl FeeModel {
    pub fn for_category(c: Category) -> Self {
        FeeModel {
            taker_rate: c.taker_fee_rate(),
            maker_rebate: c.maker_rebate(),
        }
    }

    /// Taker fee per share at price `p`, in micro-dollars. Always >= 0.
    #[inline(always)]
    pub fn taker_per_share(&self, p: Px) -> f64 {
        let x = p.as_f64();
        self.taker_rate * x * (1.0 - x) * 1e6
    }

    /// Maker rebate per share at price `p`, in micro-dollars. A credit, so the
    /// sign convention here is positive-is-good.
    #[inline(always)]
    pub fn maker_per_share(&self, p: Px) -> f64 {
        self.taker_rate
            * self.maker_rebate
            * {
                let x = p.as_f64();
                x * (1.0 - x)
            }
            * 1e6
    }

    /// Total taker fee for a quantity at a price.
    #[inline]
    pub fn taker_total(&self, p: Px, q: Qty) -> Usd {
        Usd((self.taker_per_share(p) * q.as_f64()) as i64)
    }

    /// The per-share cost of choosing to cross rather than rest, at a given
    /// price. This is the hurdle a taking decision must clear *on top of* any
    /// price improvement, and it is the number the selector state machine
    /// consults before ever considering an aggressive structure.
    #[inline(always)]
    pub fn cross_penalty(&self, p: Px) -> f64 {
        self.taker_per_share(p) + self.maker_per_share(p)
    }

    /// Price at which the taker fee is maximised. Always 50 cents — the fee is
    /// a downward parabola in price. Useful to state explicitly because it
    /// means near-resolution strategies (buying at 97c) face a fee an order of
    /// magnitude smaller than at-the-money strategies.
    #[inline(always)]
    pub fn worst_price(&self) -> Px {
        Px(500_000)
    }
}

/// Expected liquidity-reward accrual for a resting order.
///
/// The venue scores resting orders with a quadratic in distance from the
/// adjusted midpoint, `S(v, s) = ((v - s)/v)^2`, where `v` is the market's
/// maximum qualifying spread and `s` is our distance from mid. Scores are
/// sampled once per minute at a random instant and normalised against every
/// other maker, then paid from a fixed daily pool.
///
/// Two design consequences fall straight out of the scoring rule:
///
/// 1. **The objective is time-weighted presence, not instantaneous presence.**
///    Because sampling is random within the minute, what earns is the fraction
///    of wall-clock time our order is resting inside `v`. A quote that is
///    cancelled and replaced fifty times a second earns exactly as much as one
///    that rests, and costs fifty times the rate-limit budget. This is the
///    single strongest argument for the quote-economy governor in `px-risk`.
///
/// 2. **The gradient is steep near the mid.** Moving from 3 ticks out to 1 tick
///    out on a `v = 3` market multiplies the score by 4. Reward capture and
///    adverse selection pull in exactly opposite directions, and the quoting
///    engine has to price that trade-off rather than pick a side.
#[derive(Clone, Copy, Debug)]
pub struct RewardModel {
    /// Daily reward pool for this market, in micro-dollars.
    pub pool_per_day: f64,
    /// Our running estimate of the total `Q` across all makers, in
    /// share-score units. Calibrated online from observed payouts.
    pub est_total_q: f64,
    /// Maximum qualifying spread from mid, in ticks.
    pub max_spread_ticks: i32,
    /// Two-sided quoting scores at full weight; one-sided is divided by this.
    /// The venue currently uses 3.0.
    pub one_sided_divisor: f64,
    /// Minimum resting size that qualifies for rewards at all.
    ///
    /// Each incentivised market publishes one, and an order below it scores
    /// **zero** — it is not a reduced payout, it is no payout. `MarketSpec`
    /// carried this field from the first version and nothing ever read it, so
    /// the edge calculator credited reward accrual to quotes the venue would
    /// not have paid a cent on. That inflates the maker economics precisely in
    /// the regime where the engine quotes small — a book in `Reducing` state,
    /// throttled to 40% size, was being told it still earned full rewards.
    pub min_qualifying_size: Qty,
}

impl RewardModel {
    /// Quadratic order-position score for an order `d` ticks from the mid.
    /// Zero outside the qualifying spread — a hard cliff, not a taper.
    #[inline(always)]
    pub fn position_score(&self, distance_ticks: i32) -> f64 {
        if distance_ticks < 0
            || distance_ticks > self.max_spread_ticks
            || self.max_spread_ticks <= 0
        {
            return 0.0;
        }
        let v = self.max_spread_ticks as f64;
        let s = distance_ticks as f64;
        let r = (v - s) / v;
        r * r
    }

    /// Expected reward accrual in micro-dollars per share per second of resting
    /// time, for an order `d` ticks from mid.
    pub fn credit_per_share_sec(&self, distance_ticks: i32, qty: Qty, two_sided: bool) -> f64 {
        if self.est_total_q <= 0.0 {
            return 0.0;
        }
        // Below the market's minimum qualifying size the order scores nothing.
        // A cliff, not a taper.
        if qty < self.min_qualifying_size {
            return 0.0;
        }
        let s = self.position_score(distance_ticks);
        let s = if two_sided {
            s
        } else {
            s / self.one_sided_divisor.max(1.0)
        };
        (self.pool_per_day / 86_400.0) * (s / self.est_total_q)
    }

    /// Total expected credit for resting `q` shares `d` ticks out for
    /// `seconds`. Enters the maker edge calculation as a positive term.
    pub fn expected_credit(
        &self,
        q: Qty,
        distance_ticks: i32,
        seconds: f64,
        two_sided: bool,
    ) -> f64 {
        self.credit_per_share_sec(distance_ticks, q, two_sided) * q.as_f64() * seconds.max(0.0)
    }
}

impl Default for RewardModel {
    fn default() -> Self {
        RewardModel {
            pool_per_day: 0.0,
            est_total_q: 1.0,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taker_fee_reproduces_the_published_crypto_table() {
        let f = FeeModel::for_category(Category::Crypto);
        // Per 100 shares, from the venue's own fee table.
        let cases = [
            (0.01, 0.07),
            (0.10, 0.63),
            (0.25, 1.31),
            (0.50, 1.75),
            (0.75, 1.31),
            (0.90, 0.63),
            (0.99, 0.07),
        ];
        for (price, expected_per_100) in cases {
            let px = Px::from_f64(price);
            let got = f.taker_per_share(px) * 100.0 / 1e6;
            assert!(
                (got - expected_per_100).abs() < 0.005,
                "at {price}: expected ${expected_per_100}, got ${got}"
            );
        }
    }

    #[test]
    fn politics_and_sports_rates_match_their_tables() {
        let pol = FeeModel::for_category(Category::Politics);
        assert!((pol.taker_per_share(Px(500_000)) * 100.0 / 1e6 - 1.00).abs() < 1e-6);
        let sp = FeeModel::for_category(Category::Sports);
        assert!((sp.taker_per_share(Px(500_000)) * 100.0 / 1e6 - 1.25).abs() < 1e-6);
        let geo = FeeModel::for_category(Category::Geopolitics);
        assert_eq!(geo.taker_per_share(Px(500_000)), 0.0);
    }

    #[test]
    fn fee_peaks_at_fifty_cents_and_is_symmetric() {
        let f = FeeModel::for_category(Category::Crypto);
        let at_50 = f.taker_per_share(Px(500_000));
        for p in [100_000, 250_000, 400_000, 600_000, 750_000, 900_000] {
            assert!(f.taker_per_share(Px(p)) < at_50);
        }
        for p in [50_000i32, 200_000, 310_000, 450_000] {
            let lo = f.taker_per_share(Px(p));
            let hi = f.taker_per_share(Px(1_000_000 - p));
            assert!((lo - hi).abs() < 1e-6);
        }
        assert_eq!(f.worst_price(), Px(500_000));
    }

    #[test]
    fn the_maker_taker_gap_is_two_ticks_at_the_money() {
        // The number that reshapes the strategy: crossing versus resting on the
        // same crypto trade at 50c differs by 2.1 cents per share.
        let f = FeeModel::for_category(Category::Crypto);
        let gap = f.cross_penalty(Px(500_000)) / 1e6;
        assert!((gap - 0.021).abs() < 1e-9, "gap = {gap}");
        // Which is 2.1 ticks on a 1-cent-tick market.
        assert!(gap / 0.01 > 2.0);
    }

    #[test]
    fn near_resolution_fees_are_an_order_of_magnitude_smaller() {
        // Why the near-resolution structure survives the fee schedule while
        // at-the-money taking does not.
        let f = FeeModel::for_category(Category::Crypto);
        let atm = f.taker_per_share(Px(500_000));
        let near = f.taker_per_share(Px(970_000));
        assert!(atm / near > 8.0, "ratio = {}", atm / near);
        assert!((near / 1e6 - 0.002_037).abs() < 1e-5);
    }

    #[test]
    fn reward_score_is_quadratic_and_cliffs_at_max_spread() {
        let r = RewardModel {
            max_spread_ticks: 3,
            ..Default::default()
        };
        assert!((r.position_score(0) - 1.0).abs() < 1e-12);
        assert!((r.position_score(1) - 4.0 / 9.0).abs() < 1e-12);
        assert!((r.position_score(2) - 1.0 / 9.0).abs() < 1e-12);
        assert_eq!(r.position_score(3), 0.0);
        assert_eq!(r.position_score(4), 0.0);
        assert_eq!(r.position_score(-1), 0.0);
        // One tick in from three-out quadruples the score.
        assert!((r.position_score(1) / r.position_score(2) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn two_sided_quoting_scores_three_times_one_sided() {
        let r = RewardModel {
            pool_per_day: 300_000.0 * 1e6,
            est_total_q: 1e6,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::shares(50),
        };
        let two = r.credit_per_share_sec(1, Qty::shares(100), true);
        let one = r.credit_per_share_sec(1, Qty::shares(100), false);
        assert!((two / one - 3.0).abs() < 1e-9);
        assert!(two > 0.0);
    }

    #[test]
    fn reward_credit_scales_with_resting_time_not_quote_count() {
        // Restating the design point in a test: the credit depends on seconds
        // resting, so cancel/replace churn buys nothing.
        let r = RewardModel {
            pool_per_day: 300_000.0 * 1e6,
            est_total_q: 1e6,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::shares(50),
        };
        let a = r.expected_credit(Qty::shares(100), 1, 60.0, true);
        let b = r.expected_credit(Qty::shares(100), 1, 30.0, true);
        assert!((a / b - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zero_pool_yields_zero_credit() {
        let r = RewardModel::default();
        assert_eq!(r.credit_per_share_sec(0, Qty::shares(100), true), 0.0);
    }
}
