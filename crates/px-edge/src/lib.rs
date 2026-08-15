//! `px-edge` — turning a fair probability into a *tradable* edge.
//!
//! The brief's formula is
//!
//! ```text
//!   Tradable Edge = Fair Value - Expected Average Entry - Execution Costs - Safety Margin
//! ```
//!
//! Every term here is computed rather than assumed:
//!
//! * **Expected average entry** comes from walking the actual resting depth for
//!   the actual intended size (`px_core::DenseBook::walk_buy`), so partial fills
//!   and multi-level slippage are priced, not hand-waved.
//! * **Execution costs** use the venue's exact fee formula, which on crypto
//!   markets is large enough to invert the strategy (see `fee`).
//! * **Safety margin** is a multiple of the model's *own* stated uncertainty
//!   (`FairValue::sigma_p`), not a constant. When the volatility estimate is
//!   cold or the reference feed is stale, the margin widens automatically and
//!   the bot stops trading without anyone having to write a special case.
//!
//! And one term the brief did not ask for but the venue insists on:
//!
//! * **Size optimisation.** The requested size is an input, not an answer. The
//!   calculator finds the quantity that maximises *total* edge dollars, because
//!   a 9-cent edge on 50 shares and a 3-cent edge on 400 shares are not the
//!   same trade, and the second one is better.

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod fee;

pub use fee::{FeeModel, RewardModel};

use px_core::{DenseBook, Px, Qty, Side, Usd};

/// Direction of an intended trade.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Buy,
    Sell,
}

impl Dir {
    /// The book side we would consume when taking liquidity in this direction.
    #[inline(always)]
    pub fn take_side(self) -> Side {
        match self {
            Dir::Buy => Side::Ask,
            Dir::Sell => Side::Bid,
        }
    }

    /// The book side we would join when resting.
    #[inline(always)]
    pub fn make_side(self) -> Side {
        match self {
            Dir::Buy => Side::Bid,
            Dir::Sell => Side::Ask,
        }
    }
}

/// Tunables for the edge calculation.
#[derive(Clone, Copy, Debug)]
pub struct EdgeParams {
    /// Safety margin in multiples of the model's `sigma_p`. 1.0 means "only
    /// trade when the edge exceeds one standard deviation of our own model
    /// error", which is deliberately conservative.
    pub safety_k: f64,
    /// Hard floor on net edge per share, micro-dollars. Guards against the
    /// degenerate case where `sigma_p` is tiny because the model is confidently
    /// wrong.
    pub min_edge: i32,
    /// Calibrated adverse selection cost of a passive fill, micro-dollars per
    /// share. Updated online from realised post-fill mark-outs.
    pub adverse: f64,
    /// Cap on how many book levels a take may consume. Deep walks are a
    /// liquidity event, not an edge.
    pub max_levels: u16,
}

impl Default for EdgeParams {
    fn default() -> Self {
        EdgeParams {
            safety_k: 1.0,
            min_edge: 1_000, // 0.1 cent
            adverse: 0.0,
            max_levels: 8,
        }
    }
}

/// Assessment of crossing the spread.
#[derive(Clone, Copy, Debug, Default)]
pub struct TakeAssessment {
    pub qty: Qty,
    pub avg_entry: Px,
    pub worst_px: Px,
    /// `fair - avg_entry` (buy) or `avg_entry - fair` (sell), micro-dollars.
    pub gross_per_share: i32,
    pub fee_per_share: i32,
    pub margin_per_share: i32,
    pub net_per_share: i32,
    pub total_edge: Usd,
    pub levels: u16,
    pub viable: bool,
}

/// Assessment of resting a quote.
#[derive(Clone, Copy, Debug, Default)]
pub struct MakeAssessment {
    pub price: Px,
    pub qty: Qty,
    /// Distance from mid in ticks — drives the reward score.
    pub distance_ticks: i32,
    /// Resting size ahead of us at our price level.
    pub queue_ahead: Qty,
    pub gross_per_share: i32,
    /// Positive: the venue pays makers.
    pub rebate_per_share: i32,
    /// Positive: expected liquidity-reward accrual over the holding horizon.
    pub reward_per_share: i32,
    /// Negative contribution: what we lose by being filled precisely when we
    /// least want to be.
    pub adverse_per_share: i32,
    /// Positive: variance this fill would *remove* from the book.
    pub risk_credit_per_share: i32,
    pub margin_per_share: i32,
    pub net_per_share: i32,
    pub viable: bool,
}

/// Convert a probability standard deviation into a per-share margin in
/// micro-dollars. A binary contract pays $1, so one unit of probability is one
/// dollar of value: the conversion is exact, not a fudge.
#[inline(always)]
fn margin_micro(sigma_p: f64, k: f64) -> i32 {
    // NaN must fail *closed*.
    //
    // `f64::max` returns the non-NaN operand, so `sigma_p.max(0.0)` quietly
    // turns a NaN model uncertainty into zero — that is, into "no safety
    // margin at all, trade freely". A model that has produced a NaN is the last
    // one that should be sized off. Check the inputs before any arithmetic can
    // launder the NaN away.
    if sigma_p.is_nan() || k.is_nan() {
        return MAX_MARGIN;
    }
    let m = sigma_p.max(0.0) * k * 1e6;
    // Clamp to twice the maximum possible value of a binary contract.
    //
    // Returning `i32::MAX` here was a real bug: downstream code computes
    // `gross + rebate + reward - adverse - margin` in i32, and a saturated
    // margin overflowed it. A margin of two dollars per share on a contract
    // that can never be worth more than one is already unambiguously
    // prohibitive, so nothing is lost by clamping there — and it leaves three
    // orders of magnitude of headroom before the arithmetic can wrap.
    //
    // Surfaced by the replay harness once `TwoSpeedVol::rel_err` began folding
    // estimator disagreement into `sigma_p`: during a shock the margin grew
    // past the old guard. The lesson is not "add a guard" — it is that a
    // saturating sentinel is not a safe value to do arithmetic on.
    if m.is_nan() {
        return MAX_MARGIN;
    }
    (m.min(MAX_MARGIN as f64)) as i32
}

/// Prohibitive-but-arithmetically-safe margin ceiling, micro-dollars per share.
pub const MAX_MARGIN: i32 = 2_000_000;

/// Assess taking a specific size.
pub fn assess_take(
    book: &DenseBook,
    dir: Dir,
    fair: Px,
    want: Qty,
    fees: &FeeModel,
    sigma_p: f64,
    p: &EdgeParams,
) -> TakeAssessment {
    let walk = match dir {
        Dir::Buy => book.walk_buy_unbounded(want),
        Dir::Sell => book.walk_sell_unbounded(want),
    };
    finish_take(&walk, dir, fair, fees, sigma_p, p)
}

fn finish_take(
    walk: &px_core::Walk,
    dir: Dir,
    fair: Px,
    fees: &FeeModel,
    sigma_p: f64,
    p: &EdgeParams,
) -> TakeAssessment {
    if walk.filled.is_zero() {
        return TakeAssessment::default();
    }
    let gross = match dir {
        Dir::Buy => fair.0 - walk.avg_px.0,
        Dir::Sell => walk.avg_px.0 - fair.0,
    };
    let fee = fees.taker_per_share(walk.avg_px) as i32;
    let margin = margin_micro(sigma_p, p.safety_k);
    let net = ((gross as i64) - (fee as i64) - (margin as i64))
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let total = Usd(((net as i64) * walk.filled.0) / 1_000_000);

    TakeAssessment {
        qty: walk.filled,
        avg_entry: walk.avg_px,
        worst_px: walk.worst_px,
        gross_per_share: gross,
        fee_per_share: fee,
        margin_per_share: margin,
        net_per_share: net,
        total_edge: total,
        levels: walk.levels,
        viable: net >= p.min_edge && net > 0 && walk.levels <= p.max_levels,
    }
}

/// Find the size that maximises **total** edge dollars, not per-share edge.
///
/// Walks the book level by level. At each cumulative depth it recomputes the
/// average entry, the fee at that average, and the resulting total. The answer
/// is frequently *not* the touch: paying an extra tick to get four times the
/// size is usually the better trade, and paying three extra ticks never is.
///
/// Returns the best assessment found, which may be `viable == false` if no size
/// clears the hurdle.
pub fn optimal_take(
    book: &DenseBook,
    dir: Dir,
    fair: Px,
    max_qty: Qty,
    fees: &FeeModel,
    sigma_p: f64,
    p: &EdgeParams,
) -> TakeAssessment {
    let side = dir.take_side();
    let margin = margin_micro(sigma_p, p.safety_k);

    let mut cum_qty: i64 = 0;
    let mut cum_cash: i128 = 0;
    let mut levels: u16 = 0;
    let mut best = TakeAssessment::default();
    let mut best_total: i64 = 0;

    book.for_each_level(side, |px, avail| {
        if levels >= p.max_levels || cum_qty >= max_qty.0 {
            return false;
        }
        let room = max_qty.0 - cum_qty;
        let take = if avail.0 < room { avail.0 } else { room };
        if take <= 0 {
            return false;
        }

        cum_cash += (px.0 as i128) * (take as i128);
        cum_qty += take;
        levels += 1;

        let avg = Px((cum_cash / (cum_qty as i128)) as i32);
        let gross = match dir {
            Dir::Buy => fair.0 - avg.0,
            Dir::Sell => avg.0 - fair.0,
        };
        let fee = fees.taker_per_share(avg) as i32;
        let net = ((gross as i64) - (fee as i64) - (margin as i64))
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let total = ((net as i64) * cum_qty) / 1_000_000;

        if net > 0 && total > best_total {
            best_total = total;
            best = TakeAssessment {
                qty: Qty(cum_qty),
                avg_entry: avg,
                worst_px: px,
                gross_per_share: gross,
                fee_per_share: fee,
                margin_per_share: margin,
                net_per_share: net,
                total_edge: Usd(total),
                levels,
                viable: net >= p.min_edge,
            };
        }

        // Average entry only worsens as we go deeper, so once net edge has gone
        // negative it cannot recover. Stop.
        net > 0
    });

    best
}

/// Assess resting a quote at `price`.
///
/// The maker economics are structurally different from the taker's, and the
/// difference is why this system is a quoting engine rather than a sniper:
///
/// ```text
///   net = (fair - price)          <- the mispricing we are quoting into
///       + maker rebate            <- venue pays us, 20% of the taker's fee
///       + expected reward accrual <- liquidity programme, time-weighted
///       - adverse selection       <- we get filled when we are wrong
///       - safety margin
/// ```
///
/// The first three terms are why a maker can profitably quote a price a taker
/// could not profitably cross.
/// `risk_credit` is the per-share value of the variance this fill would remove,
/// in micro-dollars. Zero for a quote that adds exposure.
///
/// # Why a reducing quote is worth more than its price says
///
/// The naive edge of a bid is `fair − price`, which for a bid placed *above*
/// fair value is negative. But when we are short, a bid that fills is not a
/// purchase at a bad price — it is the removal of a position we are paying to
/// carry. Judging it on price alone pulls the quote precisely when we most need
/// it, which is how a book gets stuck holding the wrong side.
///
/// The credit is the inventory penalty the fill would retire. It is bounded by
/// that penalty, so this cannot become a licence to buy at any price: once the
/// position is flat the credit is zero and ordinary economics resume.
#[allow(clippy::too_many_arguments)]
pub fn assess_make(
    book: &DenseBook,
    dir: Dir,
    fair: Px,
    price: Px,
    qty: Qty,
    fees: &FeeModel,
    rewards: &RewardModel,
    expected_rest_s: f64,
    two_sided: bool,
    sigma_p: f64,
    risk_credit: i32,
    p: &EdgeParams,
) -> MakeAssessment {
    let side = dir.make_side();
    let queue_ahead = book.size_at(side, price);
    let mid = book.mid().unwrap_or(price);
    let distance_ticks = ((mid.0 - price.0).abs() / book.tick.max(1)) as i32;

    let gross = match dir {
        Dir::Buy => fair.0 - price.0,
        Dir::Sell => price.0 - fair.0,
    };
    let rebate = fees.maker_per_share(price) as i32;
    let reward = rewards.credit_per_share_sec(distance_ticks, qty, two_sided) * expected_rest_s;
    let reward = reward.max(0.0).min(MAX_MARGIN as f64) as i32;
    // Same failure-closed rule as the margin: an unknown adverse-selection
    // estimate is prohibitive, not free.
    let adverse = if p.adverse.is_nan() {
        MAX_MARGIN
    } else {
        p.adverse.max(0.0).min(MAX_MARGIN as f64) as i32
    };
    let margin = margin_micro(sigma_p, p.safety_k);

    // Accumulate in i64 and saturate back. Every term is bounded above by
    // MAX_MARGIN, so this cannot wrap even with all five at their limits.
    let credit = risk_credit.clamp(0, MAX_MARGIN);
    let net = ((gross as i64) + (rebate as i64) + (reward as i64) + (credit as i64)
        - (adverse as i64)
        - (margin as i64))
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;

    MakeAssessment {
        price,
        qty,
        distance_ticks,
        queue_ahead,
        gross_per_share: gross,
        rebate_per_share: rebate,
        reward_per_share: reward,
        adverse_per_share: adverse,
        risk_credit_per_share: credit,
        margin_per_share: margin,
        net_per_share: net,
        viable: net >= p.min_edge,
    }
}

/// Online calibration of adverse selection.
///
/// After each passive fill we record where fair value moved over the following
/// mark-out horizon. The exponentially-weighted mean of that move *is* the
/// adverse selection cost, measured rather than guessed, and it feeds straight
/// back into `EdgeParams::adverse`.
///
/// This closes the loop the brief asks for in its self-calibration section: the
/// number that decides whether a quote is worth posting is estimated from the
/// consequences of the quotes we actually posted.
#[derive(Clone, Copy, Debug)]
pub struct AdverseSelectionEstimator {
    ewma: f64,
    alpha: f64,
    n: u64,
}

impl AdverseSelectionEstimator {
    pub fn new(alpha: f64) -> Self {
        AdverseSelectionEstimator {
            ewma: 0.0,
            alpha: alpha.clamp(1e-4, 1.0),
            n: 0,
        }
    }

    /// Record a mark-out. `fill_px` is where we were filled, `fair_after` is
    /// fair value at the mark-out horizon, `dir` is the side we were filled on.
    /// A positive result means the fill went against us.
    pub fn observe(&mut self, dir: Dir, fill_px: Px, fair_after: Px) {
        let loss = match dir {
            // We bought at fill_px; if fair fell below it, we lost the difference.
            Dir::Buy => (fill_px.0 - fair_after.0) as f64,
            Dir::Sell => (fair_after.0 - fill_px.0) as f64,
        };
        self.ewma = if self.n == 0 {
            loss
        } else {
            (1.0 - self.alpha) * self.ewma + self.alpha * loss
        };
        self.n += 1;
    }

    /// Current estimate, micro-dollars per share. Floored at zero: a negative
    /// adverse selection estimate means we have been getting favourable fills,
    /// which is not a reason to relax the hurdle.
    #[inline(always)]
    pub fn estimate(&self) -> f64 {
        self.ewma.max(0.0)
    }

    #[inline(always)]
    pub fn samples(&self) -> u64 {
        self.n
    }
}

impl Default for AdverseSelectionEstimator {
    fn default() -> Self {
        AdverseSelectionEstimator::new(0.02)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_core::Category;

    fn ladder() -> DenseBook {
        let mut b = DenseBook::new(10_000);
        b.set_level(Side::Bid, Px(480_000), Qty::shares(30));
        b.set_level(Side::Bid, Px(470_000), Qty::shares(100));
        b.set_level(Side::Bid, Px(460_000), Qty::shares(500));
        b.set_level(Side::Ask, Px(520_000), Qty::shares(25));
        b.set_level(Side::Ask, Px(530_000), Qty::shares(60));
        b.set_level(Side::Ask, Px(540_000), Qty::shares(400));
        b
    }

    fn crypto() -> FeeModel {
        FeeModel::for_category(Category::Crypto)
    }

    fn no_margin() -> EdgeParams {
        EdgeParams {
            safety_k: 0.0,
            min_edge: 0,
            adverse: 0.0,
            max_levels: 8,
        }
    }

    #[test]
    fn a_small_gross_edge_is_destroyed_by_the_taker_fee() {
        // Fair is 53.5c, the offer is 52c: 1.5 cents of apparent edge.
        // The crypto taker fee at 52c is 1.747c. The trade loses money.
        let b = ladder();
        let a = assess_take(
            &b,
            Dir::Buy,
            Px(535_000),
            Qty::shares(20),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert_eq!(a.avg_entry, Px(520_000));
        assert_eq!(a.gross_per_share, 15_000);
        assert!(
            (a.fee_per_share - 17_472).abs() < 50,
            "fee {}",
            a.fee_per_share
        );
        assert!(a.net_per_share < 0, "net {}", a.net_per_share);
        assert!(!a.viable);
    }

    #[test]
    fn the_same_edge_is_profitable_as_a_maker() {
        // Same 1.5c mispricing, quoted passively instead of crossed. No fee, a
        // rebate, and a reward credit. This single comparison is the reason the
        // engine is built around resting rather than sniping.
        let b = ladder();
        let rewards = RewardModel {
            pool_per_day: 300_000.0 * 1e6,
            est_total_q: 5e6,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::shares(50),
        };
        let m = assess_make(
            &b,
            Dir::Buy,
            Px(535_000),
            Px(520_000),
            Qty::shares(100),
            &crypto(),
            &rewards,
            30.0,
            true,
            0.0,
            0,
            &no_margin(),
        );
        assert_eq!(m.gross_per_share, 15_000);
        assert!(m.rebate_per_share > 3_000, "rebate {}", m.rebate_per_share);
        assert!(m.net_per_share > 15_000, "net {}", m.net_per_share);
        assert!(m.viable);
    }

    #[test]
    fn optimal_take_prefers_more_size_at_a_worse_average() {
        // Fair 60c. Level 1: 25 @ 52c. Level 2: 60 @ 53c. Level 3: 400 @ 54c.
        // Touch-only would capture 25 shares; the optimiser should go deeper.
        let b = ladder();
        let a = optimal_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(1000),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert!(a.viable);
        assert!(a.qty > Qty::shares(25), "qty {:?}", a.qty);
        assert_eq!(a.qty, Qty::shares(485));
        assert_eq!(a.levels, 3);

        // And the deeper trade genuinely beats the touch-only trade.
        let touch = assess_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(25),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert!(a.total_edge > touch.total_edge);
    }

    #[test]
    fn optimal_take_stops_before_the_edge_turns_negative() {
        // Fair 53.4c: level 1 (52c) clears the fee, level 2 (53c) does not.
        let b = ladder();
        let a = optimal_take(
            &b,
            Dir::Buy,
            Px(545_000),
            Qty::shares(1000),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert_eq!(a.qty, Qty::shares(25));
        assert_eq!(a.levels, 1);
        assert!(a.net_per_share > 0);
    }

    #[test]
    fn optimal_take_returns_nothing_when_no_size_is_viable() {
        let b = ladder();
        // Fair is below the best offer: no buy is ever right.
        let a = optimal_take(
            &b,
            Dir::Buy,
            Px(500_000),
            Qty::shares(1000),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert!(!a.viable);
        assert_eq!(a.qty, Qty::ZERO);
    }

    #[test]
    fn optimal_take_respects_the_level_cap() {
        let mut b = DenseBook::new(1_000);
        // Twenty thin levels, each one tick apart.
        for i in 0..20 {
            b.set_level(Side::Ask, Px(500_000 + i * 1_000), Qty::shares(5));
        }
        let p = EdgeParams {
            max_levels: 3,
            ..no_margin()
        };
        let a = optimal_take(
            &b,
            Dir::Buy,
            Px(900_000),
            Qty::shares(1000),
            &crypto(),
            0.0,
            &p,
        );
        assert!(a.levels <= 3, "levels {}", a.levels);
    }

    #[test]
    fn safety_margin_scales_with_model_uncertainty() {
        let b = ladder();
        let p = EdgeParams {
            safety_k: 2.0,
            min_edge: 0,
            adverse: 0.0,
            max_levels: 8,
        };
        // sigma_p = 0.02 (2 probability points) -> 2 sigma margin = 4c/share.
        let a = assess_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(20),
            &crypto(),
            0.02,
            &p,
        );
        assert_eq!(a.margin_per_share, 40_000);
        // Confident model, same edge: margin vanishes and the trade opens up.
        let confident = assess_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(20),
            &crypto(),
            0.001,
            &p,
        );
        assert_eq!(confident.margin_per_share, 2_000);
        assert!(confident.net_per_share > a.net_per_share);
    }

    #[test]
    fn a_stale_model_prices_itself_out_of_the_market() {
        // The self-limiting property: as sigma_p grows, nothing is viable.
        let b = ladder();
        let p = EdgeParams {
            safety_k: 1.0,
            min_edge: 0,
            adverse: 0.0,
            max_levels: 8,
        };
        let a = assess_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(20),
            &crypto(),
            0.50,
            &p,
        );
        assert!(!a.viable);
    }

    #[test]
    fn sell_direction_is_symmetric() {
        let b = ladder();
        // Fair 40c, best bid 48c: selling is 8c gross.
        let a = assess_take(
            &b,
            Dir::Sell,
            Px(400_000),
            Qty::shares(20),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert_eq!(a.avg_entry, Px(480_000));
        assert_eq!(a.gross_per_share, 80_000);
        assert!(a.viable);
    }

    #[test]
    fn make_assessment_reports_queue_position() {
        let b = ladder();
        let m = assess_make(
            &b,
            Dir::Buy,
            Px(535_000),
            Px(480_000),
            Qty::shares(50),
            &crypto(),
            &RewardModel::default(),
            10.0,
            true,
            0.0,
            0,
            &no_margin(),
        );
        // Joining the 48c bid means queueing behind the 30 shares already there.
        assert_eq!(m.queue_ahead, Qty::shares(30));
        assert_eq!(m.distance_ticks, 2);
    }

    #[test]
    fn adverse_selection_beyond_the_rebate_kills_the_quote() {
        let b = ladder();
        let p = EdgeParams {
            safety_k: 0.0,
            min_edge: 0,
            adverse: 20_000.0, // 2 cents of measured mark-out
            max_levels: 8,
        };
        let m = assess_make(
            &b,
            Dir::Buy,
            Px(535_000),
            Px(520_000),
            Qty::shares(100),
            &crypto(),
            &RewardModel::default(),
            30.0,
            true,
            0.0,
            0,
            &p,
        );
        assert!(m.net_per_share < 0, "net {}", m.net_per_share);
        assert!(!m.viable);
    }

    #[test]
    fn adverse_estimator_learns_from_bad_fills() {
        let mut e = AdverseSelectionEstimator::new(0.1);
        assert_eq!(e.estimate(), 0.0);
        // Every time we buy at 52c, fair is 51c a moment later: 1c of mark-out.
        for _ in 0..200 {
            e.observe(Dir::Buy, Px(520_000), Px(510_000));
        }
        assert!((e.estimate() - 10_000.0).abs() < 100.0, "{}", e.estimate());
        assert_eq!(e.samples(), 200);
    }

    #[test]
    fn adverse_estimator_floors_at_zero() {
        let mut e = AdverseSelectionEstimator::new(0.1);
        for _ in 0..200 {
            e.observe(Dir::Buy, Px(520_000), Px(540_000)); // favourable fills
        }
        assert_eq!(e.estimate(), 0.0);
    }

    #[test]
    fn empty_book_yields_no_assessment() {
        let b = DenseBook::new(10_000);
        let a = optimal_take(
            &b,
            Dir::Buy,
            Px(900_000),
            Qty::shares(100),
            &crypto(),
            0.0,
            &no_margin(),
        );
        assert!(!a.viable);
        assert_eq!(a.qty, Qty::ZERO);
    }

    #[test]
    fn a_risk_reducing_quote_is_worth_posting_at_negative_naive_edge() {
        // The bug this fixes: held short, a bid that covers sits *above* fair
        // value, so `fair - price` is negative and the quote reads as
        // unprofitable. Pulling it is the worst possible response — the fill
        // would have retired the variance we are paying to carry. Judged on
        // price alone the engine abandons its own risk management.
        let b = ladder();
        let no_credit = assess_make(
            &b,
            Dir::Buy,
            Px(500_000),
            Px(520_000), // bidding 2c above fair, to cover
            Qty::shares(100),
            &crypto(),
            &RewardModel::default(),
            10.0,
            true,
            0.0,
            0,
            &no_margin(),
        );
        assert!(no_credit.gross_per_share < 0);
        assert!(!no_credit.viable, "should not post without the credit");

        // Now with the variance the fill would retire priced in.
        let with_credit = assess_make(
            &b,
            Dir::Buy,
            Px(500_000),
            Px(520_000),
            Qty::shares(100),
            &crypto(),
            &RewardModel::default(),
            10.0,
            true,
            0.0,
            30_000, // 3c/share of carried risk retired
            &no_margin(),
        );
        assert_eq!(with_credit.risk_credit_per_share, 30_000);
        assert!(with_credit.viable, "covering quote should now post");
        assert!(with_credit.net_per_share > no_credit.net_per_share);
    }

    #[test]
    fn the_risk_credit_cannot_become_a_licence_to_overpay() {
        // Bounded and non-negative: a flat book gets no credit, and a hostile
        // value cannot manufacture unlimited willingness to pay.
        let b = ladder();
        let mk = |credit: i32| {
            assess_make(
                &b,
                Dir::Buy,
                Px(500_000),
                Px(900_000),
                Qty::shares(100),
                &crypto(),
                &RewardModel::default(),
                10.0,
                true,
                0.0,
                credit,
                &no_margin(),
            )
        };
        assert_eq!(mk(0).risk_credit_per_share, 0);
        assert_eq!(mk(-50_000).risk_credit_per_share, 0);
        assert_eq!(mk(i32::MAX).risk_credit_per_share, MAX_MARGIN);
        // Even at the ceiling, paying 90c for a 50c contract stays a bad trade
        // once the position is flat, because the credit is then zero.
        assert!(!mk(0).viable);
    }

    #[test]
    fn an_enormous_model_uncertainty_does_not_overflow() {
        // Regression: `margin_micro` used to saturate at `i32::MAX`, and the
        // downstream i32 arithmetic then wrapped. A saturating sentinel is not
        // a safe value to subtract. Found by the replay harness during a
        // simulated volatility shock, not by a unit test — which is the case
        // for the harness earning its keep.
        let b = ladder();
        let p = EdgeParams {
            safety_k: 1e9,
            min_edge: 0,
            adverse: 1e18,
            max_levels: 8,
        };
        for sigma in [1e6_f64, 1e30, f64::MAX, f64::INFINITY, f64::NAN] {
            let t = assess_take(
                &b,
                Dir::Buy,
                Px(600_000),
                Qty::shares(50),
                &crypto(),
                sigma,
                &p,
            );
            assert!(!t.viable);
            let o = optimal_take(
                &b,
                Dir::Buy,
                Px(600_000),
                Qty::shares(500),
                &crypto(),
                sigma,
                &p,
            );
            assert!(!o.viable);
            let m = assess_make(
                &b,
                Dir::Buy,
                Px(600_000),
                Px(480_000),
                Qty::shares(50),
                &crypto(),
                &RewardModel::default(),
                1e12,
                true,
                sigma,
                0,
                &p,
            );
            assert!(!m.viable);
        }
    }

    #[test]
    fn margin_is_clamped_to_a_prohibitive_but_finite_ceiling() {
        let p = EdgeParams {
            safety_k: 1.0,
            min_edge: 0,
            adverse: 0.0,
            max_levels: 8,
        };
        let b = ladder();
        // sigma_p of 50 probability points is nonsense, but must not wrap.
        let t = assess_take(
            &b,
            Dir::Buy,
            Px(600_000),
            Qty::shares(20),
            &crypto(),
            50.0,
            &p,
        );
        assert_eq!(t.margin_per_share, MAX_MARGIN);
        assert!(t.net_per_share < 0);
    }

    #[test]
    fn geopolitics_has_no_fee_hurdle() {
        // The one category where aggressive taking is structurally viable.
        let b = ladder();
        let geo = FeeModel::for_category(Category::Geopolitics);
        let a = assess_take(
            &b,
            Dir::Buy,
            Px(535_000),
            Qty::shares(20),
            &geo,
            0.0,
            &no_margin(),
        );
        assert_eq!(a.fee_per_share, 0);
        assert_eq!(a.net_per_share, 15_000);
        assert!(a.viable);
    }
}
