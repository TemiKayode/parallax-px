//! `px-risk` — limits, health, sizing gates, and the kill switch.
//!
//! Everything here is a *veto*. No component in this crate can cause a trade;
//! each can only reduce or refuse one. That asymmetry is deliberate: it means a
//! bug in the risk layer fails toward inaction, and the worst outcome of a
//! false positive is a missed trade rather than an unbounded position.

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

pub mod governor;
pub mod kelly;

pub use governor::{QuoteGovernor, RequoteVerdict, Tier, TokenBucket};
pub use kelly::{kelly_fraction, kelly_fraction_after_fee, kelly_shares};

use px_core::{Nanos, Px, Qty, Usd};

/// Number of correlation buckets. One per underlying, with room to spare.
pub const BUCKETS: usize = 16;

// ---------------------------------------------------------------------------
// Feed health
// ---------------------------------------------------------------------------

/// One monitored data source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feed {
    /// Reference spot prices (exchange or oracle).
    Reference,
    /// Polymarket market data websocket.
    MarketData,
    /// Polymarket authenticated order/trade stream.
    OrderUpdates,
    /// Settlement oracle TWAP relay.
    Oracle,
}

impl Feed {
    #[inline(always)]
    fn index(self) -> usize {
        match self {
            Feed::Reference => 0,
            Feed::MarketData => 1,
            Feed::OrderUpdates => 2,
            Feed::Oracle => 3,
        }
    }
    pub const ALL: [Feed; 4] = [
        Feed::Reference,
        Feed::MarketData,
        Feed::OrderUpdates,
        Feed::Oracle,
    ];
}

/// Why the kill switch tripped. Named causes, not a boolean, because the
/// recovery procedure differs per cause and the post-mortem needs to know.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultKind {
    None,
    StaleFeed(Feed),
    SequenceGap,
    CrossedBook,
    OracleDivergence,
    ApiAnomaly,
    LossLimit,
    Manual,
}

/// Watchdog over every input the strategy depends on.
///
/// The brief asks for a kill switch that fires "if feeds lag, become
/// unreliable, or the API behaves unexpectedly". Each of those is a separate
/// detector here, because they fail differently: a lagging feed recovers on its
/// own, a sequence gap requires a resync, and an API anomaly requires a human.
#[derive(Clone, Copy, Debug)]
pub struct FeedHealth {
    last_seen: [Nanos; 4],
    max_age: [f64; 4],
    fault: FaultKind,
    /// Consecutive healthy checks required before we resume after a fault.
    recovery_required: u32,
    recovery_count: u32,
}

// `last_seen`/`max_age` are both declared `[_; 4]`, and `Feed::index()`
// is an exhaustive match over `Feed`'s own 4 variants mapping each to
// `0..4` — every index used below comes from that method, so it is
// always in bounds by construction.
#[allow(clippy::indexing_slicing)]
impl FeedHealth {
    pub fn new(now: Nanos) -> Self {
        FeedHealth {
            last_seen: [now; 4],
            // Reference and market data must be near-continuous; the oracle
            // relay and our own order stream are permitted to be quieter.
            max_age: [1.0, 2.0, 30.0, 15.0],
            fault: FaultKind::None,
            recovery_required: 50,
            recovery_count: 0,
        }
    }

    #[inline(always)]
    pub fn touch(&mut self, f: Feed, now: Nanos) {
        self.last_seen[f.index()] = now;
    }

    pub fn set_max_age(&mut self, f: Feed, secs: f64) {
        self.max_age[f.index()] = secs;
    }

    /// Report a non-staleness fault. Sticky until `recovery_required`
    /// consecutive clean checks.
    pub fn fault(&mut self, k: FaultKind) {
        if k != FaultKind::None {
            self.fault = k;
            self.recovery_count = 0;
        }
    }

    /// Evaluate. Returns the current fault, if any.
    pub fn check(&mut self, now: Nanos) -> FaultKind {
        let mut stale = FaultKind::None;
        for f in Feed::ALL {
            let age = now.since(self.last_seen[f.index()]).as_secs_f64();
            if age > self.max_age[f.index()] {
                stale = FaultKind::StaleFeed(f);
                break;
            }
        }

        if stale != FaultKind::None {
            self.fault = stale;
            self.recovery_count = 0;
            return self.fault;
        }

        if self.fault != FaultKind::None {
            // Feeds look fine, but a sticky fault needs sustained health before
            // we trust it again. Flapping back into the market the instant a
            // gap closes is how a resync bug becomes a position.
            self.recovery_count = self.recovery_count.saturating_add(1);
            if self.recovery_count >= self.recovery_required {
                self.fault = FaultKind::None;
                self.recovery_count = 0;
            }
        }
        self.fault
    }

    #[inline(always)]
    pub fn healthy(&self) -> bool {
        self.fault == FaultKind::None
    }

    pub fn clear(&mut self) {
        self.fault = FaultKind::None;
        self.recovery_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Rare-loss guard
// ---------------------------------------------------------------------------

/// Guard for near-resolution strategies.
///
/// Buying a 97-cent contract that is genuinely 99% to settle at a dollar has a
/// positive expectation: `0.99*0.03 - 0.01*0.97 = +2.0 cents`. It also loses
/// 32 times its average win when it loses. Two things can ruin it, and neither
/// shows up in a Sharpe ratio computed over a good week:
///
/// 1. **One loss too large.** Sizing is capped so a single settlement against
///    us costs a bounded fraction of capital, regardless of what Kelly says.
///    Kelly assumes the probability is known; near resolution, the probability
///    is exactly what a settlement dispute or a final-second wick puts in
///    question.
/// 2. **The true loss rate is worse than modelled.** A 99%-win strategy losing
///    3% of the time is a losing strategy, but it takes a long time to notice by
///    watching P&L. A sequential probability ratio test notices in tens of
///    observations rather than thousands, because it asks the right question:
///    "is this evidence more consistent with 1% or with 3%?"
#[derive(Clone, Copy, Debug)]
pub struct RareLossGuard {
    /// Modelled loss rate under the null.
    p0: f64,
    /// Loss rate we want to detect.
    p1: f64,
    log_lr: f64,
    upper: f64,
    lower: f64,
    pub wins: u64,
    pub losses: u64,
    /// Maximum fraction of capital a single loss may cost.
    pub max_single_loss_frac: f64,
    tripped: bool,
}

impl RareLossGuard {
    /// `alpha` is the false-halt rate, `beta` the miss rate.
    pub fn new(p0: f64, p1: f64, alpha: f64, beta: f64, max_single_loss_frac: f64) -> Self {
        let p0 = p0.clamp(1e-6, 0.5);
        let p1 = p1.clamp(p0 + 1e-6, 0.9);
        RareLossGuard {
            p0,
            p1,
            log_lr: 0.0,
            upper: px_core::math::ln((1.0 - beta) / alpha),
            lower: px_core::math::ln(beta / (1.0 - alpha)),
            wins: 0,
            losses: 0,
            max_single_loss_frac,
            tripped: false,
        }
    }

    /// Record a settled near-resolution trade.
    pub fn observe(&mut self, lost: bool) {
        let inc = if lost {
            self.losses = self.losses.saturating_add(1);
            px_core::math::ln(self.p1 / self.p0)
        } else {
            self.wins = self.wins.saturating_add(1);
            px_core::math::ln((1.0 - self.p1) / (1.0 - self.p0))
        };
        self.log_lr += inc;

        if self.log_lr >= self.upper {
            self.tripped = true;
        } else if self.log_lr <= self.lower {
            // Evidence favours the null; reset and keep watching.
            self.log_lr = 0.0;
        }
    }

    #[inline(always)]
    pub fn tripped(&self) -> bool {
        self.tripped
    }

    #[inline(always)]
    pub fn log_likelihood_ratio(&self) -> f64 {
        self.log_lr
    }

    pub fn reset(&mut self) {
        self.log_lr = 0.0;
        self.tripped = false;
    }

    /// Largest position whose total loss stays inside the single-loss cap.
    /// A share bought at `entry` loses `entry` if the outcome goes against us.
    pub fn max_shares(&self, bankroll: Usd, entry: Px) -> Qty {
        if self.tripped || entry.0 <= 0 {
            return Qty::ZERO;
        }
        let budget = bankroll.as_f64() * self.max_single_loss_frac;
        let shares = budget / entry.as_f64();
        Qty((shares * 1e6).max(0.0) as i64)
    }
}

impl Default for RareLossGuard {
    fn default() -> Self {
        // Null: 1% loss rate. Alternative worth halting for: 3%.
        RareLossGuard::new(0.01, 0.03, 0.01, 0.05, 0.005)
    }
}

// ---------------------------------------------------------------------------
// Exposure limits
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct RiskLimits {
    pub max_capital_per_market: Usd,
    /// Absolute cap on net (unhedged) shares in any one market.
    pub max_unhedged_shares: i64,
    /// Directional exposure cap per correlation bucket.
    pub max_bucket_exposure: Usd,
    /// Total gross notional across everything.
    pub max_gross_exposure: Usd,
    /// Fraction of full Kelly to size at.
    pub kelly_fraction: f64,
    /// Realised + unrealised loss, measured from the session's equity high
    /// water mark, at which the engine halts and requires a human.
    ///
    /// `FaultKind::LossLimit` existed from the first version and nothing ever
    /// set it — the enum arm was a comment. A limit that is declared but not
    /// wired is worse than no limit, because it reads like protection.
    pub max_drawdown: Usd,
}

impl Default for RiskLimits {
    fn default() -> Self {
        RiskLimits {
            max_capital_per_market: Usd::dollars(5_000),
            max_unhedged_shares: Qty::shares(2_000).0,
            max_bucket_exposure: Usd::dollars(20_000),
            max_gross_exposure: Usd::dollars(100_000),
            kelly_fraction: 0.2,
            max_drawdown: Usd::dollars(5_000),
        }
    }
}

/// Running exposure, aggregated the way risk actually accrues.
///
/// # Gross is derived, not accumulated
///
/// The first version kept a running `gross += signed_notional.abs()` on every
/// fill. That is turnover, not exposure. Buying $1,000 and then selling it back
/// left the bucket correctly at zero and `gross` at $2,000 — so gross grew
/// monotonically with trading activity and, after enough round trips, exceeded
/// the limit permanently. The risk gate would then reject every order forever,
/// with the position flat and nothing actually at risk.
///
/// A market maker turns its book over hundreds of times a session, so this was
/// not a corner case; it was a guaranteed wedge on a long enough run. Gross is
/// now computed from the per-bucket state, which is the only thing that
/// represents position.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExposureBook {
    /// Signed directional notional per correlation bucket, micro-dollars.
    bucket: [i64; BUCKETS],
    /// Notional we have approved but not yet seen filled. Reserved at the gate
    /// and released on fill or cancel, so two markets cannot each be told they
    /// have the last of the headroom.
    reserved: [i64; BUCKETS],
    /// Lifetime traded notional. Diagnostics only — never a limit.
    turnover: i64,
}

// Every `[bucket]`/`[b]` access below is behind an explicit `bucket <
// BUCKETS` (or `b < BUCKETS`) guard — provably in range, the same pattern
// `px_core::book`'s `idx()`-guarded indexing already establishes.
#[allow(clippy::indexing_slicing)]
impl ExposureBook {
    pub fn new() -> Self {
        ExposureBook::default()
    }

    #[inline]
    pub fn add(&mut self, bucket: usize, signed_notional: i64) {
        if bucket < BUCKETS {
            self.bucket[bucket] = self.bucket[bucket].saturating_add(signed_notional);
        }
        self.turnover = self
            .turnover
            .saturating_add(signed_notional.saturating_abs());
    }

    /// Reserve headroom for an order that has been approved but not filled.
    #[inline]
    pub fn reserve(&mut self, bucket: usize, notional: i64) {
        if bucket < BUCKETS {
            self.reserved[bucket] = self.reserved[bucket].saturating_add(notional.saturating_abs());
        }
    }

    /// Release a reservation, on fill or on cancel. Floors at zero so a
    /// double-release cannot manufacture headroom.
    #[inline]
    pub fn release(&mut self, bucket: usize, notional: i64) {
        if bucket < BUCKETS {
            self.reserved[bucket] = self.reserved[bucket]
                .saturating_sub(notional.saturating_abs())
                .max(0);
        }
    }

    /// Directional exposure in a bucket, including anything in flight.
    #[inline(always)]
    pub fn bucket_exposure(&self, b: usize) -> i64 {
        if b < BUCKETS {
            self.bucket[b]
                .saturating_abs()
                .saturating_add(self.reserved[b])
        } else {
            0
        }
    }

    /// Signed exposure, without reservations. For reporting.
    #[inline(always)]
    pub fn bucket_signed(&self, b: usize) -> i64 {
        if b < BUCKETS {
            self.bucket[b]
        } else {
            0
        }
    }

    /// Gross exposure: the sum of absolute positions across buckets, plus
    /// anything reserved. Derived from state, so it falls when we flatten.
    #[inline]
    pub fn gross(&self) -> i64 {
        let mut g: i64 = 0;
        for b in 0..BUCKETS {
            g = g
                .saturating_add(self.bucket[b].saturating_abs())
                .saturating_add(self.reserved[b]);
        }
        g
    }

    #[inline(always)]
    pub fn turnover(&self) -> i64 {
        self.turnover
    }

    pub fn clear(&mut self) {
        self.bucket = [0; BUCKETS];
        self.reserved = [0; BUCKETS];
    }

    /// Scale the gross limit by how independent the current correlation regime
    /// actually is.
    ///
    /// When the cross-asset monitor reports one effective degree of freedom,
    /// every position is the same position and the portfolio carries the risk
    /// of a single concentrated bet. The limit contracts to match. This is the
    /// correlation aggregation the brief asks for, applied at the portfolio
    /// level rather than pairwise.
    pub fn effective_gross_limit(base: Usd, effective_dof: f64, n_assets: f64) -> Usd {
        let dof = effective_dof.clamp(1.0, n_assets.max(1.0));
        Usd((base.0 as f64 * (dof / n_assets.max(1.0)).sqrt()) as i64)
    }
}

/// `qty_micro * price_micro / 1e6`, in i128 so it cannot wrap — same
/// helper `px_inventory::Position::cash` already establishes for the
/// identical shape. The multiplication cannot overflow `i128` for any
/// `i64`/`i32` input (`i64::MAX * i32::MAX` ≈ 1.98e28, far under
/// `i128::MAX` ≈ 1.7e38); the cast back to `i64` is not similarly proven,
/// which is why every caller feeds the result into a `saturating_*`
/// exposure update rather than trusting it directly.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
fn notional_micro(qty_micro: i64, price_micro: i32) -> i64 {
    (qty_micro as i128 * price_micro as i128 / 1_000_000) as i64
}

/// The verdict on a proposed trade.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Approved at the requested size.
    Approved(Qty),
    /// Approved, but smaller.
    Reduced(Qty),
    Rejected(RejectReason),
}

impl Verdict {
    #[inline(always)]
    pub fn size(self) -> Qty {
        match self {
            Verdict::Approved(q) | Verdict::Reduced(q) => q,
            Verdict::Rejected(_) => Qty::ZERO,
        }
    }

    #[inline(always)]
    pub fn is_rejected(self) -> bool {
        matches!(self, Verdict::Rejected(_))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RejectReason {
    KillSwitch,
    NoEdge,
    MarketCapital,
    UnhedgedShares,
    BucketExposure,
    GrossExposure,
    RareLossHalt,
    SizeBelowMinimum,
}

/// A proposed trade, presented to the gate.
#[derive(Clone, Copy, Debug)]
pub struct TradeRequest {
    pub bucket: usize,
    pub entry: Px,
    pub want: Qty,
    /// Fair probability of the side we are buying.
    pub fair_p: f64,
    /// Taker fee per share at the entry price, micro-dollars. Zero when making.
    pub fee_per_share: f64,
    /// Net position already held in this market, in shares (micro).
    pub existing_net: i64,
    /// Capital already committed to this market, micro-dollars.
    pub existing_capital: i64,
    pub min_size: Qty,
    /// Whether this is a near-resolution trade, subject to the rare-loss guard.
    pub near_resolution: bool,
}

/// The single gate every order passes through.
#[derive(Debug)]
pub struct RiskGate {
    pub limits: RiskLimits,
    pub health: FeedHealth,
    pub exposure: ExposureBook,
    pub rare_loss: RareLossGuard,
    pub bankroll: Usd,
    /// Effective degrees of freedom from the cross-asset monitor.
    pub effective_dof: f64,
    pub n_assets: f64,
    pub manual_halt: bool,
    /// Session equity high water mark, micro-dollars.
    peak_equity: i64,
    /// Latest equity observation, micro-dollars.
    last_equity: i64,
    /// Whether the drawdown limit has been breached. Sticky: only a human
    /// clears it. A drawdown limit that resets itself when the market bounces
    /// is a suggestion, not a limit.
    drawdown_halt: bool,
}

impl RiskGate {
    pub fn new(limits: RiskLimits, bankroll: Usd, now: Nanos) -> Self {
        RiskGate {
            limits,
            health: FeedHealth::new(now),
            exposure: ExposureBook::new(),
            rare_loss: RareLossGuard::default(),
            bankroll,
            effective_dof: 7.0,
            n_assets: 7.0,
            manual_halt: false,
            peak_equity: 0,
            last_equity: 0,
            drawdown_halt: false,
        }
    }

    /// Report mark-to-model equity, in micro-dollars, relative to session start.
    ///
    /// Called from the same loop that marks the book. Trips `LossLimit` when the
    /// decline from the high water mark exceeds the configured drawdown.
    pub fn observe_equity(&mut self, equity: i64) {
        self.last_equity = equity;
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        // `peak_equity >= equity` always holds here: either this equity
        // just became the new peak, or the check above left it unchanged
        // because it was already `>=`.
        if self.peak_equity.saturating_sub(equity) >= self.limits.max_drawdown.0 {
            self.drawdown_halt = true;
            self.health.fault(FaultKind::LossLimit);
        }
    }

    #[inline(always)]
    pub fn drawdown(&self) -> i64 {
        // `peak_equity >= last_equity` is an invariant of every
        // `observe_equity` call — same reasoning as there.
        self.peak_equity.saturating_sub(self.last_equity)
    }

    #[inline(always)]
    pub fn is_halted(&self) -> bool {
        self.manual_halt || self.drawdown_halt
    }

    /// Clear a drawdown halt. Deliberately explicit and deliberately separate
    /// from feed recovery: resuming after a loss limit is a human decision.
    pub fn clear_drawdown_halt(&mut self) {
        self.drawdown_halt = false;
        self.peak_equity = self.last_equity;
        self.health.clear();
    }

    /// Evaluate a proposed trade. Applies, in order: kill switch, Kelly sizing,
    /// per-market capital, unhedged share cap, correlation-bucket cap, gross
    /// cap, rare-loss cap, minimum size.
    pub fn check(&mut self, req: &TradeRequest, now: Nanos) -> Verdict {
        if self.is_halted() {
            return Verdict::Rejected(RejectReason::KillSwitch);
        }
        if self.health.check(now) != FaultKind::None {
            return Verdict::Rejected(RejectReason::KillSwitch);
        }

        // --- Kelly ceiling ---
        let f = kelly_fraction_after_fee(req.fair_p, req.entry, req.fee_per_share);
        if f <= 0.0 {
            return Verdict::Rejected(RejectReason::NoEdge);
        }
        let kelly_cap = kelly_shares(
            req.fair_p,
            Px((req.entry.0 as f64 + req.fee_per_share) as i32).clamp_unit(),
            self.bankroll,
            self.limits.kelly_fraction,
        );

        let mut q = req.want.min(kelly_cap);
        let mut reduced = q != req.want;

        // --- Per-market capital ---
        let room_micro = self
            .limits
            .max_capital_per_market
            .0
            .saturating_sub(req.existing_capital);
        if room_micro <= 0 {
            return Verdict::Rejected(RejectReason::MarketCapital);
        }
        let entry = req.entry.as_f64().max(1e-9);
        let cap_shares = Qty(((room_micro as f64) / entry) as i64);
        if cap_shares < q {
            q = cap_shares;
            reduced = true;
        }

        // --- Unhedged share cap ---
        let headroom = self
            .limits
            .max_unhedged_shares
            .saturating_sub(req.existing_net.saturating_abs());
        if headroom <= 0 {
            return Verdict::Rejected(RejectReason::UnhedgedShares);
        }
        if Qty(headroom) < q {
            q = Qty(headroom);
            reduced = true;
        }

        // --- Correlation bucket ---
        let bucket_used = self.exposure.bucket_exposure(req.bucket).saturating_abs();
        let bucket_room = self
            .limits
            .max_bucket_exposure
            .0
            .saturating_sub(bucket_used);
        if bucket_room <= 0 {
            return Verdict::Rejected(RejectReason::BucketExposure);
        }
        let bucket_shares = Qty(((bucket_room as f64) / entry) as i64);
        if bucket_shares < q {
            q = bucket_shares;
            reduced = true;
        }

        // --- Gross, scaled by the correlation regime ---
        let gross_limit = ExposureBook::effective_gross_limit(
            self.limits.max_gross_exposure,
            self.effective_dof,
            self.n_assets,
        );
        let gross_room = gross_limit.0.saturating_sub(self.exposure.gross());
        if gross_room <= 0 {
            return Verdict::Rejected(RejectReason::GrossExposure);
        }
        let gross_shares = Qty(((gross_room as f64) / entry) as i64);
        if gross_shares < q {
            q = gross_shares;
            reduced = true;
        }

        // --- Rare-loss guard ---
        if req.near_resolution {
            if self.rare_loss.tripped() {
                return Verdict::Rejected(RejectReason::RareLossHalt);
            }
            let cap = self.rare_loss.max_shares(self.bankroll, req.entry);
            if cap < q {
                q = cap;
                reduced = true;
            }
        }

        if q < req.min_size || q.0 <= 0 {
            return Verdict::Rejected(RejectReason::SizeBelowMinimum);
        }

        // Reserve the headroom we just granted.
        //
        // Without this, two markets in the same correlation bucket evaluated in
        // the same tick are each told they have the full remaining headroom,
        // because neither has filled yet. Both get approved, both fill, and the
        // bucket limit is breached by up to a factor of the number of markets
        // sharing it. The reservation is released by `on_fill` or `on_reject`.
        let notional = notional_micro(q.0, req.entry.0);
        self.exposure.reserve(req.bucket, notional);

        if reduced {
            Verdict::Reduced(q)
        } else {
            Verdict::Approved(q)
        }
    }

    /// Record an executed trade against the exposure book, releasing the
    /// reservation the gate took when it approved the order.
    pub fn on_fill(&mut self, bucket: usize, signed_qty: i64, px: Px) {
        let notional = notional_micro(signed_qty, px.0);
        self.exposure.release(bucket, notional);
        self.exposure.add(bucket, notional);
    }

    /// An approved order died without filling. Release its reservation, or the
    /// headroom leaks and the gate tightens for no reason.
    pub fn on_reject(&mut self, bucket: usize, qty: Qty, px: Px) {
        let notional = notional_micro(qty.0, px.0);
        self.exposure.release(bucket, notional);
    }

    /// Largest passive quote we are willing to rest on the *adding* side.
    ///
    /// # Why passive orders need a gate at all
    ///
    /// Only aggressive orders passed through `check`. Resting quotes were sized
    /// by the inventory throttle alone, and that throttle is a soft multiplier,
    /// not a limit — it scales the base size down but never consults the
    /// per-market capital cap, the unhedged share cap, or the bucket limit. A
    /// steady stream of passive fills could therefore walk straight past every
    /// hard limit in this file without one of them being evaluated.
    ///
    /// Resting size is *not* reserved, deliberately. A quote that may never fill
    /// should not consume headroom an aggressive order could use; reserving it
    /// would make the bot quote far less than it safely can. The cap is applied
    /// at post time and re-applied on every requote, which is the right
    /// treatment for an order whose fill is contingent.
    ///
    /// Returns `Qty::ZERO` when halted or out of room — the caller should then
    /// quote one-sided rather than not at all.
    pub fn max_passive_size(
        &mut self,
        bucket: usize,
        entry: Px,
        existing_net: i64,
        existing_capital: i64,
        now: Nanos,
    ) -> Qty {
        if self.is_halted() || self.health.check(now) != FaultKind::None {
            return Qty::ZERO;
        }
        let px = entry.as_f64();
        if px <= 0.0 {
            return Qty::ZERO;
        }

        let mut cap = i64::MAX;

        let market_room = self
            .limits
            .max_capital_per_market
            .0
            .saturating_sub(existing_capital);
        cap = cap.min(((market_room.max(0) as f64) / px) as i64);

        let share_room = self
            .limits
            .max_unhedged_shares
            .saturating_sub(existing_net.saturating_abs());
        cap = cap.min(share_room.max(0));

        let bucket_room = self
            .limits
            .max_bucket_exposure
            .0
            .saturating_sub(self.exposure.bucket_exposure(bucket));
        cap = cap.min(((bucket_room.max(0) as f64) / px) as i64);

        let gross_limit = ExposureBook::effective_gross_limit(
            self.limits.max_gross_exposure,
            self.effective_dof,
            self.n_assets,
        );
        let gross_room = gross_limit.0.saturating_sub(self.exposure.gross());
        cap = cap.min(((gross_room.max(0) as f64) / px) as i64);

        Qty(cap.max(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> RiskGate {
        RiskGate::new(RiskLimits::default(), Usd::dollars(100_000), Nanos::ZERO)
    }

    fn req() -> TradeRequest {
        TradeRequest {
            bucket: 0,
            entry: Px(500_000),
            want: Qty::shares(100),
            fair_p: 0.60,
            fee_per_share: 0.0,
            existing_net: 0,
            existing_capital: 0,
            min_size: Qty::shares(5),
            near_resolution: false,
        }
    }

    fn fresh(g: &mut RiskGate, now: Nanos) {
        for f in Feed::ALL {
            g.health.touch(f, now);
        }
    }

    #[test]
    fn a_healthy_gate_approves_a_sound_trade() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let v = g.check(&req(), Nanos::from_millis(100));
        assert_eq!(v, Verdict::Approved(Qty::shares(100)));
    }

    #[test]
    fn a_stale_reference_feed_stops_everything() {
        let mut g = gate();
        fresh(&mut g, Nanos::ZERO);
        // Reference feed tolerance is 1 second.
        let v = g.check(&req(), Nanos::from_millis(1500));
        assert_eq!(v, Verdict::Rejected(RejectReason::KillSwitch));
        assert!(!g.health.healthy());
    }

    #[test]
    fn recovery_requires_sustained_health_not_a_single_good_tick() {
        let mut g = gate();
        fresh(&mut g, Nanos::ZERO);
        g.check(&req(), Nanos::from_millis(1500)); // trips
        assert!(!g.health.healthy());

        let mut t = 1500u64;
        // One clean check is not enough.
        t += 10;
        fresh(&mut g, Nanos::from_millis(t));
        g.health.check(Nanos::from_millis(t));
        assert!(!g.health.healthy());

        // Fifty are.
        for _ in 0..60 {
            t += 10;
            fresh(&mut g, Nanos::from_millis(t));
            g.health.check(Nanos::from_millis(t));
        }
        assert!(g.health.healthy());
    }

    #[test]
    fn sticky_faults_survive_healthy_feeds() {
        let mut g = gate();
        fresh(&mut g, Nanos::ZERO);
        g.health.fault(FaultKind::SequenceGap);
        assert_eq!(
            g.check(&req(), Nanos::from_millis(10)),
            Verdict::Rejected(RejectReason::KillSwitch)
        );
    }

    #[test]
    fn no_edge_is_rejected_outright() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.fair_p = 0.45;
        assert_eq!(
            g.check(&r, Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::NoEdge)
        );
    }

    #[test]
    fn the_fee_alone_can_turn_an_edge_into_no_edge() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.entry = Px(520_000);
        r.fair_p = 0.53; // one cent of apparent edge
        r.fee_per_share = 17_472.0; // crypto taker fee at 52c
        assert_eq!(
            g.check(&r, Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::NoEdge)
        );
    }

    #[test]
    fn kelly_caps_an_oversized_request() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.want = Qty::shares(1_000_000);
        let v = g.check(&r, Nanos::from_millis(100));
        // 20% Kelly on $100k at p=0.6, c=0.5 is 8,000 shares — but the
        // per-market capital cap of $5,000 binds first at 10,000 shares.
        assert!(matches!(v, Verdict::Reduced(_)));
        assert!(v.size() <= Qty::shares(8_000));
        assert!(v.size() > Qty::shares(1_000));
    }

    #[test]
    fn per_market_capital_binds() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.want = Qty::shares(100_000);
        r.existing_capital = Usd::dollars(4_900).0; // $100 of room at 50c = 200 shares
        let v = g.check(&r, Nanos::from_millis(100));
        assert!(v.size() <= Qty::shares(200));
    }

    #[test]
    fn a_full_market_is_rejected_not_reduced_to_zero() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.existing_capital = Usd::dollars(5_000).0;
        assert_eq!(
            g.check(&r, Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::MarketCapital)
        );
    }

    #[test]
    fn unhedged_share_cap_binds() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.want = Qty::shares(1000);
        r.existing_net = Qty::shares(1_950).0; // 50 shares of headroom
        let v = g.check(&r, Nanos::from_millis(100));
        assert_eq!(v.size(), Qty::shares(50));
    }

    #[test]
    fn correlated_markets_share_one_bucket_limit() {
        // BTC 5m and BTC 15m are the same bet. Filling one must consume the
        // other's headroom.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));

        // Take $19,900 of BTC exposure in bucket 0.
        g.on_fill(0, Qty::shares(39_800).0, Px(500_000));
        assert!(g.exposure.bucket_exposure(0) > Usd::dollars(19_000).0);

        let mut r = req();
        r.bucket = 0;
        r.want = Qty::shares(1000);
        let v = g.check(&r, Nanos::from_millis(100));
        // Only ~$100 of bucket room left: about 200 shares at 50c.
        assert!(v.size() <= Qty::shares(210), "size {:?}", v.size());

        // A different bucket is unaffected.
        let mut r2 = req();
        r2.bucket = 3;
        assert_eq!(
            g.check(&r2, Nanos::from_millis(100)).size(),
            Qty::shares(100)
        );
    }

    #[test]
    fn the_gross_limit_contracts_when_correlations_collapse() {
        let base = Usd::dollars(100_000);
        let independent = ExposureBook::effective_gross_limit(base, 7.0, 7.0);
        let one_bet = ExposureBook::effective_gross_limit(base, 1.0, 7.0);
        assert_eq!(independent, base);
        // sqrt(1/7) = 0.378
        assert!((one_bet.as_f64() / base.as_f64() - 0.377_96).abs() < 1e-3);
        assert!(one_bet < independent);
    }

    #[test]
    fn a_collapsed_correlation_regime_actually_reduces_approved_size() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        // $38,000 of gross already on the book. The dof=7 limit is $100,000;
        // the dof=1 limit is $100,000 * sqrt(1/7) = $37,796 — already breached.
        g.on_fill(0, Qty::shares(76_000).0, Px(500_000));
        let mut r = req();
        r.bucket = 5;
        r.want = Qty::shares(1000);

        g.effective_dof = 7.0;
        let wide = g.check(&r, Nanos::from_millis(100));
        assert_eq!(wide, Verdict::Approved(Qty::shares(1000)));

        g.effective_dof = 1.0;
        let tight = g.check(&r, Nanos::from_millis(100));
        assert_eq!(tight, Verdict::Rejected(RejectReason::GrossExposure));
    }

    #[test]
    fn sprt_halts_a_near_resolution_strategy_that_is_losing_too_often() {
        let mut guard = RareLossGuard::new(0.01, 0.03, 0.01, 0.05, 0.005);
        // A true 1% loss rate: 99 wins per loss. Should not trip.
        for i in 0..2000 {
            guard.observe(i % 100 == 0);
        }
        assert!(!guard.tripped(), "llr = {}", guard.log_likelihood_ratio());

        // Now a 6% loss rate. Should trip, and quickly.
        let mut bad = RareLossGuard::new(0.01, 0.03, 0.01, 0.05, 0.005);
        let mut n = 0;
        for i in 0..2000 {
            bad.observe(i % 16 == 0);
            n += 1;
            if bad.tripped() {
                break;
            }
        }
        assert!(bad.tripped());
        assert!(n < 400, "took {n} observations to notice");
    }

    #[test]
    fn a_tripped_rare_loss_guard_blocks_near_resolution_trades_only() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        for _ in 0..500 {
            g.rare_loss.observe(true);
        }
        assert!(g.rare_loss.tripped());

        let mut nr = req();
        nr.near_resolution = true;
        assert_eq!(
            g.check(&nr, Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::RareLossHalt)
        );

        // Ordinary quoting continues.
        assert!(!g.check(&req(), Nanos::from_millis(100)).is_rejected());
    }

    #[test]
    fn single_loss_cap_bounds_a_near_resolution_position() {
        let guard = RareLossGuard::new(0.01, 0.03, 0.01, 0.05, 0.005);
        // 0.5% of a $100k bankroll is $500. At 97c that is ~515 shares, and a
        // loss costs exactly $500.
        let q = guard.max_shares(Usd::dollars(100_000), Px(970_000));
        assert!((q.as_f64() - 515.46).abs() < 1.0, "{}", q.as_f64());
        let loss = q.as_f64() * 0.97;
        assert!((loss - 500.0).abs() < 1.0);
    }

    #[test]
    fn sizes_below_the_venue_minimum_are_rejected_not_rounded_up() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut r = req();
        r.existing_net = Qty::shares(1_999).0; // 1 share of headroom
        r.min_size = Qty::shares(5);
        assert_eq!(
            g.check(&r, Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::SizeBelowMinimum)
        );
    }

    #[test]
    fn gross_exposure_falls_when_we_flatten() {
        // Regression: gross used to accumulate `|notional|` on every fill, so it
        // measured turnover rather than exposure. A market maker turning its
        // book over would eventually exceed the limit with a flat position and
        // reject every order thereafter — permanently.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));

        g.on_fill(0, Qty::shares(2_000).0, Px(500_000)); // buy $1,000
        let after_buy = g.exposure.gross();
        assert!(after_buy > 0);

        g.on_fill(0, -Qty::shares(2_000).0, Px(500_000)); // sell it back
        assert_eq!(g.exposure.gross(), 0, "gross did not return to zero");
        assert!(g.exposure.turnover() > 0, "turnover should still record it");

        // And the gate still works after heavy churn.
        for _ in 0..500 {
            g.on_fill(0, Qty::shares(2_000).0, Px(500_000));
            g.on_fill(0, -Qty::shares(2_000).0, Px(500_000));
        }
        assert_eq!(g.exposure.gross(), 0);
        assert!(!g.check(&req(), Nanos::from_millis(100)).is_rejected());
    }

    #[test]
    fn approved_orders_reserve_headroom_against_each_other() {
        // Two markets in one bucket, evaluated before either fills. Without
        // reservations both are told they have the full remaining headroom and
        // the bucket limit is breached by 2x.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        g.limits.max_bucket_exposure = Usd::dollars(1_000);

        let mut r = req();
        r.want = Qty::shares(2_000); // $1,000 at 50c — exactly the limit
        let first = g.check(&r, Nanos::from_millis(100));
        assert!(!first.is_rejected());

        // Second market, same bucket, nothing filled yet.
        let second = g.check(&r, Nanos::from_millis(100));
        assert!(
            second.is_rejected() || second.size() < first.size(),
            "second order got {:?} on top of {:?}",
            second.size(),
            first.size()
        );
    }

    #[test]
    fn reservations_are_released_on_fill_and_on_reject() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let base = g.exposure.gross();

        let v = g.check(&req(), Nanos::from_millis(100));
        assert!(g.exposure.gross() > base, "reservation not taken");

        g.on_reject(0, v.size(), Px(500_000));
        assert_eq!(g.exposure.gross(), base, "reservation leaked on reject");

        let v2 = g.check(&req(), Nanos::from_millis(100));
        g.on_fill(0, v2.size().0, Px(500_000));
        // After the fill, exposure reflects the position, not the reservation.
        assert_eq!(g.exposure.gross(), g.exposure.bucket_signed(0).abs());
    }

    #[test]
    fn the_drawdown_limit_actually_halts() {
        // `FaultKind::LossLimit` existed from the first version and nothing set
        // it. A limit that is declared but not wired reads like protection.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        g.limits.max_drawdown = Usd::dollars(1_000);

        g.observe_equity(Usd::dollars(2_000).0); // peak
        assert!(!g.is_halted());
        assert!(!g.check(&req(), Nanos::from_millis(100)).is_rejected());

        g.observe_equity(Usd::dollars(1_500).0); // -$500, within limit
        assert!(!g.is_halted());

        g.observe_equity(Usd::dollars(900).0); // -$1,100 from peak
        assert!(g.is_halted());
        assert_eq!(
            g.check(&req(), Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::KillSwitch)
        );
    }

    #[test]
    fn a_drawdown_halt_does_not_clear_itself_on_a_bounce() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        g.limits.max_drawdown = Usd::dollars(1_000);
        g.observe_equity(Usd::dollars(5_000).0);
        g.observe_equity(Usd::dollars(3_000).0);
        assert!(g.is_halted());

        // Market recovers fully. We stay halted: resuming is a human decision.
        g.observe_equity(Usd::dollars(6_000).0);
        assert!(g.is_halted());

        g.clear_drawdown_halt();
        fresh(&mut g, Nanos::from_millis(200));
        assert!(!g.is_halted());
        assert!(!g.check(&req(), Nanos::from_millis(200)).is_rejected());
    }

    #[test]
    fn passive_quotes_respect_the_hard_limits() {
        // Regression: only aggressive orders passed through `check`. Resting
        // quotes were sized by the inventory throttle alone — a soft multiplier
        // that never consults the capital, share, or bucket caps. A steady
        // stream of passive fills could walk past every hard limit here.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));

        // Fresh book. The capital cap allows 10,000 shares at 50c, but the
        // 2,000-share unhedged limit is tighter and is the one that binds.
        // Whichever is smallest must win — that is the whole point.
        let cap = g.max_passive_size(0, Px(500_000), 0, 0, Nanos::from_millis(100));
        assert_eq!(cap, Qty::shares(2_000));

        // Near the unhedged share limit, that binds instead.
        let cap = g.max_passive_size(
            0,
            Px(500_000),
            Qty::shares(1_900).0,
            0,
            Nanos::from_millis(100),
        );
        assert_eq!(cap, Qty::shares(100));

        // At the limit, nothing.
        let cap = g.max_passive_size(
            0,
            Px(500_000),
            Qty::shares(2_000).0,
            0,
            Nanos::from_millis(100),
        );
        assert_eq!(cap, Qty::ZERO);
    }

    #[test]
    fn passive_sizing_honours_the_bucket_and_the_kill_switch() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));

        // Fill bucket 0 nearly to its $20,000 ceiling.
        g.on_fill(0, Qty::shares(39_800).0, Px(500_000));
        let cap = g.max_passive_size(0, Px(500_000), 0, 0, Nanos::from_millis(100));
        assert!(cap <= Qty::shares(250), "bucket cap ignored: {cap:?}");

        // A different bucket is unaffected.
        let other = g.max_passive_size(3, Px(500_000), 0, 0, Nanos::from_millis(100));
        assert!(other > cap);

        // And a halt stops passive quoting outright, not just aggression.
        g.manual_halt = true;
        assert_eq!(
            g.max_passive_size(3, Px(500_000), 0, 0, Nanos::from_millis(100)),
            Qty::ZERO
        );
    }

    #[test]
    fn passive_sizing_does_not_consume_reservations() {
        // A quote that may never fill must not eat headroom an aggressive order
        // could use — otherwise the bot quotes far smaller than it safely can.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let before = g.exposure.gross();
        g.max_passive_size(0, Px(500_000), 0, 0, Nanos::from_millis(100));
        g.max_passive_size(0, Px(500_000), 0, 0, Nanos::from_millis(100));
        assert_eq!(g.exposure.gross(), before);
    }

    #[test]
    fn manual_halt_overrides_everything() {
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        g.manual_halt = true;
        assert_eq!(
            g.check(&req(), Nanos::from_millis(100)),
            Verdict::Rejected(RejectReason::KillSwitch)
        );
    }

    #[test]
    fn the_gate_can_only_ever_reduce() {
        // Property test: no input produces a size larger than requested.
        let mut g = gate();
        fresh(&mut g, Nanos::from_millis(100));
        let mut seed = 12345u64;
        for _ in 0..2000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let a = (seed >> 33) as i64;
            let mut r = req();
            r.want = Qty(1 + a % 50_000_000);
            r.entry = Px(10_000 + (a % 980_000) as i32);
            r.fair_p = ((a % 1000) as f64) / 1000.0;
            r.existing_net = a % 1_000_000_000;
            r.bucket = (a % 16) as usize;
            let v = g.check(&r, Nanos::from_millis(100));
            assert!(v.size() <= r.want, "gate increased size");
        }
    }
}
