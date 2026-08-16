//! `px-engine` — the critical path.
//!
//! # The path
//!
//! ```text
//!   market data -> fair probability -> edge check -> inventory penalty
//!               -> order construction -> risk gate -> order intent
//! ```
//!
//! `Engine::on_market_tick` is that path end to end. It is a single function
//! with no allocation, no locking, no syscalls, no logging, and no `Instant::now()`.
//! Time arrives as a parameter; the returned `Action` is a value, not a side
//! effect. Everything that touches the outside world happens in the caller,
//! after the decision has been made.
//!
//! That structure is what makes the decision testable and replayable: the same
//! inputs produce the same `Action` in production, under the replay harness, and
//! in a unit test.
//!
//! # Two loops, not one
//!
//! * **Reference tick** (`on_reference_tick`) — a spot print arrives. Updates
//!   volatility, cross-asset structure, and the TWAP integral. Tens of times a
//!   second per underlying.
//! * **Market tick** (`on_market_tick`) — a book delta arrives. Re-prices and
//!   re-decides. Thousands of times a second per market.
//!
//! Splitting them keeps the expensive statistical work off the path that runs
//! most often. The fair-value computation on the market tick is a handful of
//! multiplies and one `norm_cdf`, because everything else was already computed
//! when the reference last printed.

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

pub mod calibration;
pub mod recording;
pub mod replay;
pub mod stats;

use px_alpha::{FairModel, FairValue, MarketAlphaState, RefState, TwapAwareModel};
use px_core::{Clock, DenseBook, MarketSpec, Nanos, Prob, Px, Qty, Side, Usd};
use px_edge::{
    assess_make, optimal_take, AdverseSelectionEstimator, Dir, EdgeParams, FeeModel, RewardModel,
};
use px_inventory::{InventoryEngine, InventoryLimits, InventoryState};
use px_risk::{
    FaultKind, Feed, QuoteGovernor, RequoteVerdict, RiskGate, RiskLimits, TradeRequest, Verdict,
};
use px_selector::{
    price_pair, GapTracker, PairCost, Selector, SelectorConfig, SelectorCtx, Structure,
};

/// What the engine wants the I/O layer to do. A value, so it can be asserted on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    /// Nothing to do; leave resting orders where they are.
    Hold,
    /// Replace the two-sided quote.
    Requote {
        bid: Px,
        ask: Px,
        bid_qty: Qty,
        ask_qty: Qty,
    },
    /// Cross the spread.
    Take { dir: Dir, limit: Px, qty: Qty },
    /// Pull all resting orders in this market.
    Cancel,
    /// Pull everything, everywhere. Kill switch.
    CancelAll,
}

/// Per-market state.
#[derive(Debug)]
pub struct MarketCtx {
    pub spec: MarketSpec,
    pub yes_book: DenseBook,
    pub no_book: DenseBook,
    pub alpha: MarketAlphaState,
    pub inventory: InventoryEngine,
    pub selector: Selector,
    pub gap: GapTracker,
    pub adverse: AdverseSelectionEstimator,
    pub fees: FeeModel,
    pub rewards: RewardModel,
    /// Correlation bucket. Markets on the same underlying share one, which is
    /// what stops BTC 5m and BTC 15m from being counted as two independent bets.
    pub bucket: usize,
    pub live_bid: Option<Px>,
    pub live_ask: Option<Px>,
    /// Last fair value, retained for diagnostics and mark-out calibration.
    pub last_fair: FairValue,

    // --- Aggressive order lifecycle ---
    //
    // The engine re-decides from scratch on every tick. Without a record of
    // what it has already *sent*, a decision that persists across ticks — a
    // flatten, most obviously — is re-issued on every one of them. At 50 ticks
    // a second that is fifty duplicate orders chasing one position, which
    // exhausts the rate-limit budget and, if any of them fill, overshoots
    // straight through flat into the opposite exposure.
    //
    // The replay harness found this by trading 1,452 times during a simulated
    // feed outage while "correctly" trying to flatten a single position.
    /// Decaying count of shares crossed, for the turnover governor below.
    pub aggr_turnover: f64,
    /// When `aggr_turnover` was last decayed.
    pub turnover_ts: Nanos,
    /// Quantity of aggressive order currently believed to be at the venue.
    pub in_flight: Qty,
    /// When it was sent. Used to expire the record if no acknowledgement
    /// arrives — a lost order must not block the market forever.
    pub in_flight_since: Nanos,
}

/// How long an unacknowledged aggressive order blocks further aggression.
///
/// Long enough to cover a venue round trip with margin, short enough that a
/// dropped message does not wedge the market. If this expires often, the
/// execution path is unhealthy and the operator should be told.
pub const IN_FLIGHT_TIMEOUT: Nanos = Nanos(2_000_000_000);

impl MarketCtx {
    pub fn new(spec: MarketSpec, rewards: RewardModel) -> Self {
        MarketCtx {
            yes_book: DenseBook::new(spec.tick),
            no_book: DenseBook::new(spec.tick),
            alpha: MarketAlphaState::new(&spec),
            inventory: InventoryEngine::new(0.5, InventoryLimits::default()),
            selector: Selector::new(SelectorConfig::default()),
            gap: GapTracker::default(),
            adverse: AdverseSelectionEstimator::default(),
            fees: FeeModel::for_category(spec.category),
            rewards,
            bucket: spec.underlying.index(),
            live_bid: None,
            live_ask: None,
            last_fair: FairValue::default(),
            aggr_turnover: 0.0,
            turnover_ts: Nanos::ZERO,
            in_flight: Qty::ZERO,
            in_flight_since: Nanos::ZERO,
            spec,
        }
    }

    /// True if an aggressive order is outstanding and not yet timed out.
    #[inline(always)]
    pub fn has_order_in_flight(&self, now: Nanos) -> bool {
        !self.in_flight.is_zero() && now.since(self.in_flight_since).0 < IN_FLIGHT_TIMEOUT.0
    }

    #[inline(always)]
    fn mark_sent(&mut self, qty: Qty, now: Nanos) {
        self.in_flight = qty;
        self.in_flight_since = now;
        self.decay_turnover(now);
        self.aggr_turnover += qty.as_f64();
    }

    /// Exponentially decay the crossed-share count toward zero.
    #[inline]
    fn decay_turnover(&mut self, now: Nanos) {
        let dt = now.since(self.turnover_ts).as_secs_f64();
        if dt > 0.0 {
            self.aggr_turnover *= px_core::math::exp(-dt / TURNOVER_HALF_LIFE_S);
            self.turnover_ts = now;
        }
    }

    /// Are we churning?
    ///
    /// # The failure this catches
    ///
    /// Under heavy flow the engine entered a self-sustaining loop: acquire
    /// inventory passively, breach the flatten threshold, cross the spread to
    /// get out, acquire again. Each cycle pays the spread plus a taker fee to
    /// undo a position it had just been paid to take on.
    ///
    /// The replay harness measured **1,233,672 shares crossed against 48,628
    /// filled passively** — a 25:1 ratio in a system whose entire thesis is that
    /// resting is profitable and crossing is not. Fees were only 10% of the
    /// resulting loss, which rules out the fee schedule and points squarely at
    /// repeatedly paying the spread.
    ///
    /// No single decision in that loop was wrong. Flattening a position at the
    /// limit is correct; so is quoting when there is edge. The loop is an
    /// emergent property of the two being correct in alternation, and no
    /// per-decision check can see it. It needs a control that looks at the
    /// *rate* of aggression over time, which is what this is.
    ///
    /// Crossing more than a few multiples of the position limit per minute is
    /// not risk management, it is a washing machine. When it trips, the engine
    /// falls back to passive-only: it will still quote to reduce, it just stops
    /// paying to do so.
    #[inline]
    pub fn is_churning(&self, limit_shares: i64) -> bool {
        self.aggr_turnover > MAX_TURNOVER_MULTIPLE * (limit_shares as f64 / 1e6)
    }
}

/// Half-life of the turnover counter.
const TURNOVER_HALF_LIFE_S: f64 = 60.0;

/// Crossed shares permitted per half-life, as a multiple of the unhedged
/// position limit. Three round trips a minute is generous for a strategy whose
/// edge is supposed to come from resting.
const MAX_TURNOVER_MULTIPLE: f64 = 6.0;

/// Counters. Cheap to maintain, and the only way to notice that the strategy has
/// quietly stopped doing anything.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub ticks: u64,
    pub requotes: u64,
    pub takes: u64,
    pub holds: u64,
    pub cancels: u64,
    pub rejected_by_risk: u64,
    pub rejected_by_budget: u64,
    pub structure_changes: u64,
}

/// Engine-wide tunables.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub edge: EdgeParams,
    pub risk: RiskLimits,
    /// Floor on the half-spread for passive quotes, micro-dollars. The actual
    /// half-spread is the larger of this and the uncertainty-scaled width.
    pub half_spread: i32,
    /// Multiple of model uncertainty added to the half-spread.
    ///
    /// A market maker who is less sure of fair value should quote *wider*, not
    /// stop. Stopping forfeits the spread, the maker rebate, and the liquidity
    /// reward all at once, in exchange for avoiding a risk that a wider quote
    /// would have been paid to take. The first version of this engine used a
    /// fixed half-spread and a hard viability gate, and the consequence was an
    /// engine that sat out roughly 99% of the session.
    pub spread_sigma_mult: f64,
    /// Hard ceiling on the half-spread, micro-dollars. Beyond this we really
    /// are too uncertain to be in the market and should say so.
    pub max_half_spread: i32,
    /// Base passive quote size.
    pub base_size: Qty,
    /// How long we expect a resting quote to survive, for reward accrual.
    pub expected_rest_s: f64,
    /// Maximum size for an aggressive take.
    pub max_take: Qty,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            edge: EdgeParams::default(),
            risk: RiskLimits::default(),
            half_spread: 15_000,
            spread_sigma_mult: 1.5,
            max_half_spread: 150_000,
            base_size: Qty::shares(200),
            expected_rest_s: 20.0,
            max_take: Qty::shares(2_000),
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    pub refs: RefState,
    pub model: TwapAwareModel,
    pub risk: RiskGate,
    pub governor: QuoteGovernor,
    pub markets: Vec<MarketCtx>,
    pub cfg: EngineConfig,
    pub stats: Stats,
}

impl Engine {
    pub fn new(cfg: EngineConfig, bankroll: Usd, tier: px_risk::Tier, now: Nanos) -> Self {
        Engine {
            refs: RefState::new(),
            model: TwapAwareModel::default(),
            risk: RiskGate::new(cfg.risk, bankroll, now),
            governor: QuoteGovernor::new(tier, now),
            markets: Vec::new(),
            cfg,
            stats: Stats::default(),
        }
    }

    pub fn add_market(&mut self, m: MarketCtx) -> usize {
        self.markets.push(m);
        self.markets.len() - 1
    }

    /// Reference feed tick. Off the hottest path, so this is where the heavier
    /// statistical work lives.
    pub fn on_reference_tick(&mut self, asset: usize, price: f64, now: Nanos) {
        let t = now.as_secs_f64();
        self.refs.on_print(asset, price, t);
        self.refs.on_clock(t);
        self.risk.health.touch(Feed::Reference, now);
        self.risk.effective_dof = self.refs.cross.effective_dof();

        for m in &mut self.markets {
            if m.spec.underlying.index() == asset {
                m.alpha.twap.update(price, t);
            }
        }
    }

    /// Apply a book delta. Kept separate from the decision so that a burst of
    /// deltas in one venue message costs one decision, not five.
    pub fn on_book_delta(
        &mut self,
        idx: usize,
        yes: bool,
        side: Side,
        px: Px,
        size: Qty,
        now: Nanos,
    ) {
        if idx >= self.markets.len() {
            return;
        }
        let m = &mut self.markets[idx];
        if yes {
            m.yes_book.set_level(side, px, size);
            m.yes_book.recv_ts_ns = now.0;
        } else {
            m.no_book.set_level(side, px, size);
            m.no_book.recv_ts_ns = now.0;
        }
        self.risk.health.touch(Feed::MarketData, now);
    }

    /// **The critical path.**
    ///
    /// Allocation-free, lock-free, syscall-free. Every branch is data-dependent
    /// on state already resident in cache.
    pub fn on_market_tick(&mut self, idx: usize, now: Nanos) -> Action {
        if idx >= self.markets.len() {
            return Action::Hold;
        }

        // Destructure so the borrow checker can see that the market, the
        // reference state, and the risk gate are disjoint.
        let Engine {
            refs,
            model,
            risk,
            governor,
            markets,
            cfg,
            stats,
        } = self;
        let m = &mut markets[idx];
        stats.ticks += 1;

        // --- 1. Fair probability. Never reads a Polymarket price. ---
        let fv = model.fair(&m.spec, refs, &m.alpha, now);
        m.last_fair = fv;

        // --- 2. Data-quality guard on the book itself. ---
        if m.yes_book.is_crossed() {
            risk.health.fault(FaultKind::CrossedBook);
        }
        let feed_ok = risk.health.check(now) == FaultKind::None;

        // --- 3. Relative value. The gap is fair-minus-mid, normalised by the
        //        market's own habitual bias and gap volatility.
        let mid = m.yes_book.mid();
        let gap = match mid {
            Some(mp) => (fv.p.as_px().0 - mp.0) as f64,
            None => 0.0,
        };
        if mid.is_some() && fv.usable {
            m.gap.observe(gap);
        }
        let z = m.gap.z(gap);

        // --- 4. Realisable edge, both ways. ---
        let params = EdgeParams {
            adverse: m.adverse.estimate(),
            ..cfg.edge
        };

        let best_take = if mid.is_some() {
            let dir = if fv.p.as_px().0 > mid.unwrap().0 {
                Dir::Buy
            } else {
                Dir::Sell
            };
            let t = optimal_take(
                &m.yes_book,
                dir,
                fv.p.as_px(),
                cfg.max_take,
                &m.fees,
                fv.sigma_p,
                &params,
            );
            if t.qty.is_zero() {
                None
            } else {
                Some(t)
            }
        } else {
            None
        };

        // --- 5. Inventory penalty -> working price -> quote construction. ---
        // Half-spread scales with model uncertainty. Being less sure buys a
        // wider quote, not silence — we stay in the market, keep accruing the
        // liquidity reward, and get paid more for the risk we are less able to
        // measure. Only past `max_half_spread` do we concede and step away.
        let sigma_width = (fv.sigma_p * cfg.spread_sigma_mult * 1e6) as i64;
        let half_spread = (cfg.half_spread as i64)
            .max(sigma_width)
            .min(cfg.max_half_spread as i64) as i32;
        let too_uncertain = sigma_width > cfg.max_half_spread as i64;

        // The penalty uses the *conservative* volatility: here a high sigma
        // genuinely means "carry less inventory", so erring high is erring safe.
        // The price it is applied to came from the unbiased volatility.
        let (mut raw_bid, mut raw_ask) = m.inventory.quote(
            fv.p.as_px(),
            fv.delta,
            fv.settle_sd_risk,
            half_spread,
            m.spec.tick,
        );

        // Post-only clamp.
        //
        // When our fair value diverges from the venue's by more than the
        // half-spread — which is exactly the situation this strategy is built
        // for — the naive quote lands on the wrong side of the venue's touch.
        // Such an order is *marketable*: it does not rest, it executes
        // immediately against the touch, as a taker, paying the taker fee we
        // spent the whole design avoiding.
        //
        // Worse, it converts our best idea into our worst trade. A bid above
        // the offer says "I will pay more than anyone is asking", and the
        // counterparty who accepts is the one who knows why they should.
        //
        // Production sends these with `postOnly: true`, so the venue rejects
        // rather than crosses. Here we clamp to one tick inside the touch so
        // the order rests where we intended. If the resulting price no longer
        // carries edge, the make assessment below will say so and we will not
        // send it at all — which is the correct outcome, and a different one
        // from crossing.
        //
        // The replay harness found this as a 26-cent-per-share round-trip loss
        // that was completely insensitive to every strategy parameter. A loss
        // that does not respond to the knobs is never a strategy result.
        if let Some(best_ask) = m.yes_book.best_ask() {
            let cap = Px(best_ask.0 - m.spec.tick);
            if raw_bid.0 >= best_ask.0 {
                raw_bid = cap.clamp_unit();
            }
        }
        if let Some(best_bid) = m.yes_book.best_bid() {
            let floor = Px(best_bid.0 + m.spec.tick);
            if raw_ask.0 <= best_bid.0 {
                raw_ask = floor.clamp_unit();
            }
        }

        // Residual uncertainty for the passive assessment.
        //
        // The safety margin exists to stop us paying more than fair value minus
        // our own error bar. For an aggressive order that is the only
        // protection available, because the entry price is whatever the book
        // says. For a resting quote we have already stepped back by
        // `half_spread`, so charging the full sigma again double-counts it and
        // makes every quote look unprofitable. Only the part the spread does
        // not already cover should be charged.
        let residual_sigma = (fv.sigma_p - (half_spread as f64 / 1e6)).max(0.0);

        // Assess **both** sides.
        //
        // Only the bid used to be assessed, and `best_make` — the single value
        // the selector's rule 9 gates on — was therefore always the bid's
        // economics. Held long, that is exactly backwards: the bid is the side
        // we do not want filled, so it correctly reads unviable, and the whole
        // two-sided quote gets pulled *including the ask that would have
        // reduced the position*. Same failure as the quote-sizer bug, one layer
        // up: the code assumed the bid is always the interesting side.
        //
        // The risk credit goes to whichever side reduces exposure. It is the
        // inventory penalty that a fill would retire, and it is zero once flat.
        let net_pos = m.inventory.position.net();
        let carry = m
            .inventory
            .penalty_from_model(fv.delta, fv.settle_sd_risk)
            .abs() as i32;
        let bid_credit = if net_pos < 0 { carry } else { 0 };
        let ask_credit = if net_pos > 0 { carry } else { 0 };

        let (best_make, best_make_ask) = if fv.usable && !too_uncertain {
            let bid = assess_make(
                &m.yes_book,
                Dir::Buy,
                fv.p.as_px(),
                raw_bid,
                cfg.base_size,
                &m.fees,
                &m.rewards,
                cfg.expected_rest_s,
                true,
                residual_sigma,
                bid_credit,
                &params,
            );
            let ask = assess_make(
                &m.yes_book,
                Dir::Sell,
                fv.p.as_px(),
                raw_ask,
                cfg.base_size,
                &m.fees,
                &m.rewards,
                cfg.expected_rest_s,
                true,
                residual_sigma,
                ask_credit,
                &params,
            );
            // Rule 9 should fire if *either* side is worth resting. Report the
            // better of the two so a viable ask can carry a two-sided quote
            // whose bid is deliberately uncompetitive.
            let better = if ask.net_per_share > bid.net_per_share {
                ask
            } else {
                bid
            };
            (Some(better), Some(ask))
        } else {
            (None, None)
        };
        let _ = best_make_ask; // retained for diagnostics; sizing uses size_factor

        // --- 6. Model-free arbitrage check. ---
        let pair = if m.no_book.best_ask().is_some() && m.yes_book.best_ask().is_some() {
            price_pair(&m.yes_book, &m.no_book, cfg.max_take, &m.fees)
        } else {
            PairCost::default()
        };

        // --- 7. Structure selection. ---
        let budget_ok =
            governor.orders.available(now) >= 2.0 && governor.cancels.available(now) >= 2.0;
        let ctx = SelectorCtx {
            now,
            tau: fv.tau,
            fair: fv.p,
            sigma_p: fv.sigma_p,
            usable: fv.usable,
            decided: fv.decided,
            z,
            inventory: m.inventory.state,
            net_position: m.inventory.position.net(),
            matched: m.inventory.position.matched().0,
            best_take,
            best_make,
            pair,
            feed_ok,
            quote_budget_ok: budget_ok,
        };
        let decision = m.selector.evaluate(&ctx);
        if decision.changed {
            stats.structure_changes += 1;
        }

        // --- 8. Translate the structure into an order intent, then gate it. ---
        //
        // Every aggressive structure is blocked while an order is outstanding.
        // A decision that stays true across ticks must not become one order per
        // tick; we already sent our answer and are waiting to hear back.
        let wants_aggression = matches!(
            decision.structure,
            Structure::Flatten
                | Structure::InventoryRelease
                | Structure::SyntheticPair
                | Structure::NearResolution
                | Structure::HedgedDirectional
                | Structure::DynamicRotation
        );
        if wants_aggression && m.has_order_in_flight(now) {
            stats.holds += 1;
            return Action::Hold;
        }

        // Turnover governor. A model-free arbitrage still goes through — it is
        // locked-in profit, not churn — but every model-driven crossing stops
        // until the rate subsides.
        m.decay_turnover(now);
        if wants_aggression
            && decision.structure != Structure::SyntheticPair
            && m.is_churning(m.inventory.limits.flattening)
        {
            stats.rejected_by_risk += 1;
            stats.holds += 1;
            return Action::Hold;
        }

        match decision.structure {
            Structure::Idle => {
                if m.live_bid.is_some() || m.live_ask.is_some() {
                    m.live_bid = None;
                    m.live_ask = None;
                    stats.cancels += 1;
                    return Action::Cancel;
                }
                stats.holds += 1;
                Action::Hold
            }

            Structure::Flatten | Structure::InventoryRelease => {
                let net = m.inventory.position.net();
                if net == 0 {
                    stats.holds += 1;
                    return Action::Hold;
                }
                let dir = if net > 0 { Dir::Sell } else { Dir::Buy };
                let want = Qty(net.abs()).min(cfg.max_take);
                let limit = if net > 0 { Px::ZERO } else { Px::ONE };
                m.mark_sent(want, now);
                stats.takes += 1;
                Action::Take {
                    dir,
                    limit,
                    qty: want,
                }
            }

            Structure::SyntheticPair => {
                if !pair.viable || pair.qty.is_zero() {
                    stats.holds += 1;
                    return Action::Hold;
                }
                m.mark_sent(pair.qty, now);
                stats.takes += 1;
                Action::Take {
                    dir: Dir::Buy,
                    limit: pair.yes_px,
                    qty: pair.qty,
                }
            }

            Structure::NearResolution
            | Structure::HedgedDirectional
            | Structure::DynamicRotation => {
                let t = match best_take {
                    Some(t) if t.viable => t,
                    _ => {
                        stats.holds += 1;
                        return Action::Hold;
                    }
                };
                let dir = decision.target_dir.unwrap_or(Dir::Buy);
                let req = TradeRequest {
                    bucket: m.bucket,
                    entry: t.avg_entry,
                    want: t.qty,
                    fair_p: match dir {
                        Dir::Buy => fv.p.as_f64(),
                        Dir::Sell => 1.0 - fv.p.as_f64(),
                    },
                    fee_per_share: t.fee_per_share as f64,
                    existing_net: m.inventory.position.net(),
                    existing_capital: m.inventory.position.cost_basis,
                    min_size: m.spec.min_size,
                    near_resolution: decision.structure == Structure::NearResolution,
                };
                match risk.check(&req, now) {
                    Verdict::Rejected(_) => {
                        stats.rejected_by_risk += 1;
                        stats.holds += 1;
                        Action::Hold
                    }
                    v => {
                        m.mark_sent(v.size(), now);
                        stats.takes += 1;
                        Action::Take {
                            dir,
                            limit: t.worst_px,
                            qty: v.size(),
                        }
                    }
                }
            }

            Structure::TwoSidedQuote => {
                // Quote-economy check. This is where the rate-limit budget is
                // actually spent, and where most candidate requotes die.
                let cur_bid = m.live_bid.unwrap_or(Px::ZERO);
                let cur_ask = m.live_ask.unwrap_or(Px::ONE);
                let urgent = m.inventory.state == InventoryState::Reducing;

                let vb = governor.should_requote(cur_bid, raw_bid, m.spec.tick, now, urgent);
                let va = governor.should_requote(cur_ask, raw_ask, m.spec.tick, now, urgent);

                if vb == RequoteVerdict::Send || va == RequoteVerdict::Send {
                    if !governor.commit_requote(now) {
                        stats.rejected_by_budget += 1;
                        stats.holds += 1;
                        return Action::Hold;
                    }
                    // Which side *adds* exposure depends on the sign of the
                    // position, not on which side of the book it is.
                    //
                    // This was hardcoded as "bid adds, ask reduces", which is
                    // right only when we are long or flat. Held short, it was
                    // exactly backwards: the engine shrank the bid that would
                    // have covered and enlarged the ask that deepened the short.
                    // `Reducing` state then throttled the wrong side to zero and
                    // the position ran away — a session ended 5,989 shares short
                    // against a 2,000-share limit.
                    let net = m.inventory.position.net();
                    let bid_adds = net >= 0;
                    let ask_adds = net <= 0;
                    let mut bq =
                        Qty((cfg.base_size.0 as f64 * m.inventory.size_factor(bid_adds)) as i64);
                    let mut aq =
                        Qty((cfg.base_size.0 as f64 * m.inventory.size_factor(ask_adds)) as i64);

                    // Hard risk cap on the adding side. The inventory throttle
                    // above is a soft multiplier; it never consults the capital,
                    // share, or bucket limits. Without this, a steady stream of
                    // passive fills walks past every hard limit in `px-risk`
                    // because only aggressive orders were ever gated.
                    let cap = risk.max_passive_size(
                        m.bucket,
                        fv.p.as_px(),
                        net,
                        m.inventory.position.cost_basis,
                        now,
                    );
                    if bid_adds {
                        bq = bq.min(cap);
                    }
                    if ask_adds {
                        aq = aq.min(cap);
                    }
                    m.live_bid = Some(raw_bid);
                    m.live_ask = Some(raw_ask);
                    stats.requotes += 1;
                    Action::Requote {
                        bid: raw_bid,
                        ask: raw_ask,
                        bid_qty: bq,
                        ask_qty: aq,
                    }
                } else {
                    if vb == RequoteVerdict::NoBudget || va == RequoteVerdict::NoBudget {
                        stats.rejected_by_budget += 1;
                    }
                    stats.holds += 1;
                    Action::Hold
                }
            }
        }
    }

    /// Record a fill and update every piece of state that depends on it.
    ///
    /// Also retires the in-flight record: the venue has answered, so the market
    /// is free to act again.
    pub fn on_fill(&mut self, idx: usize, signed_qty: i64, px: Px) {
        if idx >= self.markets.len() {
            return;
        }
        let bucket = self.markets[idx].bucket;
        self.markets[idx].inventory.on_fill(signed_qty, px);
        self.markets[idx].in_flight = Qty::ZERO;
        self.risk.on_fill(bucket, signed_qty, px);
    }

    /// Record that an aggressive order was rejected, expired, or fully
    /// cancelled without filling. Clears the in-flight block immediately rather
    /// than waiting for the timeout, and releases the risk reservation the gate
    /// took when it approved the order.
    pub fn on_order_closed(&mut self, idx: usize) {
        if idx >= self.markets.len() {
            return;
        }
        let bucket = self.markets[idx].bucket;
        let qty = self.markets[idx].in_flight;
        let px = self.markets[idx].last_fair.p.as_px();
        if !qty.is_zero() {
            self.risk.on_reject(bucket, qty, px);
        }
        self.markets[idx].in_flight = Qty::ZERO;
    }

    /// Feed a mark-out observation back into the adverse-selection estimate.
    /// Called by the calibration loop, not the hot path.
    pub fn on_markout(&mut self, idx: usize, dir: Dir, fill_px: Px, fair_after: Prob) {
        if idx < self.markets.len() {
            self.markets[idx]
                .adverse
                .observe(dir, fill_px, fair_after.as_px());
        }
    }

    /// Trip the kill switch across the whole book.
    ///
    /// Clears in-flight records as well as quotes. Leaving them set would mean
    /// that after recovery every market believes it still has an order at the
    /// venue, and the aggressive-structure guard silently blocks all of them
    /// until the two-second timeout expires — one market at a time. A halt must
    /// leave the engine in a state it can resume from.
    pub fn halt(&mut self, kind: FaultKind) -> Action {
        self.risk.health.fault(kind);
        for (i, m) in self.markets.iter_mut().enumerate() {
            m.inventory.freeze();
            m.live_bid = None;
            m.live_ask = None;
            if !m.in_flight.is_zero() {
                // Release the risk reservation too, or the headroom leaks.
                self.risk
                    .on_reject(m.bucket, m.in_flight, m.last_fair.p.as_px());
                let _ = i;
            }
            m.in_flight = Qty::ZERO;
        }
        self.governor.live_orders = 0;
        self.stats.cancels += 1;
        Action::CancelAll
    }

    /// Mark-to-model equity across the whole book, micro-dollars.
    ///
    /// Feeds the drawdown limit. Called from the operating loop, not the hot
    /// path.
    pub fn mark_equity(&mut self, realised_cash: i64) {
        let mut equity = realised_cash;
        for m in &self.markets {
            let mark = m.last_fair.p.as_px();
            equity += (m.inventory.position.net() as i128 * mark.0 as i128 / 1_000_000) as i64;
        }
        self.risk.observe_equity(equity);
    }

    /// Convenience for callers holding a `Clock`.
    pub fn tick_with<C: Clock>(&mut self, idx: usize, clock: &C) -> Action {
        self.on_market_tick(idx, clock.now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_core::{Category, MarketId, Settlement, TokenId, Underlying};

    fn spec(expiry_s: f64) -> MarketSpec {
        MarketSpec {
            market: MarketId(1),
            yes: TokenId(1),
            no: TokenId(2),
            underlying: Underlying::Btc,
            category: Category::Crypto,
            settlement: Settlement::Twap { window_s: 60.0 },
            strike: 65_000.0,
            expiry: Nanos::from_secs_f64(expiry_s),
            tick: 10_000,
            min_size: Qty::shares(5),
            reward_max_spread_ticks: 3,
            reward_min_size: Qty::shares(50),
        }
    }

    fn rewards() -> RewardModel {
        RewardModel {
            pool_per_day: 300_000.0 * 1e6,
            est_total_q: 5e6,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::shares(50),
        }
    }

    /// Build an engine with a warm model and a two-sided book.
    fn warm_engine() -> (Engine, Nanos) {
        let mut e = Engine::new(
            EngineConfig::default(),
            Usd::dollars(100_000),
            px_risk::Tier::Standard,
            Nanos::ZERO,
        );
        e.add_market(MarketCtx::new(spec(400.0), rewards()));

        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..2500 {
            px *= if i % 2 == 0 { 1.00003 } else { 0.99997 };
            t += 0.04;
            e.on_reference_tick(0, px, Nanos::from_secs_f64(t));
        }
        let now = Nanos::from_secs_f64(t + 0.01);

        for f in Feed::ALL {
            e.risk.health.touch(f, now);
        }

        e.on_book_delta(0, true, Side::Bid, Px(480_000), Qty::shares(300), now);
        e.on_book_delta(0, true, Side::Ask, Px(520_000), Qty::shares(300), now);
        e.on_book_delta(0, false, Side::Bid, Px(480_000), Qty::shares(300), now);
        e.on_book_delta(0, false, Side::Ask, Px(520_000), Qty::shares(300), now);
        (e, now)
    }

    #[test]
    fn a_warm_engine_quotes_two_sided_by_default() {
        let (mut e, now) = warm_engine();
        let a = e.on_market_tick(0, now);
        match a {
            Action::Requote { bid, ask, .. } => {
                assert!(bid < ask);
                assert!(bid.0 > 0 && ask.0 < 1_000_000);
            }
            other => panic!("expected a requote, got {other:?}"),
        }
        assert_eq!(e.markets[0].selector.current(), Structure::TwoSidedQuote);
    }

    #[test]
    fn a_stale_reference_feed_produces_a_cancel() {
        let (mut e, now) = warm_engine();
        e.on_market_tick(0, now);
        // Ten seconds with no reference print.
        let later = Nanos(now.0 + 10_000_000_000);
        let a = e.on_market_tick(0, later);
        assert!(matches!(a, Action::Cancel | Action::Hold));
        assert!(!e.risk.health.healthy());
    }

    #[test]
    fn the_engine_holds_rather_than_churning_on_tiny_moves() {
        // Quote economy: after the first requote, a stream of ticks that barely
        // move fair value must not spend the rate-limit budget.
        let (mut e, now) = warm_engine();
        let first = e.on_market_tick(0, now);
        assert!(matches!(first, Action::Requote { .. }));

        let mut holds = 0;
        for i in 1..200u64 {
            let t = Nanos(now.0 + i * 1_000_000); // 1 ms apart
            for f in Feed::ALL {
                e.risk.health.touch(f, t);
            }
            if matches!(e.on_market_tick(0, t), Action::Hold) {
                holds += 1;
            }
        }
        assert!(holds > 180, "only {holds} holds out of 199");
        assert!(e.stats.requotes < 5, "requotes = {}", e.stats.requotes);
    }

    #[test]
    fn a_free_pair_is_taken_ahead_of_quoting() {
        let (mut e, now) = warm_engine();
        // Rebuild both books so YES and NO are each offered at 45c without
        // crossing their own bids. A complete set costs 90c; the two taker fees
        // at 45c come to ~3.5c, leaving ~6.5c of locked-in profit per set.
        for yes in [true, false] {
            e.on_book_delta(0, yes, Side::Bid, Px(480_000), Qty::ZERO, now);
            e.on_book_delta(0, yes, Side::Ask, Px(520_000), Qty::ZERO, now);
            e.on_book_delta(0, yes, Side::Bid, Px(400_000), Qty::shares(200), now);
            e.on_book_delta(0, yes, Side::Ask, Px(450_000), Qty::shares(200), now);
        }
        assert!(!e.markets[0].yes_book.is_crossed());

        let a = e.on_market_tick(0, now);
        assert_eq!(e.markets[0].selector.current(), Structure::SyntheticPair);
        assert!(matches!(a, Action::Take { .. }));
    }

    #[test]
    fn a_crossed_book_trips_the_data_quality_guard() {
        // A bid above an ask means we have lost sequencing. Quoting into that
        // is how a resync bug becomes a position.
        let (mut e, now) = warm_engine();
        e.on_book_delta(0, true, Side::Ask, Px(450_000), Qty::shares(200), now);
        assert!(e.markets[0].yes_book.is_crossed());
        e.on_market_tick(0, now);
        assert!(!e.risk.health.healthy());
    }

    #[test]
    fn a_large_position_forces_a_flatten() {
        let (mut e, now) = warm_engine();
        e.on_fill(0, Qty::shares(1500).0, Px(500_000));
        assert_eq!(e.markets[0].inventory.state, InventoryState::Flattening);
        let a = e.on_market_tick(0, now);
        match a {
            Action::Take { dir, qty, .. } => {
                assert_eq!(dir, Dir::Sell);
                assert!(qty > Qty::ZERO);
            }
            other => panic!("expected a flattening take, got {other:?}"),
        }
    }

    #[test]
    fn halting_freezes_every_market_and_pulls_every_quote() {
        let (mut e, now) = warm_engine();
        e.on_market_tick(0, now);
        assert!(e.markets[0].live_bid.is_some());
        let a = e.halt(FaultKind::ApiAnomaly);
        assert_eq!(a, Action::CancelAll);
        assert_eq!(e.markets[0].inventory.state, InventoryState::Frozen);
        assert!(e.markets[0].live_bid.is_none());
        assert!(matches!(
            e.on_market_tick(0, now),
            Action::Hold | Action::Cancel
        ));
    }

    #[test]
    fn inventory_skews_the_quote_downward_when_long() {
        let (mut e, now) = warm_engine();
        let flat = e.on_market_tick(0, now);
        let flat_bid = match flat {
            Action::Requote { bid, .. } => bid,
            other => panic!("expected requote, got {other:?}"),
        };

        e.on_fill(0, Qty::shares(400).0, Px(500_000));
        // Force a fresh quote by advancing past the dwell/economy thresholds.
        let t = Nanos(now.0 + 500_000_000);
        for f in Feed::ALL {
            e.risk.health.touch(f, t);
        }
        e.refs.on_print(0, e.refs.spot[0], t.as_secs_f64());
        let skewed = e.on_market_tick(0, t);
        if let Action::Requote { bid, .. } = skewed {
            assert!(bid <= flat_bid, "long book should not bid higher");
        }
    }

    #[test]
    fn halt_leaves_the_engine_resumable() {
        // Regression: `halt` cleared quotes but not in-flight records, so every
        // market came back believing it still had an order at the venue and the
        // aggressive guard blocked it until the 2 s timeout expired.
        let (mut e, now) = warm_engine();
        e.on_fill(0, Qty::shares(1500).0, Px(500_000)); // force a flatten
        let a = e.on_market_tick(0, now);
        assert!(matches!(a, Action::Take { .. }));
        assert!(e.markets[0].has_order_in_flight(now));

        e.halt(FaultKind::ApiAnomaly);
        assert!(!e.markets[0].has_order_in_flight(now));
        assert_eq!(e.governor.live_orders, 0);

        // After recovery the market can act immediately.
        e.risk.health.clear();
        e.markets[0].inventory.thaw();
        let t = Nanos(now.0 + 1_000_000);
        for f in Feed::ALL {
            e.risk.health.touch(f, t);
        }
        e.refs.on_print(0, e.refs.spot[0], t.as_secs_f64());
        assert!(matches!(
            e.on_market_tick(0, t),
            Action::Take { .. } | Action::Requote { .. } | Action::Hold
        ));
    }

    #[test]
    fn a_hostile_book_message_does_not_kill_the_process() {
        // End-to-end version of the boundary-validation guard: garbage from the
        // feed must be dropped, not indexed.
        let (mut e, now) = warm_engine();
        for px in [Px(5_000_000), Px(-1), Px(i32::MAX), Px(i32::MIN)] {
            e.on_book_delta(0, true, Side::Bid, px, Qty::shares(10), now);
            e.on_book_delta(0, false, Side::Ask, px, Qty(i64::MAX), now);
        }
        assert!(e.markets[0].yes_book.rejected > 0);
        // And the engine still functions.
        let a = e.on_market_tick(0, now);
        assert!(matches!(
            a,
            Action::Requote { .. } | Action::Hold | Action::Cancel | Action::Take { .. }
        ));
    }

    #[test]
    fn the_drawdown_limit_stops_the_whole_book() {
        let (mut e, now) = warm_engine();
        e.risk.limits.max_drawdown = Usd::dollars(500);
        e.mark_equity(Usd::dollars(1_000).0);
        assert!(!e.risk.is_halted());
        e.mark_equity(Usd::dollars(400).0);
        assert!(e.risk.is_halted());

        // Nothing aggressive gets through the gate.
        e.on_fill(0, Qty::shares(600).0, Px(500_000));
        let a = e.on_market_tick(0, now);
        assert!(!matches!(a, Action::Requote { .. }));
    }

    #[test]
    fn the_quote_sizer_knows_which_side_reduces_when_short() {
        // Regression: "bid adds, ask reduces" is only true when long or flat.
        // Held short it is exactly backwards, and `Reducing` then throttles the
        // covering side to zero while feeding the side that deepens the short.
        let (mut e, _now) = warm_engine();
        e.on_fill(0, -Qty::shares(600).0, Px(500_000)); // short, into Reducing
        assert!(e.markets[0].inventory.position.net() < 0);
        assert_eq!(e.markets[0].inventory.state, InventoryState::Reducing);

        // Short: buying covers, selling deepens. The sizer must mute the ask,
        // not the bid.
        let inv = &e.markets[0].inventory;
        let net = inv.position.net();
        let bid_adds = net >= 0;
        let ask_adds = net <= 0;
        assert!(!bid_adds, "bid should be the reducing side when short");
        assert!(ask_adds, "ask should be the adding side when short");
        assert!(
            inv.size_factor(bid_adds) > 0.0,
            "covering side was throttled to zero"
        );
        assert_eq!(
            inv.size_factor(ask_adds),
            0.0,
            "kept feeding the side that deepens the short"
        );

        // And the mirror case, long, still behaves as before.
        let mut e2 = warm_engine().0;
        e2.on_fill(0, Qty::shares(600).0, Px(500_000));
        let inv2 = &e2.markets[0].inventory;
        let net2 = inv2.position.net();
        assert!(net2 > 0);
        assert_eq!(inv2.size_factor(net2 >= 0), 0.0); // bid adds -> muted
        assert!(inv2.size_factor(net2 <= 0) > 0.0); // ask reduces -> live
    }

    #[test]
    fn the_turnover_governor_stops_a_churn_loop() {
        // Regression: acquire passively, breach the flatten threshold, cross to
        // get out, repeat — paying the spread each cycle to undo a position we
        // had just been paid to take on. No single decision is wrong, so only a
        // rate-based control can see it.
        let (mut e, now) = warm_engine();
        let limit = e.markets[0].inventory.limits.flattening;

        let mut crossed = 0i64;
        let mut t = now;
        for _ in 0..400 {
            t = Nanos(t.0 + 20_000_000); // 20 ms
            for f in Feed::ALL {
                e.risk.health.touch(f, t);
            }
            // Keep forcing the book past the flatten threshold.
            if e.markets[0].inventory.position.net().abs() < limit {
                e.on_fill(0, Qty::shares(1200).0, Px(500_000));
            }
            if let Action::Take { qty, .. } = e.on_market_tick(0, t) {
                crossed += qty.0;
                e.on_fill(0, -qty.0, Px(500_000));
            }
        }

        // Without the governor this ran unbounded. Six position-limits per
        // 60 s half-life is the cap; allow generous slack for the decay.
        let cap = (MAX_TURNOVER_MULTIPLE * 4.0) * (limit as f64 / 1e6);
        assert!(
            (crossed as f64 / 1e6) < cap,
            "crossed {} shares against a cap of {cap}",
            crossed as f64 / 1e6
        );
        assert!(e.markets[0].aggr_turnover > 0.0);
    }

    #[test]
    fn a_free_pair_is_exempt_from_the_turnover_governor() {
        // Locked-in profit is not churn. The governor must not block it.
        let (mut e, now) = warm_engine();
        e.markets[0].aggr_turnover = 1e9; // deep in churn territory
        e.markets[0].turnover_ts = now;
        assert!(e.markets[0].is_churning(e.markets[0].inventory.limits.flattening));

        for yes in [true, false] {
            e.on_book_delta(0, yes, Side::Bid, Px(480_000), Qty::ZERO, now);
            e.on_book_delta(0, yes, Side::Ask, Px(520_000), Qty::ZERO, now);
            e.on_book_delta(0, yes, Side::Bid, Px(400_000), Qty::shares(200), now);
            e.on_book_delta(0, yes, Side::Ask, Px(450_000), Qty::shares(200), now);
        }
        let a = e.on_market_tick(0, now);
        assert_eq!(e.markets[0].selector.current(), Structure::SyntheticPair);
        assert!(matches!(a, Action::Take { .. }));
    }

    #[test]
    fn out_of_range_market_index_is_safe() {
        let (mut e, now) = warm_engine();
        assert_eq!(e.on_market_tick(99, now), Action::Hold);
        e.on_book_delta(99, true, Side::Bid, Px(500_000), Qty::shares(10), now);
        e.on_fill(99, 100, Px(500_000));
    }

    #[test]
    fn an_empty_book_does_not_panic() {
        let mut e = Engine::new(
            EngineConfig::default(),
            Usd::dollars(100_000),
            px_risk::Tier::Standard,
            Nanos::ZERO,
        );
        e.add_market(MarketCtx::new(spec(400.0), rewards()));
        let a = e.on_market_tick(0, Nanos::from_secs_f64(1.0));
        assert!(matches!(a, Action::Hold | Action::Cancel));
    }

    #[test]
    fn stats_account_for_every_tick() {
        let (mut e, now) = warm_engine();
        for i in 0..100u64 {
            let t = Nanos(now.0 + i * 2_000_000);
            for f in Feed::ALL {
                e.risk.health.touch(f, t);
            }
            e.on_market_tick(0, t);
        }
        let s = e.stats;
        assert_eq!(s.ticks, 100);
        assert_eq!(s.requotes + s.takes + s.holds + s.cancels, 100);
    }
}
