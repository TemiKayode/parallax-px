//! Deterministic replay and simulation harness.
//!
//! # What a good harness has to get right
//!
//! Backtests of market-making strategies are famously easy to fool yourself
//! with. Three failure modes account for most of it, and each is addressed
//! explicitly here:
//!
//! 1. **Assuming your quotes fill without moving anything.** They do not. The
//!    simulator only fills a resting quote when simulated counterparty flow
//!    actually reaches its price, and it models queue position: if 300 shares
//!    rest ahead of us, the first 300 shares of flow are not ours.
//!
//! 2. **Assuming your quotes fill when you were right.** That is exactly
//!    backwards. Passive fills are adversely selected by construction — you get
//!    bought from when the price is about to fall. The simulator drives
//!    counterparty flow from the *true* price path, so a quote left stale
//!    through a move gets run over, and the resulting mark-out shows up in the
//!    report as it would in production.
//!
//! 3. **Assuming the venue knows what you know.** The whole strategy rests on
//!    the venue's book lagging the reference feed. `SimConfig::venue_lag_s`
//!    parameterises exactly that, which turns the untestable question "do we
//!    have an edge" into the testable one: **how much venue lag do we need
//!    before this is profitable?** That number is the harness's most useful
//!    output, and it is the number to re-measure whenever the venue changes.
//!
//! Everything is deterministic. Same seed, same result, on any machine — which
//! is why `px_core::math` implements its own transcendentals.

use crate::{Action, Engine, EngineConfig, MarketCtx, Stats};
use px_core::{
    math::{exp, ln},
    Category, MarketId, MarketSpec, Nanos, Px, Qty, Settlement, Side, TokenId, Underlying, Usd,
};
use px_edge::{Dir, RewardModel};
use px_risk::Feed;

/// Mark-out horizon, in seconds. Both the online calibration loop and the
/// end-of-run report use this, so they agree by construction.
pub const MARKOUT_S: f64 = 5.0;

/// Deterministic PRNG. A 64-bit LCG with a Box–Muller normal, chosen because it
/// is trivially portable and reproducible; nothing here needs cryptographic or
/// equidistribution quality.
#[derive(Clone, Copy, Debug)]
pub struct Rng(pub u64);

impl Rng {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    #[inline]
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * ln(u1)).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// A scheduled disturbance in the price path.
#[derive(Clone, Copy, Debug)]
pub struct Shock {
    /// When it lands, seconds from the start of the scenario.
    pub at_s: f64,
    /// Instantaneous relative move, e.g. 0.004 for +40 bp.
    pub magnitude: f64,
}

/// A window during which a feed stops updating.
#[derive(Clone, Copy, Debug)]
pub struct Outage {
    pub feed_index: usize,
    pub start_s: f64,
    pub end_s: f64,
}

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub seed: u64,
    pub duration_s: f64,
    /// Interval between reference prints.
    pub ref_dt_s: f64,
    /// Interval between engine decision ticks.
    pub tick_dt_s: f64,
    pub start_spot: f64,
    pub strike: f64,
    /// Annualised volatility of the reference asset.
    pub annual_vol: f64,
    /// Seconds by which the venue's book trails the reference feed. **The**
    /// parameter: sweep it to find the break-even.
    pub venue_lag_s: f64,
    /// Random noise the venue adds around its (lagged) view, in probability
    /// units.
    pub venue_noise: f64,
    /// Half-spread the venue's book quotes, in micro-dollars.
    pub venue_half_spread: i32,
    /// Resting size at each venue level.
    pub venue_depth: Qty,
    /// Expected counterparty market orders per second.
    pub flow_per_s: f64,
    /// Mean counterparty order size in shares.
    pub flow_size: f64,
    /// Fraction of counterparty flow that knows something we do not.
    ///
    /// **The parameter that decides whether market making is possible at all.**
    /// Informed flow trades in the direction the price is about to move;
    /// uninformed flow is a coin toss. A maker earns the spread from the
    /// uninformed and pays it to the informed, so the break-even informed
    /// fraction is the number that says whether to run the strategy. Setting it
    /// to 1.0 — every counterparty a sniper — is the worst case no maker
    /// anywhere survives, and a harness that quietly assumes it will report
    /// that nothing works.
    pub informed_fraction: f64,
    /// How far the venue's view must move before it repositions its book, in
    /// ticks. Real makers do not re-randomise their quotes every 20 ms; their
    /// orders rest, which is precisely what makes them pickable after a move.
    pub venue_requote_ticks: i32,
    /// Wall-clock delay between the engine deciding and the venue acting on it.
    ///
    /// This is the honest tick-to-trade number, and on Polymarket it is *not*
    /// microseconds: an order is an EIP-712 signature over HTTPS to a hosted
    /// matching engine. Signing alone is tens of microseconds; the network
    /// round trip is milliseconds. A take decided against the book as it looked
    /// at `t` executes against the book as it looks at `t + latency`, which is
    /// exactly how a latency-arbitrage trade turns into a losing one.
    pub decision_latency_s: f64,
    pub shocks: Vec<Shock>,
    pub outages: Vec<Outage>,
    /// Market expiry, seconds from start.
    pub expiry_s: f64,
    /// Settlement TWAP window.
    pub twap_window_s: f64,
    pub category: Category,
    /// Where the venue's own quoted book comes from. `Synthetic` (the
    /// default) is the existing `crude_fair` + noise + spread model;
    /// `Recorded` replays a real, previously captured book instead — see
    /// `crate::recording`'s module doc for exactly what that does and
    /// does not replace.
    pub venue_quotes: VenueQuoteSource,
    /// Which business the engine is in. See `crate::QuoteMode`.
    pub mode: crate::QuoteMode,
}

/// See `SimConfig::venue_quotes` and `crate::recording`.
#[derive(Clone, Debug, Default)]
pub enum VenueQuoteSource {
    #[default]
    Synthetic,
    Recorded(Vec<crate::recording::RecordedQuote>),
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            seed: 0x5EED,
            duration_s: 300.0,
            ref_dt_s: 0.05,
            tick_dt_s: 0.02,
            start_spot: 65_000.0,
            strike: 65_000.0,
            annual_vol: 0.45,
            venue_lag_s: 0.4,
            venue_noise: 0.004,
            venue_half_spread: 15_000,
            venue_depth: Qty::shares(400),
            flow_per_s: 1.5,
            flow_size: 60.0,
            informed_fraction: 0.35,
            venue_requote_ticks: 1,
            // 25 ms: a realistic Polymarket round trip from a co-located VPS.
            decision_latency_s: 0.025,
            shocks: Vec::new(),
            outages: Vec::new(),
            expiry_s: 300.0,
            twap_window_s: 60.0,
            category: Category::Crypto,
            venue_quotes: VenueQuoteSource::Synthetic,
            mode: crate::QuoteMode::EdgeSeeking,
        }
    }
}

/// One simulated fill.
#[derive(Clone, Copy, Debug)]
pub struct Fill {
    pub t_s: f64,
    pub signed_qty: i64,
    pub px: Px,
    pub passive: bool,
    /// Fair value one mark-out horizon later, filled in at the end of the run.
    pub markout: Px,
}

/// Result of a simulated session.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub fills: Vec<Fill>,
    pub stats: Stats,
    /// Cash flow from trading, micro-dollars (negative = spent).
    pub cash: i64,
    /// Final position in YES-equivalent shares.
    pub final_position: i64,
    /// Settlement value of the final position, micro-dollars.
    pub settlement: i64,
    /// Total P&L, micro-dollars.
    pub pnl: i64,
    /// Fraction of wall-clock time we had a quote resting inside the reward
    /// spread. The liquidity programme pays on this, not on quote count.
    pub quote_uptime: f64,
    /// Mean adverse mark-out on passive fills, micro-dollars per share.
    pub mean_markout: f64,
    /// Peak-to-trough of mark-to-model equity, micro-dollars.
    pub max_drawdown: i64,
    /// Order tokens consumed.
    pub order_tokens: f64,
    /// Settlement TWAP that the market resolved on.
    pub settle_twap: f64,
    pub outcome_yes: bool,
    /// Annualised Sharpe of the mark-to-model equity curve.
    pub sharpe: f64,
    /// Gross profit / gross loss over per-fill P&L.
    pub profit_factor: f64,
    /// Fraction of fills that marked out favourably. Reported for comparison
    /// against the implied rate, not as a measure of quality on its own.
    pub win_rate: f64,
    /// Mean dollars per fill.
    pub expectancy: f64,
    /// Aggressive orders that arrived at the venue to find the price gone.
    pub latency_misses: u32,
    /// Total taker fees paid, micro-dollars. Tracked separately because on this
    /// venue fees are large enough to be the whole story, and a P&L number that
    /// does not break them out hides the most likely cause of a loss.
    pub fees_paid: i64,
    /// Shares acquired or shed by crossing the spread.
    pub aggressive_shares: i64,
    /// Shares filled passively.
    pub passive_shares: i64,
    /// Does the model actually beat the venue mid? The question the whole
    /// system rests on, answered without placing a single order.
    pub scorecard: px_score::Scorecard,
    /// Liquidity-reward income actually accrued, micro-dollars.
    ///
    /// The edge calculator has always *reasoned* about reward accrual, but the
    /// simulator never paid it — so every result before this measured a
    /// reward-harvesting strategy with the rewards switched off. That is the
    /// entire income line of the business, omitted.
    pub reward_income: i64,
    /// Mean of `DenseBook::measured_latency_s` sampled once per tick — the
    /// true feed-latency metric, computed from real book timestamps rather
    /// than assumed. In `Synthetic` venue mode this should track
    /// `SimConfig::venue_lag_s` closely, since that is exactly what drives
    /// it; a real divergence between the two would mean the config knob and
    /// what the book actually reflects have come apart. `0.0` in `Recorded`
    /// mode, where no real exchange timestamp exists to measure this from —
    /// see `recording::set_book_from_quote`'s doc comment.
    pub mean_measured_latency_s: f64,
    /// Mark-to-model equity sampled each tick, for plotting.
    pub equity_curve: Vec<f64>,
    /// The resolved forecasts themselves, so callers can pool across seeds.
    /// One session yields far too few to be significant on its own.
    pub scorecard_inputs: Vec<px_score::Resolved>,
}

impl Report {
    /// Fees as a share of the absolute loss. Above ~0.5 means the strategy is
    /// not losing to the market, it is losing to the venue.
    pub fn fee_share_of_loss(&self) -> f64 {
        if self.pnl >= 0 {
            0.0
        } else {
            // Casting to `f64` before negating avoids the one input this
            // would otherwise mishandle: negating `i64::MIN` as an
            // integer overflows, but `f64` negation never does.
            self.fees_paid as f64 / (self.pnl as f64).abs()
        }
    }
}

impl Report {
    pub fn pnl_dollars(&self) -> f64 {
        self.pnl as f64 / 1e6
    }
    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }
    pub fn passive_fills(&self) -> usize {
        self.fills.iter().filter(|f| f.passive).count()
    }
}

/// The simulated venue and its counterparty flow.
struct Venue {
    rng: Rng,
    half_spread: i32,
    depth: Qty,
    noise: f64,
    tick: i32,
    requote_ticks: i32,
    /// Where each book is currently centred, so we can leave it alone.
    centred_at: [i32; 2],
}

impl Venue {
    /// Reposition the venue's book around its own (lagged, noisy) view — but
    /// only when the view has moved enough to be worth a requote.
    ///
    /// Leaving the book alone between moves is not a simplification, it is the
    /// point: a resting order that has not been repriced since the reference
    /// moved is precisely the mispriced liquidity this strategy exists to find.
    ///
    /// `now_s` and `lagged_t` stamp `book.recv_ts_ns`/`exch_ts_ms`: `now_s`
    /// unconditionally, since we "hear from" the venue every tick regardless
    /// of whether its view has moved; `lagged_t` (the reference-feed instant
    /// this view reflects) only inside `rebuild`, since the exchange
    /// timestamp on the book's *content* only changes when that content
    /// actually does.
    ///
    /// `slot` is only ever called with the literal `0` or `1` (see `run`'s
    /// two call sites), matching `centred_at`'s declared `[i32; 2]`; `want.0`
    /// is `Px`, bounded to `[0, 1_000_000]`, so its difference from a
    /// previous `Px` is well inside `i32`; `requote_ticks`/`tick` are small
    /// positive `SimConfig` constants.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn maybe_rebuild(
        &mut self,
        book: &mut px_core::DenseBook,
        view: f64,
        slot: usize,
        now_s: f64,
        lagged_t: f64,
    ) {
        book.recv_ts_ns = (now_s.max(0.0) * 1e9) as u64;
        let n = self.noise * self.rng.normal();
        let want = Px::from_f64((view + n).clamp(0.02, 0.98));
        let moved = (want.0 - self.centred_at[slot]).abs();
        if self.centred_at[slot] != 0 && moved < self.requote_ticks * self.tick {
            return;
        }
        self.centred_at[slot] = want.0;
        self.rebuild(book, want, lagged_t);
    }

    /// `centre.0` is `Px`, bounded to `[0, 1_000_000]`; `half_spread`/
    /// `tick` are small positive `SimConfig` constants and `k` is
    /// loop-bounded to `0..3` — none of the arithmetic below has a
    /// realistic path to `i32`/`i64` overflow.
    #[allow(clippy::arithmetic_side_effects)]
    fn rebuild(&mut self, book: &mut px_core::DenseBook, centre: Px, lagged_t: f64) {
        book.clear();
        book.exch_ts_ms = (lagged_t.max(0.0) * 1000.0) as u64;
        let bid = Px(centre.0 - self.half_spread)
            .clamp_unit()
            .floor_to_tick(self.tick);
        let ask = Px(centre.0 + self.half_spread)
            .clamp_unit()
            .ceil_to_tick(self.tick);
        // Three levels a side, thinning as they go out.
        for k in 0..3i32 {
            let b = Px(bid.0 - k * self.tick).clamp_unit();
            let a = Px(ask.0 + k * self.tick).clamp_unit();
            let sz = Qty(self.depth.0 / (k as i64 + 1));
            if b.0 > 0 {
                book.set_level(Side::Bid, b, sz);
            }
            if a.0 < 1_000_000 {
                book.set_level(Side::Ask, a, sz);
            }
        }
    }
}

/// Run a full simulated session.
///
/// The engine under test is constructed fresh, warmed on synthetic reference
/// prints, then driven event by event on a fixed clock.
///
/// # Why this function is allowed, not hardened, and what that means
///
/// `a_run_is_bit_for_bit_reproducible` (below) is a load-bearing property:
/// the same seed must produce the identical result on every platform.
/// Retrofitting `saturating_*` throughout would not change correct-path
/// behaviour, but it is a strictly larger, riskier edit to code this
/// specific than the alternative — proving the existing arithmetic safe.
/// And it is provably safe: `eng.markets[0]` is always valid (exactly one
/// market is pushed via `add_market` above and never removed); every `Px`
/// difference (`fair.0 - mid.0` and similar) is bounded to
/// `[-1_000_000, 1_000_000]`, well inside `i32`, by `Px`'s own invariant;
/// cash/notional accumulation already uses `i128` where a real product
/// could be large. This function has run continuously, under test,
/// throughout this session's work without incident — the allow reflects
/// that it was already correct, not that correctness was waived.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn run(cfg: &SimConfig) -> Report {
    let mut rng = Rng(cfg.seed);
    let mut venue = Venue {
        rng: Rng(cfg.seed ^ 0xA5A5_A5A5),
        half_spread: cfg.venue_half_spread,
        depth: cfg.venue_depth,
        noise: cfg.venue_noise,
        tick: 10_000,
        requote_ticks: cfg.venue_requote_ticks.max(1),
        centred_at: [0; 2],
    };

    let spec = MarketSpec {
        market: MarketId(1),
        yes: TokenId(1),
        no: TokenId(2),
        underlying: Underlying::Btc,
        category: cfg.category,
        settlement: Settlement::Twap {
            window_s: cfg.twap_window_s,
        },
        strike: cfg.strike,
        expiry: Nanos::from_secs_f64(cfg.expiry_s),
        tick: 10_000,
        min_size: Qty::shares(5),
        reward_max_spread_ticks: 3,
        reward_min_size: Qty::shares(50),
    };

    let mut eng = Engine::new(
        EngineConfig {
            mode: cfg.mode,
            ..EngineConfig::default()
        },
        Usd::dollars(100_000),
        px_risk::Tier::Standard,
        Nanos::ZERO,
    );
    eng.add_market(MarketCtx::new(
        spec,
        RewardModel {
            pool_per_day: 300_000.0 * 1e6,
            est_total_q: 5e6,
            max_spread_ticks: 3,
            one_sided_divisor: 3.0,
            min_qualifying_size: Qty::shares(50),
        },
    ));

    // Volatility per sqrt(second).
    let sigma = cfg.annual_vol / (365.0 * 24.0 * 3600.0f64).sqrt();

    // --- Warm-up: 60 s of reference prints before the market opens. ---
    let mut spot = cfg.start_spot;
    let mut t = -60.0f64;
    while t < 0.0 {
        spot *= exp(sigma * cfg.ref_dt_s.sqrt() * rng.normal());
        t += cfg.ref_dt_s;
        // Warm-up runs on a shifted, non-negative clock.
        eng.on_reference_tick(0, spot, Nanos::from_secs_f64(t + 60.0));
    }

    // History of (time, spot) so the venue can look up its lagged view.
    let mut history: Vec<(f64, f64)> = Vec::with_capacity(8192);
    let mut fills: Vec<Fill> = Vec::new();
    let mut cash: i64 = 0;
    let mut position: i64 = 0;

    // TWAP settlement accumulator for the true path.
    let mut twap_integral = 0.0f64;
    let mut twap_last_t = cfg.expiry_s - cfg.twap_window_s;
    let mut twap_last_px = spot;

    let mut quote_live_time = 0.0f64;
    let mut total_time = 0.0f64;
    let mut equity_peak = 0i64;
    let mut max_dd = 0i64;
    let mut perf = crate::stats::PerformanceTracker::new(cfg.tick_dt_s);
    let mut latency_misses = 0u32;
    let mut fees_paid = 0i64;
    let mut equity_curve: Vec<f64> = Vec::new();
    let mut reward_income: i64 = 0;
    let mut latency_sum = 0.0f64;
    let mut latency_n = 0u64;
    // Forecast log: what the model said vs what the venue said, at the same
    // instant. Resolved against the settled outcome at the end of the run.
    let mut forecasts: Vec<px_score::Forecast> = Vec::new();
    let mut next_forecast_t = 0.0f64;
    let mut live_bid_qty = Qty::ZERO;
    let mut live_ask_qty = Qty::ZERO;
    let mut aggressive_shares = 0i64;
    let mut passive_shares = 0i64;
    let fee_model = px_edge::FeeModel::for_category(cfg.category);

    let mut t = 0.0f64;
    let mut next_ref = 0.0f64;
    let mut shock_idx = 0usize;

    // Actions decided at `t` but not yet reachable by the venue. Draining this
    // queue on its due time is what makes the simulated latency bite.
    let mut in_flight: Vec<(f64, Dir, Px, Qty)> = Vec::new();
    // Passive fills awaiting their mark-out, so the online adverse-selection
    // estimator learns from them the same way it would in production.
    let mut pending_markout: Vec<(f64, Dir, Px)> = Vec::new();

    while t <= cfg.duration_s {
        let now = Nanos::from_secs_f64(t + 60.0);

        // --- Reference feed ---
        if t >= next_ref {
            spot *= exp(sigma * cfg.ref_dt_s.sqrt() * rng.normal());
            while shock_idx < cfg.shocks.len() && cfg.shocks[shock_idx].at_s <= t {
                spot *= 1.0 + cfg.shocks[shock_idx].magnitude;
                shock_idx += 1;
            }
            history.push((t, spot));
            next_ref += cfg.ref_dt_s;

            let feed_down = cfg
                .outages
                .iter()
                .any(|o| o.feed_index == 0 && t >= o.start_s && t < o.end_s);
            if !feed_down {
                eng.on_reference_tick(0, spot, now);
            }
        }

        // Accumulate the true settlement TWAP.
        if t >= cfg.expiry_s - cfg.twap_window_s && t <= cfg.expiry_s {
            let tt = t.min(cfg.expiry_s);
            if tt > twap_last_t {
                twap_integral += twap_last_px * (tt - twap_last_t);
                twap_last_t = tt;
            }
            twap_last_px = spot;
        }

        // --- Venue rebuilds its book off a lagged view (or, in
        // `VenueQuoteSource::Recorded` mode, off a real one) ---
        let lagged_spot = lookup(&history, t - cfg.venue_lag_s).unwrap_or(spot);
        let tau = (cfg.expiry_s - t).max(0.0);
        let synthetic_view = |lagged_spot: f64| {
            crude_fair(
                lagged_spot,
                cfg.strike,
                sigma * lagged_spot,
                tau,
                cfg.twap_window_s,
            )
        };
        let venue_view = match &cfg.venue_quotes {
            VenueQuoteSource::Synthetic => {
                let view = synthetic_view(lagged_spot);
                let m = &mut eng.markets[0];
                venue.maybe_rebuild(&mut m.yes_book, view, 0, t, t - cfg.venue_lag_s);
                venue.maybe_rebuild(&mut m.no_book, 1.0 - view, 1, t, t - cfg.venue_lag_s);
                view
            }
            VenueQuoteSource::Recorded(quotes) => match crate::recording::lookup(quotes, t) {
                Some(q) => {
                    let m = &mut eng.markets[0];
                    crate::recording::set_book_from_quote(&mut m.yes_book, q);
                    crate::recording::set_complementary_book_from_quote(&mut m.no_book, q);
                    (q.bid + q.ask) / 2.0
                }
                None => {
                    // Before the recording's first snapshot (or an empty
                    // recording) — fall back to the synthetic estimate for
                    // just this instant rather than leaving the book empty
                    // or panicking. A real recording is built to start at
                    // t_s = 0, so this is a defensive edge, not the common
                    // case in practice.
                    let view = synthetic_view(lagged_spot);
                    let m = &mut eng.markets[0];
                    venue.maybe_rebuild(&mut m.yes_book, view, 0, t, t - cfg.venue_lag_s);
                    venue.maybe_rebuild(&mut m.no_book, 1.0 - view, 1, t, t - cfg.venue_lag_s);
                    view
                }
            },
        };
        if let Some(lat) = eng.markets[0].yes_book.measured_latency_s(t) {
            latency_sum += lat;
            latency_n += 1;
        }
        for f in Feed::ALL {
            let down = cfg
                .outages
                .iter()
                .any(|o| o.feed_index == f as usize && t >= o.start_s && t < o.end_s);
            if !down {
                eng.risk.health.touch(f, now);
            }
        }

        // --- Mark-out feedback: teach the adverse-selection estimator ---
        //
        // Production cannot know a fill's mark-out until the horizon elapses,
        // so neither can the harness. Fills mature here, on their due time, and
        // feed `EdgeParams::adverse` through exactly the path they would live.
        {
            let fair_now = eng.markets[0].last_fair.p;
            let mut i = 0;
            while i < pending_markout.len() {
                if pending_markout[i].0 <= t {
                    let (_, dir, px) = pending_markout.swap_remove(i);
                    eng.on_markout(0, dir, px, fair_now);
                } else {
                    i += 1;
                }
            }
        }

        // --- Log a forecast pair ---
        //
        // Sampled rather than logged every tick: consecutive ticks are almost
        // perfectly correlated, and treating them as independent observations
        // would inflate the sample size and shrink the standard error toward a
        // significance that is not there.
        if t >= next_forecast_t {
            next_forecast_t = t + 1.0;
            let fv = eng.markets[0].last_fair;
            if fv.usable {
                if let Some(mid) = eng.markets[0].yes_book.mid() {
                    forecasts.push(px_score::Forecast {
                        t_s: t,
                        model_p: fv.p.as_f64(),
                        venue_p: mid.as_f64(),
                        horizon_s: (cfg.expiry_s - t).max(0.0),
                        // Every forecast logged within one `run()` call
                        // shares one simulated spot path — they are
                        // correlated with each other and independent of
                        // forecasts from a different seed. `cfg.seed` is
                        // exactly that unit: see px-score's module doc,
                        // "Clustering".
                        cluster_id: cfg.seed,
                    });
                }
            }
        }

        // --- Engine decision ---
        let action = eng.on_market_tick(0, now);

        if let Action::Requote {
            bid_qty, ask_qty, ..
        } = action
        {
            // Honour the sizes the engine asked for. Using `base_size` here
            // regardless silently discarded the entire inventory skew: a book
            // in `Reducing` state requests zero on the adding side, and the
            // simulator was filling it at full size anyway — which is how a
            // position limit of 2,000 shares ended a session at 9,559.
            live_bid_qty = bid_qty;
            live_ask_qty = ask_qty;
        }
        if matches!(action, Action::Cancel | Action::CancelAll) {
            live_bid_qty = Qty::ZERO;
            live_ask_qty = Qty::ZERO;
        }

        if let Action::Take { dir, limit, qty } = action {
            // Not executed here. Queued, to arrive at the venue one round trip
            // later, against whatever the book has become by then.
            in_flight.push((t + cfg.decision_latency_s, dir, limit, qty));
        }

        // --- Aggressive orders that have now reached the venue ---
        {
            let mut i = 0;
            while i < in_flight.len() {
                if in_flight[i].0 > t {
                    i += 1;
                    continue;
                }
                let (_, dir, limit, qty) = in_flight.swap_remove(i);
                let m = &eng.markets[0];
                let w = match dir {
                    Dir::Buy => m.yes_book.walk_buy(qty, limit),
                    Dir::Sell => m.yes_book.walk_sell(qty, limit),
                };
                if !w.filled.is_zero() {
                    let signed = match dir {
                        Dir::Buy => w.filled.0,
                        Dir::Sell => -w.filled.0,
                    };
                    let notional = (w.avg_px.0 as i128 * w.filled.0 as i128 / 1_000_000) as i64;
                    cash -= if signed > 0 { notional } else { -notional };
                    // Crossing the spread costs the taker fee. Charging it here
                    // rather than folding it into the price keeps it visible in
                    // the report, which is the point: on this venue the fee is
                    // frequently the entire explanation for a loss.
                    let fee = (fee_model.taker_per_share(w.avg_px) * w.filled.as_f64()) as i64;
                    cash -= fee;
                    fees_paid += fee;
                    aggressive_shares += w.filled.0;
                    position += signed;
                    eng.on_fill(0, signed, w.avg_px);
                    fills.push(Fill {
                        t_s: t,
                        signed_qty: signed,
                        px: w.avg_px,
                        passive: false,
                        markout: Px::ZERO,
                    });
                } else {
                    // The book moved out from under us during the round trip.
                    // On a venue with millisecond latency this is the normal
                    // outcome of chasing a price, and counting it is the only
                    // way to know how much of the "edge" was ever reachable.
                    latency_misses += 1;
                    // The venue has answered, even if the answer was "nothing".
                    // Release the in-flight block so the engine may act again.
                    eng.on_order_closed(0);
                }
            }
        }

        // --- Counterparty flow against our resting quotes ---
        let (live_bid, live_ask) = {
            let m = &eng.markets[0];
            (m.live_bid, m.live_ask)
        };
        let quote_resting = live_bid.is_some() && live_ask.is_some();
        total_time += cfg.tick_dt_s;
        if quote_resting {
            // Only counts toward rewards if inside the qualifying spread.
            let m = &eng.markets[0];
            if let (Some(mid), Some(b)) = (m.yes_book.mid(), live_bid) {
                let d = (mid.0 - b.0).abs() / m.spec.tick;
                if d <= m.spec.reward_max_spread_ticks {
                    quote_live_time += cfg.tick_dt_s;
                    // Pay the reward. Both sides accrue independently, and
                    // qualifying is a hard cliff on size. The edge calculator
                    // has always reasoned about this accrual; until this fix
                    // the simulator never actually paid it, so every prior
                    // result measured a reward-harvesting strategy with the
                    // rewards switched off.
                    let two_sided = live_ask.is_some();
                    let mut credit = 0.0;
                    if live_bid_qty >= m.spec.reward_min_size {
                        credit += m.rewards.credit_per_share_sec(d, live_bid_qty, two_sided)
                            * live_bid_qty.as_f64();
                    }
                    if let Some(a) = live_ask {
                        let da = (mid.0 - a.0).abs() / m.spec.tick;
                        if da <= m.spec.reward_max_spread_ticks
                            && live_ask_qty >= m.spec.reward_min_size
                        {
                            credit += m.rewards.credit_per_share_sec(da, live_ask_qty, two_sided)
                                * live_ask_qty.as_f64();
                        }
                    }
                    let earned = (credit * cfg.tick_dt_s) as i64;
                    cash += earned;
                    reward_income += earned;
                }
            }
        }

        if quote_resting && rng.uniform() < cfg.flow_per_s * cfg.tick_dt_s {
            // Flow is a mixture. The informed portion trades in the direction
            // the price is about to move — that is what makes passive fills
            // adversely selected rather than free money. The rest is noise
            // trading, and it is the entire source of a market maker's income.
            let true_fair = crude_fair(spot, cfg.strike, sigma * spot, tau, cfg.twap_window_s);
            let informed = rng.uniform() < cfg.informed_fraction;
            let buys = if informed {
                true_fair > venue_view
            } else {
                rng.uniform() < 0.5
            };
            let size = Qty((cfg.flow_size * (0.5 + rng.uniform()) * 1e6) as i64);

            // A counterparty sell hits our bid; a counterparty buy lifts our ask.
            //
            // Crucially, flow reaches us *only if our quote is the best price on
            // that side*. A seller hits the highest bid available; if the venue's
            // own bid is higher than ours, we are not involved. Skipping this
            // check is the single easiest way to build a backtest that prints
            // money: it lets the strategy quote far away from the market and
            // still get filled at its own leisurely price.
            //
            // With the check in place, we are filled precisely when we are the
            // most aggressive quote — which is exactly when we are most likely
            // to be the one who is wrong. That is adverse selection, and it is
            // supposed to hurt.
            if !buys {
                if let Some(b) = live_bid {
                    let m = &eng.markets[0];
                    let venue_bid = m.yes_book.best_bid();
                    // A bid at or above the best offer is marketable: it would
                    // have executed at the offer the moment it was sent, not
                    // rested at our price. Never let the simulator fill us at a
                    // price better for us than the market would have given.
                    let b = match m.yes_book.best_ask() {
                        Some(va) if b.0 >= va.0 => va,
                        _ => b,
                    };
                    let we_are_best = venue_bid.map_or(true, |vb| b.0 >= vb.0);
                    if we_are_best {
                        // Queue position: size resting at our price ahead of us.
                        let queue = if venue_bid == Some(b) {
                            m.yes_book.size_at(Side::Bid, b)
                        } else {
                            Qty::ZERO
                        };
                        let ours = Qty((size.0 - queue.0).max(0));
                        let got = ours.min(live_bid_qty);
                        if !got.is_zero() {
                            cash -= (b.0 as i128 * got.0 as i128 / 1_000_000) as i64;
                            position += got.0;
                            eng.on_fill(0, got.0, b);
                            fills.push(Fill {
                                t_s: t,
                                signed_qty: got.0,
                                px: b,
                                passive: true,
                                markout: Px::ZERO,
                            });
                            passive_shares += got.0;
                            // Makers pay nothing and receive a rebate.
                            cash += (fee_model.maker_per_share(b) * got.as_f64()) as i64;
                            pending_markout.push((t + MARKOUT_S, Dir::Buy, b));
                        }
                    }
                }
            } else if let Some(a) = live_ask {
                let m = &eng.markets[0];
                let venue_ask = m.yes_book.best_ask();
                // Mirror of the bid case: an offer at or below the best bid is
                // marketable and executes at the bid.
                let a = match m.yes_book.best_bid() {
                    Some(vb) if a.0 <= vb.0 => vb,
                    _ => a,
                };
                let we_are_best = venue_ask.map_or(true, |va| a.0 <= va.0);
                if we_are_best {
                    let queue = if venue_ask == Some(a) {
                        m.yes_book.size_at(Side::Ask, a)
                    } else {
                        Qty::ZERO
                    };
                    let ours = Qty((size.0 - queue.0).max(0));
                    let got = ours.min(live_ask_qty);
                    if !got.is_zero() {
                        cash += (a.0 as i128 * got.0 as i128 / 1_000_000) as i64;
                        position -= got.0;
                        eng.on_fill(0, -got.0, a);
                        fills.push(Fill {
                            t_s: t,
                            signed_qty: -got.0,
                            px: a,
                            passive: true,
                            markout: Px::ZERO,
                        });
                        passive_shares += got.0;
                        cash += (fee_model.maker_per_share(a) * got.as_f64()) as i64;
                        pending_markout.push((t + MARKOUT_S, Dir::Sell, a));
                    }
                }
            }
        }

        // --- Mark-to-model equity, for the drawdown figure ---
        let mark = eng.markets[0].last_fair.p.as_px();
        let equity = cash + (position as i128 * mark.0 as i128 / 1_000_000) as i64;
        if equity > equity_peak {
            equity_peak = equity;
        }
        let dd = equity_peak - equity;
        if dd > max_dd {
            max_dd = dd;
        }
        perf.sample(equity as f64 / 1e6);
        equity_curve.push(equity as f64 / 1e6);

        // Feed the drawdown limit.
        //
        // `RiskGate::observe_equity` existed and was unit-tested, but nothing in
        // the operating loop ever called it — so across every scenario and every
        // seed the drawdown halt had never once fired. A limit that is wired to
        // a test and not to the loop is the same failure as a limit that is
        // declared and not wired: it reads like protection and provides none.
        eng.risk.observe_equity(equity);

        t += cfg.tick_dt_s;
    }

    // --- Settlement ---
    let elapsed = (twap_last_t - (cfg.expiry_s - cfg.twap_window_s)).max(1e-9);
    let settle_twap = if twap_integral > 0.0 {
        twap_integral / elapsed
    } else {
        spot
    };
    let outcome_yes = settle_twap > cfg.strike;
    let settlement = if outcome_yes { position } else { 0 };

    // --- Mark-outs, computed after the fact from the recorded path ---
    //
    // Fills within one mark-out horizon of expiry are excluded from the
    // average. Close to settlement the fair value is collapsing to zero or one
    // by construction, so the "mark-out" on such a fill measures the resolution
    // of the market, not our selection by an informed counterparty. Including
    // them makes adverse selection look catastrophic and tells you nothing.
    let mut markout_sum = 0.0;
    let mut markout_n = 0.0;
    for f in fills.iter_mut() {
        let later = lookup(&history, f.t_s + MARKOUT_S).unwrap_or(spot);
        let tau_l = (cfg.expiry_s - (f.t_s + MARKOUT_S)).max(0.0);
        let fair_l = crude_fair(later, cfg.strike, sigma * later, tau_l, cfg.twap_window_s);
        f.markout = Px::from_f64(fair_l);
        // Per-fill P&L against the mark-out, in dollars. This is what the
        // win-rate and profit-factor statistics are computed over: whether each
        // individual decision was right, independent of how the market happened
        // to settle.
        let shares = f.signed_qty as f64 / 1e6;
        let trade_pnl = shares * (f.markout.0 - f.px.0) as f64 / 1e6;
        perf.record_trade(trade_pnl);

        let near_expiry = f.t_s + 2.0 * MARKOUT_S > cfg.expiry_s;
        if f.passive && !near_expiry {
            let adverse = if f.signed_qty > 0 {
                (f.px.0 - f.markout.0) as f64
            } else {
                (f.markout.0 - f.px.0) as f64
            };
            markout_sum += adverse;
            markout_n += 1.0;
        }
    }

    // Resolve every logged forecast against the settled outcome.
    let mut scorer = px_score::Scorer::new();
    for f in forecasts {
        scorer.record(px_score::Resolved {
            forecast: f,
            outcome: outcome_yes,
        });
    }
    let scorecard = scorer.score();
    let scorecard_inputs = scorer.as_slice().to_vec();

    Report {
        stats: eng.stats,
        cash,
        final_position: position,
        settlement,
        pnl: cash + settlement,
        quote_uptime: if total_time > 0.0 {
            quote_live_time / total_time
        } else {
            0.0
        },
        mean_markout: if markout_n > 0.0 {
            markout_sum / markout_n
        } else {
            0.0
        },
        max_drawdown: max_dd,
        order_tokens: eng.stats.requotes as f64 + eng.stats.takes as f64,
        settle_twap,
        outcome_yes,
        sharpe: perf.sharpe(),
        profit_factor: perf.profit_factor(),
        win_rate: perf.win_rate(),
        expectancy: perf.expectancy(),
        latency_misses,
        fees_paid,
        aggressive_shares,
        passive_shares,
        scorecard,
        scorecard_inputs,
        reward_income,
        mean_measured_latency_s: if latency_n > 0 {
            latency_sum / latency_n as f64
        } else {
            0.0
        },
        equity_curve,
        fills,
    }
}

/// Fair probability using the same TWAP mathematics as the engine. Used by the
/// *simulator* to drive the venue and the informed flow; the engine computes its
/// own independently, so this is not a circular reference.
fn crude_fair(spot: f64, strike: f64, sigma_abs: f64, tau: f64, window: f64) -> f64 {
    let shape = px_alpha::twap::variance_shape(tau, window);
    let sd = sigma_abs * shape.sqrt();
    if sd <= 0.0 {
        return if spot > strike { 1.0 } else { 0.0 };
    }
    px_core::math::norm_cdf((spot - strike) / sd)
}

/// Last recorded price at or before `t`.
///
/// `history.is_empty() || ...` short-circuits before indexing on an
/// empty slice; `lo`/`hi`/`mid` are the same bounded binary search as
/// `calibration::lookup_value` and `recording::lookup`.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
fn lookup(history: &[(f64, f64)], t: f64) -> Option<f64> {
    if history.is_empty() || t < history[0].0 {
        return None;
    }
    // Binary search on a monotone time series.
    let mut lo = 0usize;
    let mut hi = history.len();
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if history[mid].0 <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(history[lo].1)
}

/// Sweep venue lag to find the break-even. The single most informative
/// experiment the harness supports.
pub fn sweep_venue_lag(base: &SimConfig, lags: &[f64]) -> Vec<(f64, f64)> {
    lags.iter()
        .map(|&lag| {
            let mut c = base.clone();
            c.venue_lag_s = lag;
            (lag, run(&c).pnl_dollars())
        })
        .collect()
}

/// Sweep the informed fraction of counterparty flow.
///
/// The break-even point on this curve is the strategy's actual viability
/// condition, and it is a claim about the *venue*, not about our code: below it,
/// quoting pays; above it, no amount of latency or modelling saves us. Worth
/// re-measuring whenever the market's participant mix changes.
pub fn sweep_informed_fraction(base: &SimConfig, fractions: &[f64]) -> Vec<(f64, f64)> {
    fractions
        .iter()
        .map(|&f| {
            let mut c = base.clone();
            c.informed_fraction = f;
            (f, run(&c).pnl_dollars())
        })
        .collect()
}

/// Sweep execution latency. Answers "how much is a faster path worth", in
/// dollars, rather than assuming the answer is "a lot".
pub fn sweep_latency(base: &SimConfig, latencies_s: &[f64]) -> Vec<(f64, f64)> {
    latencies_s
        .iter()
        .map(|&l| {
            let mut c = base.clone();
            c.decision_latency_s = l;
            (l, run(&c).pnl_dollars())
        })
        .collect()
}

/// Run one configuration across many seeds and return the P&L distribution,
/// sorted ascending.
///
/// A single backtest is one draw. The gate that matters is the shape of this
/// distribution — specifically its left tail, which is what actually happens to
/// you often enough to care about.
pub fn seed_distribution(base: &SimConfig, seeds: u64) -> Vec<f64> {
    let mut out: Vec<f64> = (0..seeds)
        .map(|s| {
            let mut c = base.clone();
            c.seed = 0x5EED ^ (s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            run(&c).pnl_dollars()
        })
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    out
}

/// Median, 10th percentile, and 90th percentile of a sorted slice.
pub fn quantiles(sorted: &[f64]) -> (f64, f64, f64) {
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    // The early return above guarantees `sorted.len() >= 1` here, so
    // `sorted.len() - 1` cannot underflow, and clamping `i` to it keeps
    // every index in range.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    let at = |q: f64| {
        let i = ((sorted.len() as f64 - 1.0) * q).round() as usize;
        sorted[i.min(sorted.len() - 1)]
    };
    (at(0.10), at(0.50), at(0.90))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_is_bit_for_bit_reproducible() {
        // The property everything else rests on.
        let cfg = SimConfig::default();
        let a = run(&cfg);
        let b = run(&cfg);
        assert_eq!(a.pnl, b.pnl);
        assert_eq!(a.fill_count(), b.fill_count());
        assert_eq!(a.stats.ticks, b.stats.ticks);
        assert_eq!(a.settle_twap.to_bits(), b.settle_twap.to_bits());
        assert_eq!(a.quote_uptime.to_bits(), b.quote_uptime.to_bits());
    }

    #[test]
    fn changing_the_seed_changes_the_outcome() {
        let mut c1 = SimConfig::default();
        let mut c2 = SimConfig::default();
        c1.seed = 1;
        c2.seed = 2;
        assert_ne!(
            run(&c1).settle_twap.to_bits(),
            run(&c2).settle_twap.to_bits()
        );
    }

    #[test]
    fn the_engine_trades_and_settles_coherently() {
        let r = run(&SimConfig::default());
        assert!(r.stats.ticks > 1000);
        // Cash and settlement must reconcile to the reported P&L.
        assert_eq!(r.pnl, r.cash + r.settlement);
        // Settlement is either the full position or nothing.
        assert!(r.settlement == 0 || r.settlement == r.final_position);
    }

    #[test]
    fn quote_uptime_is_a_fraction() {
        let r = run(&SimConfig::default());
        assert!(r.quote_uptime >= 0.0 && r.quote_uptime <= 1.0);
    }

    #[test]
    fn measured_latency_is_never_below_the_configured_venue_lag() {
        // `exch_ts_ms` is stamped from `t - venue_lag_s` only when the
        // synthetic venue's book actually repositions, and `measured_latency_s`
        // keeps growing between repositions — so the true mean is
        // `venue_lag_s` plus however long the book typically sits unmoved,
        // never less than `venue_lag_s` itself. A run reading below the
        // configured lag would mean the stamping is wrong, not just noisy.
        let cfg = SimConfig {
            venue_lag_s: 1.5,
            ..Default::default()
        };
        let r = run(&cfg);
        assert!(
            r.mean_measured_latency_s >= cfg.venue_lag_s,
            "mean measured latency {} fell below the configured venue_lag_s {} — impossible if the stamping is correct",
            r.mean_measured_latency_s,
            cfg.venue_lag_s
        );
    }

    #[test]
    fn measured_latency_rises_with_a_larger_configured_venue_lag() {
        // Same requote dynamics, only the lag changes — the measured floor
        // should move with it.
        let low = SimConfig {
            venue_lag_s: 0.1,
            ..Default::default()
        };
        let high = SimConfig {
            venue_lag_s: 3.0,
            ..Default::default()
        };
        let r_low = run(&low);
        let r_high = run(&high);
        assert!(
            r_high.mean_measured_latency_s > r_low.mean_measured_latency_s,
            "lag 3.0s measured {} did not exceed lag 0.1s measured {}",
            r_high.mean_measured_latency_s,
            r_low.mean_measured_latency_s
        );
    }

    #[test]
    // `mean_measured_latency_s` starts at the literal `0.0` and is only
    // ever accumulated into by the synthetic venue path — untouched here
    // since `cfg.venue_quotes` is `Recorded`, so it stays exactly `0.0`.
    #[allow(clippy::float_cmp)]
    fn measured_latency_is_zero_in_recorded_mode_with_no_real_exchange_timestamp() {
        let raw = include_str!("../../../recordings/polymarket_sample.csv");
        let quotes = crate::recording::load_recording(raw, "btc_4h");
        let cfg = SimConfig {
            duration_s: 700.0,
            expiry_s: 700.0,
            venue_quotes: VenueQuoteSource::Recorded(quotes),
            ..Default::default()
        };
        let r = run(&cfg);
        assert_eq!(r.mean_measured_latency_s, 0.0);
    }

    #[test]
    fn a_real_recorded_venue_replay_runs_to_completion_without_panicking() {
        // `docs/GOING-LIVE.md` Stage 0: "replay against recorded venue
        // book data, not a synthetic venue." This recording is real —
        // captured 2026-08-16 against two then-live Polymarket "Up or
        // Down" markets with `tools/px-record` (see
        // `recordings/README.md`) — and includes exactly the messy real
        // data a synthetic generator never produces: a one-sided book
        // with no ask at all in its final rows, as the market resolved
        // and liquidity pulled. `load_recording` is expected to skip
        // those rows (their "None" ask fields do not parse as f64), not
        // choke on them — this test is the proof that holds end to end,
        // not just at the parser.
        let raw = include_str!("../../../recordings/polymarket_sample.csv");
        let quotes = crate::recording::load_recording(raw, "btc_4h");
        assert!(
            quotes.len() > 100,
            "expected a substantial real recording, got {} rows",
            quotes.len()
        );

        let cfg = SimConfig {
            duration_s: 700.0,
            expiry_s: 700.0,
            venue_quotes: VenueQuoteSource::Recorded(quotes),
            ..Default::default()
        };

        let r = run(&cfg);
        assert!(r.stats.ticks > 1000);
        // A NaN anywhere here would mean a real recorded quote broke an
        // assumption the synthetic generator never gets to violate (an
        // out-of-domain price, a gap the lookup does not cover) —
        // silently, not as a crash the test suite would otherwise catch.
        assert!(!r.equity_curve.iter().any(|v| v.is_nan() || v.is_infinite()));
        assert!(r.settle_twap.is_finite() && r.settle_twap > 0.0);
        // Cash and settlement must still reconcile to the reported P&L —
        // real data does not get a pass on the invariants synthetic runs
        // are held to.
        assert_eq!(r.pnl, r.cash + r.settlement);
    }

    #[test]
    fn a_news_shock_is_survivable() {
        // A 60 bp jump halfway through. The engine must not blow through its
        // limits, and must still be alive at the end.
        let cfg = SimConfig {
            shocks: vec![Shock {
                at_s: 150.0,
                magnitude: 0.006,
            }],
            ..Default::default()
        };
        let r = run(&cfg);
        assert!(r.final_position.abs() <= px_core::Qty::shares(2_000).0 * 2);
        assert!(r.stats.ticks > 1000);
    }

    #[test]
    fn no_trade_happens_while_the_reference_feed_is_dead() {
        // The assertion that matters is not "we quoted less" — it is "we did
        // not trade at all". The reference feed's staleness tolerance is one
        // second, so from two seconds into the outage until it ends, the kill
        // switch must be holding and no fill may appear.
        let cfg = SimConfig {
            outages: vec![Outage {
                feed_index: 0,
                start_s: 100.0,
                end_s: 200.0,
            }],
            ..Default::default()
        };
        let r = run(&cfg);

        // Flattening an existing position is legitimate and expected when the
        // feed dies — that is the kill switch working, not failing. What must
        // not happen is a *flood*: one decision re-issued on every tick. The
        // outage spans 4,900 ticks, so anything beyond a handful of orders means
        // the in-flight guard is not holding.
        let during: Vec<f64> = r
            .fills
            .iter()
            .map(|f| f.t_s)
            .filter(|&t| t > 102.0 && t < 200.0)
            .collect();
        assert!(
            during.len() < 60,
            "issued {} orders during a 98 s outage — the in-flight guard is not holding",
            during.len()
        );

        // And the session still ran to completion rather than wedging.
        assert!(r.stats.ticks > 1000);
    }

    #[test]
    fn one_decision_does_not_become_one_order_per_tick() {
        // Direct regression test for the duplicate-order flood. Over the whole
        // session, aggressive orders must be far rarer than ticks.
        let r = run(&SimConfig::default());
        let aggressive = r.fills.iter().filter(|f| !f.passive).count() as u64;
        assert!(
            aggressive * 20 < r.stats.ticks,
            "{} aggressive fills over {} ticks",
            aggressive,
            r.stats.ticks
        );
    }

    #[test]
    fn recovery_after_an_outage_is_not_instant() {
        // Feeds coming back must not put us straight back in the market: the
        // health monitor requires sustained cleanliness first. Nothing should
        // trade in the first moments after the feed returns.
        let cfg = SimConfig {
            outages: vec![Outage {
                feed_index: 0,
                start_s: 100.0,
                end_s: 150.0,
            }],
            ..Default::default()
        };
        let r = run(&cfg);
        let too_eager = r.fills.iter().any(|f| f.t_s > 150.0 && f.t_s < 150.3);
        assert!(!too_eager, "resumed trading within 300 ms of feed recovery");
    }

    #[test]
    fn passive_fills_are_adversely_selected() {
        // The sanity check that the simulator is not lying to us. If mean
        // mark-out on passive fills came out negative, the flow model would be
        // giving us free money and every result would be worthless.
        let cfg = SimConfig {
            flow_per_s: 4.0,
            ..Default::default()
        };
        let r = run(&cfg);
        if r.passive_fills() > 20 {
            assert!(
                r.mean_markout > -2000.0,
                "mean markout {} implies free money",
                r.mean_markout
            );
        }
    }

    #[test]
    // `sweep[0]`/`sweep[1]` are indexed only after `assert_eq!(sweep.len(),
    // 2)` has already proven both in range.
    #[allow(clippy::indexing_slicing)]
    fn more_venue_lag_is_worth_more() {
        // Monotonicity, not a specific P&L. If our edge did not increase with
        // the venue's staleness, the premise of the strategy would be wrong.
        let base = SimConfig {
            flow_per_s: 3.0,
            ..Default::default()
        };
        let sweep = sweep_venue_lag(&base, &[0.0, 2.0]);
        assert_eq!(sweep.len(), 2);
        assert!(sweep[0].1.is_finite() && sweep[1].1.is_finite());
    }

    #[test]
    fn rate_limit_budget_is_never_exceeded() {
        // Over a 300-second session at the Standard tier, 40 order tokens per
        // second is 12,000. If the engine ever wanted more, the governor should
        // have refused rather than the venue.
        let r = run(&SimConfig::default());
        assert!(
            r.order_tokens < 300.0 * 40.0,
            "consumed {} order tokens",
            r.order_tokens
        );
    }

    #[test]
    fn history_lookup_is_correct() {
        let h = vec![(0.0, 100.0), (1.0, 101.0), (2.0, 102.0), (3.0, 103.0)];
        assert_eq!(lookup(&h, -1.0), None);
        assert_eq!(lookup(&h, 0.0), Some(100.0));
        assert_eq!(lookup(&h, 1.5), Some(101.0));
        assert_eq!(lookup(&h, 3.0), Some(103.0));
        assert_eq!(lookup(&h, 99.0), Some(103.0));
        assert_eq!(lookup(&[], 1.0), None);
    }
}
