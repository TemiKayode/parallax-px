//! `px-alpha` — independent fair-probability estimation.
//!
//! # The one rule
//!
//! Nothing in this crate may read a Polymarket price. Not the mid, not the
//! touch, not the last trade. The fair probability is computed from the
//! reference asset, its volatility, the settlement mechanics, and the clock.
//!
//! This is not a stylistic preference. A model that anchors on the venue's own
//! price cannot, even in principle, tell you that the venue's price is wrong;
//! it will converge on whatever the book says and report a comfortable,
//! useless zero edge. The type system enforces the rule: `FairModel::fair` is
//! not handed a book.
//!
//! Order book state *is* used elsewhere — by `px-edge` to compute the realisable
//! entry price, and by `px-risk` to size — but it enters after the fair value
//! exists, never before.

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

pub mod cross;
pub mod twap;
pub mod vol;

use px_core::{MarketSpec, Nanos, Prob, Settlement};

pub use cross::{CrossAsset, Nowcast, N};
pub use twap::{TwapFair, TwapInputs};
pub use vol::{EwmaVol, TwapAccumulator, TwoSpeedVol};

/// The model's verdict on a single market.
#[derive(Clone, Copy, Debug, Default)]
pub struct FairValue {
    /// Fair probability of the YES outcome.
    pub p: Prob,
    /// 1-sigma uncertainty on `p`, in probability units. The safety margin in
    /// the edge calculator is a multiple of this.
    pub sigma_p: f64,
    /// Sensitivity of `p` to a one-unit move in the underlying. The hedge ratio.
    pub delta: f64,
    /// Remaining standard deviation of settlement value, in price units, using
    /// the unbiased (precision-weighted) volatility. This is what prices.
    pub settle_sd: f64,
    /// The same quantity computed with the conservative (maximum) volatility.
    /// Used by the inventory penalty and the position limits, where a high
    /// sigma correctly means "carry less" rather than "shift the price".
    pub settle_sd_risk: f64,
    /// Seconds to expiry at evaluation time.
    pub tau: f64,
    /// Outcome determined to within f64 resolution.
    pub decided: bool,
    /// Inputs were warm and fresh enough to act on. When false the engine may
    /// still compute a value for logging, but the risk gate will refuse to
    /// trade it.
    pub usable: bool,
}

/// Live reference-asset state, shared by every market on that underlying.
///
/// Updated on the reference tick path — one update per spot print, not per
/// order book delta. On a busy BTC market the book ticks two orders of
/// magnitude more often than the reference does, and recomputing volatility on
/// each book delta would be the single largest waste in the system.
#[derive(Debug)]
pub struct RefState {
    pub vol: [TwoSpeedVol; N],
    pub spot: [f64; N],
    pub spot_ts: [f64; N],
    pub cross: CrossAsset,
    /// Nowcast-adjusted spot: the last print, plus the cross-asset correction
    /// for however long this feed has been quiet.
    pub adjusted: [f64; N],
}

impl RefState {
    pub fn new() -> Self {
        RefState {
            vol: [TwoSpeedVol::new(3.0, 180.0); N],
            spot: [0.0; N],
            spot_ts: [0.0; N],
            cross: CrossAsset::new(120.0, 0.25),
            adjusted: [0.0; N],
        }
    }

    /// Reference feed tick.
    pub fn on_print(&mut self, asset: usize, price: f64, ts_s: f64) {
        if asset >= N || !(price > 0.0) || !price.is_finite() {
            return;
        }
        self.vol[asset].update(price, ts_s);
        self.cross.observe(asset, price, ts_s);
        self.spot[asset] = price;
        self.spot_ts[asset] = ts_s;
        self.adjusted[asset] = price;
    }

    /// Advance the cross-asset sampler and refresh nowcasts for any feed that
    /// has gone quiet while its peers kept printing.
    pub fn on_clock(&mut self, ts_s: f64) {
        self.cross.tick(ts_s);

        let mut sigma = [0.0f64; N];
        for i in 0..N {
            sigma[i] = self.vol[i].slow.sigma_rel().max(1e-9);
        }
        for i in 0..N {
            if self.spot[i] <= 0.0 {
                continue;
            }
            let nc = self.cross.nowcast(i, &sigma);
            self.adjusted[i] = if nc.contributors > 0 {
                self.spot[i] * px_core::math::exp(nc.shrunk())
            } else {
                self.spot[i]
            };
        }
    }

    #[inline(always)]
    pub fn age(&self, asset: usize, now_s: f64) -> f64 {
        if asset >= N {
            return f64::INFINITY;
        }
        (now_s - self.spot_ts[asset]).max(0.0)
    }
}

impl Default for RefState {
    fn default() -> Self {
        RefState::new()
    }
}

/// Per-market model state: currently just the settlement-window integrator,
/// but this is where any future path-dependent term would live.
#[derive(Clone, Copy, Debug)]
pub struct MarketAlphaState {
    pub twap: TwapAccumulator,
}

impl MarketAlphaState {
    pub fn new(spec: &MarketSpec) -> Self {
        let expiry_s = spec.expiry.as_secs_f64();
        MarketAlphaState {
            twap: TwapAccumulator::new(expiry_s, spec.twap_window()),
        }
    }
}

/// Pluggable alpha model.
///
/// New models are registered at runtime and selected by index, so a model can
/// be added, shadowed against production flow, and promoted without restarting
/// the process or dropping the book.
pub trait FairModel: Send + Sync {
    fn fair(
        &self,
        spec: &MarketSpec,
        refs: &RefState,
        mkt: &MarketAlphaState,
        now: Nanos,
    ) -> FairValue;

    fn name(&self) -> &'static str;
}

/// The production model: TWAP-aware, cross-asset-nowcast-adjusted.
#[derive(Clone, Copy, Debug)]
pub struct TwapAwareModel {
    /// Whether to use the cross-asset nowcast in place of the raw last print.
    pub use_nowcast: bool,
}

impl Default for TwapAwareModel {
    fn default() -> Self {
        TwapAwareModel { use_nowcast: true }
    }
}

impl FairModel for TwapAwareModel {
    fn name(&self) -> &'static str {
        "twap-aware-v1"
    }

    fn fair(
        &self,
        spec: &MarketSpec,
        refs: &RefState,
        mkt: &MarketAlphaState,
        now: Nanos,
    ) -> FairValue {
        let a = spec.underlying.index();
        let now_s = now.as_secs_f64();
        let tau = spec.tau_secs(now);

        let raw = refs.spot[a];
        if raw <= 0.0 {
            return FairValue {
                tau,
                ..Default::default()
            };
        }

        let spot = if self.use_nowcast && refs.adjusted[a] > 0.0 {
            refs.adjusted[a]
        } else {
            raw
        };

        let age = refs.age(a, now_s);
        // Two volatilities, deliberately. The blend prices; the maximum sizes.
        let sigma_abs = refs.vol[a].sigma_rel() * spot;
        let sigma_abs_risk = refs.vol[a].sigma_rel_conservative() * spot;

        let window = match spec.settlement {
            Settlement::Twap { window_s } => window_s,
            Settlement::Spot => 0.0,
        };

        let inp = TwapInputs {
            spot,
            strike: spec.strike,
            sigma: sigma_abs,
            tau,
            window,
            observed_avg: mkt.twap.observed_avg(),
            sigma_rel_err: refs.vol[a].rel_err(),
            spot_age: age,
        };

        let f = twap::fair(&inp);
        let settle_sd_risk = sigma_abs_risk * twap::variance_shape(tau, window).sqrt();

        // Usability gate. Every one of these has cost someone money somewhere:
        // a cold volatility estimate, a feed that stopped, a market past its
        // expiry that the venue has not marked resolved yet.
        let usable =
            refs.vol[a].is_warm() && age < 2.0 && sigma_abs > 0.0 && tau > 0.0 && f.p.is_finite();

        FairValue {
            p: Prob::from_f64(f.p),
            sigma_p: f.sigma_p,
            delta: f.d_spot,
            settle_sd: f.sd,
            settle_sd_risk,
            tau,
            decided: f.decided,
            usable,
        }
    }
}

/// Hot-swappable set of models.
pub struct ModelRegistry {
    models: Vec<Box<dyn FairModel>>,
    active: usize,
}

/// Names only. `dyn FairModel` is not `Debug`, and requiring it would force
/// every future model to carry a formatter it will never use.
impl core::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field(
                "models",
                &self.models.iter().map(|m| m.name()).collect::<Vec<_>>(),
            )
            .field("active", &self.active)
            .finish()
    }
}

impl ModelRegistry {
    pub fn new(initial: Box<dyn FairModel>) -> Self {
        ModelRegistry {
            models: vec![initial],
            active: 0,
        }
    }

    /// Register an additional model. Returns its index. Registration does not
    /// activate it: a new model runs in shadow first.
    pub fn register(&mut self, m: Box<dyn FairModel>) -> usize {
        self.models.push(m);
        self.models.len() - 1
    }

    /// Promote a model to production. The only mutation on the hot path is a
    /// single index store.
    pub fn activate(&mut self, idx: usize) -> bool {
        if idx < self.models.len() {
            self.active = idx;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn active(&self) -> &dyn FairModel {
        self.models[self.active].as_ref()
    }

    /// Evaluate every registered model. Used by the shadow harness to compare a
    /// candidate against production on identical inputs.
    pub fn evaluate_all(
        &self,
        spec: &MarketSpec,
        refs: &RefState,
        mkt: &MarketAlphaState,
        now: Nanos,
        out: &mut Vec<(&'static str, FairValue)>,
    ) {
        out.clear();
        for m in &self.models {
            out.push((m.name(), m.fair(spec, refs, mkt, now)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_core::{Category, MarketId, Qty, TokenId, Underlying};

    fn spec() -> MarketSpec {
        MarketSpec {
            market: MarketId(1),
            yes: TokenId(1),
            no: TokenId(2),
            underlying: Underlying::Btc,
            category: Category::Crypto,
            settlement: Settlement::Twap { window_s: 60.0 },
            strike: 65_000.0,
            expiry: Nanos::from_secs_f64(300.0),
            tick: 10_000,
            min_size: Qty::shares(5),
            reward_max_spread_ticks: 3,
            reward_min_size: Qty::shares(50),
        }
    }

    fn warm_refs(final_px: f64) -> RefState {
        let mut r = RefState::new();
        let mut px = 65_000.0;
        let mut t = 0.0;
        for i in 0..2000 {
            px *= if i % 2 == 0 { 1.00005 } else { 0.99995 };
            t += 0.05;
            r.on_print(0, px, t);
            r.on_clock(t);
        }
        r.on_print(0, final_px, t + 0.05);
        r.on_clock(t + 0.05);
        r
    }

    #[test]
    fn model_never_sees_a_book() {
        // Compile-time property, asserted here in prose for the reader and in
        // the signature for the compiler: `fair` takes no book argument.
        let m = TwapAwareModel::default();
        let refs = warm_refs(65_000.0);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(100.1));
        assert!(fv.usable);
    }

    #[test]
    fn at_the_money_prices_near_fifty() {
        let m = TwapAwareModel { use_nowcast: false };
        let refs = warm_refs(65_000.0);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(100.1));
        assert!((fv.p.as_f64() - 0.5).abs() < 0.02, "p = {}", fv.p.as_f64());
        assert!(fv.delta > 0.0);
        assert!(!fv.decided);
    }

    #[test]
    fn cold_volatility_is_not_usable() {
        let m = TwapAwareModel::default();
        let mut refs = RefState::new();
        refs.on_print(0, 65_000.0, 1.0);
        refs.on_print(0, 65_010.0, 1.1);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(1.2));
        assert!(!fv.usable);
    }

    #[test]
    fn stale_feed_is_not_usable() {
        let m = TwapAwareModel::default();
        let refs = warm_refs(65_000.0);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        // Ten seconds after the last print.
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(110.0));
        assert!(!fv.usable);
    }

    #[test]
    fn expired_market_is_not_usable() {
        let m = TwapAwareModel::default();
        let refs = warm_refs(65_000.0);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(400.0));
        assert!(!fv.usable);
        assert_eq!(fv.tau, 0.0);
    }

    #[test]
    fn deep_in_the_money_near_expiry_is_decided() {
        let m = TwapAwareModel { use_nowcast: false };
        let refs = warm_refs(70_000.0);
        let mut s = spec();
        s.expiry = Nanos::from_secs_f64(100.5);
        let mut ms = MarketAlphaState::new(&s);
        // Fill the settlement window well above the strike.
        let mut t = 40.5;
        while t <= 100.4 {
            ms.twap.update(70_000.0, t);
            t += 1.0;
        }
        let fv = m.fair(&s, &refs, &ms, Nanos::from_secs_f64(100.4));
        assert!(fv.decided);
        assert!(fv.p.as_f64() > 0.999);
    }

    #[test]
    fn registry_hot_swaps_without_dropping_state() {
        let mut reg = ModelRegistry::new(Box::new(TwapAwareModel { use_nowcast: false }));
        let idx = reg.register(Box::new(TwapAwareModel { use_nowcast: true }));
        assert_eq!(reg.active().name(), "twap-aware-v1");
        assert!(reg.activate(idx));
        assert!(!reg.activate(99));

        let refs = warm_refs(65_020.0);
        let s = spec();
        let ms = MarketAlphaState::new(&s);
        let mut out = Vec::new();
        reg.evaluate_all(&s, &refs, &ms, Nanos::from_secs_f64(100.1), &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn nowcast_moves_fair_value_when_peers_lead() {
        // BTC's feed is a beat behind; the peers have all rallied. The
        // nowcast-enabled model should price higher than the raw-print model.
        let mut refs = RefState::new();
        let mut px = [65_000.0, 3_200.0, 150.0, 0.6, 0.15, 600.0, 30.0];
        let mut t = 0.0;
        let mut seed = 1u64;
        let mut nrand = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((seed >> 11) as f64) / ((1u64 << 53) as f64)) - 0.5
        };
        for _ in 0..3000 {
            let shock = 0.0004 * nrand();
            for i in 0..N {
                px[i] *= 1.0 + shock;
                refs.on_print(i, px[i], t);
            }
            t += 0.1;
            refs.on_clock(t);
        }

        // Peers move up; BTC does not print.
        for i in 1..N {
            px[i] *= 1.003;
            refs.on_print(i, px[i], t + 0.05);
        }
        refs.on_clock(t + 0.05);

        assert!(
            refs.adjusted[0] > refs.spot[0],
            "adjusted {} vs raw {}",
            refs.adjusted[0],
            refs.spot[0]
        );

        let s = MarketSpec {
            strike: refs.spot[0],
            expiry: Nanos::from_secs_f64(t + 120.0),
            ..spec()
        };
        let ms = MarketAlphaState::new(&s);
        let with = TwapAwareModel { use_nowcast: true }.fair(
            &s,
            &refs,
            &ms,
            Nanos::from_secs_f64(t + 0.06),
        );
        let without = TwapAwareModel { use_nowcast: false }.fair(
            &s,
            &refs,
            &ms,
            Nanos::from_secs_f64(t + 0.06),
        );
        assert!(
            with.p.as_f64() > without.p.as_f64(),
            "with {} without {}",
            with.p.as_f64(),
            without.p.as_f64()
        );
    }
}
