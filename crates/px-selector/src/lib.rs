//! `px-selector` — position-structure selection.
//!
//! # What this is not
//!
//! It is not a follower. It never reads another wallet, never mirrors an order,
//! never keys off observed trades by any particular counterparty. The
//! structures below are *shapes* — well-known ways to express a view given a
//! fee schedule and an inventory state. Which shape gets used, at what price,
//! in what size, is decided entirely by this bot's own fair value, its own book
//! walk, and its own risk state.
//!
//! # The ladder
//!
//! Selection is a strict priority ladder, not a scoring function. Ladders are
//! auditable: for any decision there is exactly one rule that fired, and the
//! `Reason` returned with every decision names it. When a fill looks wrong at
//! 3am, "rule 5 fired because the pair cost 96.2 cents" is a debuggable
//! sentence. "The utility function preferred it" is not.
//!
//! ```text
//!   1. feed fault / model unusable ....... Flatten (if exposed) else Idle
//!   2. risk layer has frozen us .......... Idle
//!   3. inventory at hard limit ........... Flatten
//!   4. complete sets held, capital tight . InventoryRelease
//!   5. sub-dollar pair available ......... SyntheticPair    <- model-free
//!   6. outcome decided, discount left .... NearResolution
//!   7. dislocation beyond z_enter ........ HedgedDirectional
//!   8. fair value crossed, cooldown ok ... DynamicRotation
//!   9. passive quote clears the hurdle ... TwoSidedQuote    <- the default
//!  10. otherwise ......................... Idle
//! ```
//!
//! Rule 9 is where the bot spends almost all of its time and earns almost all
//! of its money. Rules 5 through 8 are the exceptions that justify the machinery.

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

pub mod relative;

pub use relative::{price_pair, Candidate, GapTracker, PairCost, Ranker, MAX_CANDIDATES};

use px_core::{Nanos, Prob};
use px_edge::{Dir, MakeAssessment, TakeAssessment};
use px_inventory::InventoryState;

/// Ways of expressing exposure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Structure {
    /// Nothing to do. Quotes pulled.
    Idle,
    /// Rest on both sides around the inventory-adjusted working price. The
    /// default and the workhorse.
    TwoSidedQuote,
    /// Assemble a complete YES+NO set for under a dollar. The only structure
    /// whose profit does not depend on the fair-value model being right.
    SyntheticPair,
    /// A matched pair plus a deliberate net bias. The pair is risk-controlled
    /// ballast; the bias is the actual bet.
    HedgedDirectional,
    /// Rotate exposure between YES and NO as fair value crosses, under an
    /// explicit cooldown and rotation cap.
    DynamicRotation,
    /// Merge complete sets or sell a near-certain leg to free collateral for a
    /// fresher edge elsewhere.
    InventoryRelease,
    /// Buy an all-but-decided outcome at a discount, under a strict loss cap.
    NearResolution,
    /// Reduce exposure, crossing the spread if necessary.
    Flatten,
}

/// Why a structure was chosen. One byte, logged with every decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    FeedFault,
    ModelUnusable,
    RiskFrozen,
    InventoryAtLimit,
    CapitalLocked,
    FreePair,
    OutcomeDecided,
    Dislocation,
    FairValueCrossed,
    PassiveEdge,
    NoEdge,
    DwellLock,
    RotationCooldown,
    RotationCapped,
    BudgetExhausted,
}

/// Selector tunables.
#[derive(Clone, Copy, Debug)]
pub struct SelectorConfig {
    /// Minimum time in a structure before a non-safety transition is allowed.
    pub min_dwell: Nanos,
    /// Minimum time between direction rotations.
    pub rotation_cooldown: Nanos,
    /// Maximum rotations permitted over one market's lifetime.
    pub max_rotations: u32,
    /// Fair value must move at least this far past 0.5 to justify a rotation.
    /// Prevents oscillation when the model sits on the fence.
    pub rotation_band: f64,
    /// Dislocation z-score required to enter a directional structure.
    pub z_enter: f64,
    /// Dislocation z-score below which we leave it.
    pub z_exit: f64,
    /// Probability beyond which an outcome counts as near-resolved.
    pub near_resolution_p: f64,
    /// ...and the time remaining below which we will act on that.
    pub near_resolution_tau: f64,
    /// Seconds to expiry below which locked collateral should be released.
    pub capital_release_tau: f64,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        SelectorConfig {
            min_dwell: Nanos::from_millis(250),
            rotation_cooldown: Nanos::from_millis(2_000),
            max_rotations: 6,
            rotation_band: 0.04,
            z_enter: 2.5,
            z_exit: 1.0,
            near_resolution_p: 0.93,
            near_resolution_tau: 45.0,
            capital_release_tau: 30.0,
        }
    }
}

/// Everything the selector needs to decide. Built on the stack each tick.
#[derive(Clone, Copy, Debug)]
pub struct SelectorCtx {
    pub now: Nanos,
    pub tau: f64,
    pub fair: Prob,
    pub sigma_p: f64,
    /// Model inputs were warm and fresh.
    pub usable: bool,
    /// Outcome determined to within model resolution.
    pub decided: bool,
    /// Normalised dislocation from the relative-value monitor.
    pub z: f64,
    pub inventory: InventoryState,
    pub net_position: i64,
    pub matched: i64,
    pub best_take: Option<TakeAssessment>,
    pub best_make: Option<MakeAssessment>,
    pub pair: PairCost,
    /// All reference and market feeds healthy.
    pub feed_ok: bool,
    /// The rate-limit governor has tokens to spare.
    pub quote_budget_ok: bool,
}

/// The selector's output.
#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub structure: Structure,
    pub reason: Reason,
    pub changed: bool,
    pub allow_take: bool,
    pub allow_make: bool,
    /// Direction the structure wants exposure in, if any.
    pub target_dir: Option<Dir>,
}

/// The state machine.
#[derive(Clone, Copy, Debug)]
pub struct Selector {
    pub cfg: SelectorConfig,
    structure: Structure,
    entered_at: Nanos,
    last_rotation: Nanos,
    rotations: u32,
    last_dir: Option<Dir>,
    /// Count of transitions, for the whipsaw monitor.
    pub transitions: u64,
}

impl Selector {
    pub fn new(cfg: SelectorConfig) -> Self {
        Selector {
            cfg,
            structure: Structure::Idle,
            entered_at: Nanos::ZERO,
            last_rotation: Nanos::ZERO,
            rotations: 0,
            last_dir: None,
            transitions: 0,
        }
    }

    #[inline(always)]
    pub fn current(&self) -> Structure {
        self.structure
    }

    #[inline(always)]
    pub fn rotations(&self) -> u32 {
        self.rotations
    }

    #[inline(always)]
    fn dwell_satisfied(&self, now: Nanos) -> bool {
        now.since(self.entered_at).0 >= self.cfg.min_dwell.0
    }

    fn enter(&mut self, s: Structure, r: Reason, now: Nanos, dir: Option<Dir>) -> Decision {
        let changed = s != self.structure;
        if changed {
            self.structure = s;
            self.entered_at = now;
            self.transitions += 1;
        }
        if let Some(d) = dir {
            if self.last_dir != Some(d) {
                if self.last_dir.is_some() {
                    self.rotations += 1;
                    self.last_rotation = now;
                }
                self.last_dir = Some(d);
            }
        }
        Decision {
            structure: s,
            reason: r,
            changed,
            allow_take: matches!(
                s,
                Structure::SyntheticPair
                    | Structure::NearResolution
                    | Structure::Flatten
                    | Structure::HedgedDirectional
                    | Structure::DynamicRotation
                    | Structure::InventoryRelease
            ),
            allow_make: matches!(
                s,
                Structure::TwoSidedQuote
                    | Structure::HedgedDirectional
                    | Structure::DynamicRotation
            ),
            target_dir: dir,
        }
    }

    /// Evaluate the ladder. Safety rules (1 through 3) bypass the dwell lock;
    /// everything else respects it.
    pub fn evaluate(&mut self, ctx: &SelectorCtx) -> Decision {
        let now = ctx.now;
        let exposed = ctx.net_position != 0;

        // --- 1. Feed fault. Nothing else matters. ---
        if !ctx.feed_ok {
            let s = if exposed {
                Structure::Flatten
            } else {
                Structure::Idle
            };
            return self.enter(s, Reason::FeedFault, now, None);
        }

        // --- 2. Risk layer has frozen us. ---
        if ctx.inventory == InventoryState::Frozen {
            return self.enter(Structure::Idle, Reason::RiskFrozen, now, None);
        }

        // --- 3. Inventory at the hard limit. ---
        if ctx.inventory.should_flatten() {
            return self.enter(Structure::Flatten, Reason::InventoryAtLimit, now, None);
        }

        // Model unusable: we may still hold and release, but we may not open.
        if !ctx.usable {
            let s = if exposed {
                Structure::Flatten
            } else {
                Structure::Idle
            };
            return self.enter(s, Reason::ModelUnusable, now, None);
        }

        // Below here, non-safety transitions honour the dwell lock. Re-entering
        // the *same* structure is always fine — the lock exists to stop
        // oscillation between different structures, not to freeze the machine.
        let locked = !self.dwell_satisfied(now);

        // --- 4. Capital locked in complete sets, and time is short. ---
        if ctx.matched > 0 && ctx.tau < self.cfg.capital_release_tau {
            if !locked || self.structure == Structure::InventoryRelease {
                return self.enter(
                    Structure::InventoryRelease,
                    Reason::CapitalLocked,
                    now,
                    None,
                );
            }
        }

        // --- 5. Free money. Takes precedence over any model-dependent view. ---
        if ctx.pair.viable && ctx.pair.net_profit > 0 {
            return self.enter(Structure::SyntheticPair, Reason::FreePair, now, None);
        }

        let p = ctx.fair.as_f64();

        // --- 6. Near-resolution capture. ---
        if ctx.tau <= self.cfg.near_resolution_tau
            && (p >= self.cfg.near_resolution_p || p <= 1.0 - self.cfg.near_resolution_p)
        {
            if let Some(t) = ctx.best_take {
                if t.viable && (!locked || self.structure == Structure::NearResolution) {
                    let dir = if p >= 0.5 { Dir::Buy } else { Dir::Sell };
                    return self.enter(
                        Structure::NearResolution,
                        Reason::OutcomeDecided,
                        now,
                        Some(dir),
                    );
                }
            }
        }

        // --- 7. Acute dislocation. ---
        let z_threshold = if self.structure == Structure::HedgedDirectional {
            self.cfg.z_exit
        } else {
            self.cfg.z_enter
        };
        if ctx.z.abs() >= z_threshold {
            if let Some(t) = ctx.best_take {
                if t.viable && (!locked || self.structure == Structure::HedgedDirectional) {
                    let dir = if ctx.z > 0.0 { Dir::Buy } else { Dir::Sell };
                    return self.enter(
                        Structure::HedgedDirectional,
                        Reason::Dislocation,
                        now,
                        Some(dir),
                    );
                }
            }
        }

        // --- 8. Rotation, with the whipsaw guards the brief asks for. ---
        let wants = if p > 0.5 + self.cfg.rotation_band {
            Some(Dir::Buy)
        } else if p < 0.5 - self.cfg.rotation_band {
            Some(Dir::Sell)
        } else {
            None
        };
        if let Some(want) = wants {
            let flipping = self.last_dir.is_some() && self.last_dir != Some(want);
            if flipping {
                if self.rotations >= self.cfg.max_rotations {
                    return self.enter(Structure::TwoSidedQuote, Reason::RotationCapped, now, None);
                }
                if now.since(self.last_rotation).0 < self.cfg.rotation_cooldown.0 {
                    return self.enter(
                        Structure::TwoSidedQuote,
                        Reason::RotationCooldown,
                        now,
                        None,
                    );
                }
                if locked {
                    return self.enter(self.structure, Reason::DwellLock, now, None);
                }
                return self.enter(
                    Structure::DynamicRotation,
                    Reason::FairValueCrossed,
                    now,
                    Some(want),
                );
            }
        }

        // --- 9. The default: rest a two-sided quote. ---
        if !ctx.quote_budget_ok {
            return self.enter(Structure::Idle, Reason::BudgetExhausted, now, None);
        }
        if let Some(m) = ctx.best_make {
            if m.viable && ctx.inventory.may_quote() {
                return self.enter(Structure::TwoSidedQuote, Reason::PassiveEdge, now, None);
            }
        }

        // --- 10. Nothing worth doing. ---
        self.enter(Structure::Idle, Reason::NoEdge, now, None)
    }

    /// Reset per-market counters when a new market is subscribed.
    pub fn reset(&mut self, now: Nanos) {
        self.structure = Structure::Idle;
        self.entered_at = now;
        self.last_rotation = Nanos::ZERO;
        self.rotations = 0;
        self.last_dir = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_core::{Px, Qty};

    fn take(viable: bool) -> TakeAssessment {
        TakeAssessment {
            qty: Qty::shares(100),
            avg_entry: Px(520_000),
            worst_px: Px(520_000),
            gross_per_share: 40_000,
            fee_per_share: 17_000,
            margin_per_share: 1_000,
            net_per_share: 22_000,
            total_edge: px_core::Usd::dollars(22),
            levels: 1,
            viable,
        }
    }

    fn make(viable: bool) -> MakeAssessment {
        MakeAssessment {
            price: Px(490_000),
            qty: Qty::shares(100),
            distance_ticks: 1,
            queue_ahead: Qty::shares(20),
            gross_per_share: 8_000,
            rebate_per_share: 3_500,
            reward_per_share: 500,
            adverse_per_share: 2_000,
            risk_credit_per_share: 0,
            margin_per_share: 1_000,
            net_per_share: 9_000,
            viable,
        }
    }

    fn ctx(now_ms: u64) -> SelectorCtx {
        SelectorCtx {
            now: Nanos::from_millis(now_ms),
            tau: 200.0,
            fair: Prob::from_f64(0.5),
            sigma_p: 0.005,
            usable: true,
            decided: false,
            z: 0.0,
            inventory: InventoryState::Flat,
            net_position: 0,
            matched: 0,
            best_take: Some(take(true)),
            best_make: Some(make(true)),
            pair: PairCost::default(),
            feed_ok: true,
            quote_budget_ok: true,
        }
    }

    #[test]
    fn default_behaviour_is_to_rest_a_two_sided_quote() {
        let mut s = Selector::new(SelectorConfig::default());
        let d = s.evaluate(&ctx(1000));
        assert_eq!(d.structure, Structure::TwoSidedQuote);
        assert_eq!(d.reason, Reason::PassiveEdge);
        assert!(d.allow_make);
        assert!(!d.allow_take);
    }

    #[test]
    fn feed_fault_beats_every_opportunity() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.feed_ok = false;
        c.z = 9.0;
        c.pair = PairCost {
            net_profit: 50_000,
            viable: true,
            ..Default::default()
        };
        c.net_position = Qty::shares(100).0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Flatten);
        assert_eq!(d.reason, Reason::FeedFault);
    }

    #[test]
    fn feed_fault_with_no_position_just_idles() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.feed_ok = false;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Idle);
    }

    #[test]
    fn frozen_risk_state_stops_everything() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.inventory = InventoryState::Frozen;
        c.z = 9.0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Idle);
        assert_eq!(d.reason, Reason::RiskFrozen);
    }

    #[test]
    fn inventory_at_limit_forces_a_flatten() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.inventory = InventoryState::Flattening;
        c.z = 9.0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Flatten);
        assert_eq!(d.reason, Reason::InventoryAtLimit);
        assert!(d.allow_take);
    }

    #[test]
    fn unusable_model_blocks_opening_but_permits_closing() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.usable = false;
        c.net_position = Qty::shares(200).0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Flatten);
        assert_eq!(d.reason, Reason::ModelUnusable);
    }

    #[test]
    fn a_free_pair_outranks_a_model_driven_view() {
        // Rule 5 sits above rule 7 precisely because its profit does not depend
        // on the model being right.
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.z = 9.0;
        c.pair = PairCost {
            net_profit: 12_000,
            viable: true,
            qty: Qty::shares(300),
            ..Default::default()
        };
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::SyntheticPair);
        assert_eq!(d.reason, Reason::FreePair);
    }

    #[test]
    fn near_resolution_fires_only_when_both_price_and_time_qualify() {
        let mut s = Selector::new(SelectorConfig::default());

        // Extreme price but plenty of time: not yet.
        let mut c = ctx(1000);
        c.fair = Prob::from_f64(0.97);
        c.tau = 200.0;
        assert_ne!(s.evaluate(&c).structure, Structure::NearResolution);

        // Extreme price and little time: yes.
        let mut c2 = ctx(5000);
        c2.fair = Prob::from_f64(0.97);
        c2.tau = 20.0;
        let d = s.evaluate(&c2);
        assert_eq!(d.structure, Structure::NearResolution);
        assert_eq!(d.target_dir, Some(Dir::Buy));
    }

    #[test]
    fn near_resolution_on_the_no_side_sells() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.fair = Prob::from_f64(0.03);
        c.tau = 20.0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::NearResolution);
        assert_eq!(d.target_dir, Some(Dir::Sell));
    }

    #[test]
    fn dislocation_enters_hedged_directional() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.z = 4.0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::HedgedDirectional);
        assert_eq!(d.reason, Reason::Dislocation);
        assert_eq!(d.target_dir, Some(Dir::Buy));
    }

    #[test]
    fn dislocation_uses_a_lower_threshold_to_exit_than_to_enter() {
        // Enter at z >= 2.5, stay until z < 1.0. Without the asymmetry the
        // structure would flicker whenever z hovered near the entry level.
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.z = 3.0;
        assert_eq!(s.evaluate(&c).structure, Structure::HedgedDirectional);

        let mut c2 = ctx(2000);
        c2.z = 1.5; // below entry, above exit
        assert_eq!(s.evaluate(&c2).structure, Structure::HedgedDirectional);

        let mut c3 = ctx(3000);
        c3.z = 0.5; // below exit
        assert_ne!(s.evaluate(&c3).structure, Structure::HedgedDirectional);
    }

    #[test]
    fn rotation_respects_the_cooldown() {
        let mut s = Selector::new(SelectorConfig::default());

        // Establish a long bias via a dislocation.
        let mut c = ctx(1000);
        c.z = 4.0;
        s.evaluate(&c);
        assert_eq!(s.rotations(), 0);

        // Fair value flips hard the other way, 300 ms later. Cooldown is 2 s.
        let mut c2 = ctx(1300);
        c2.z = 0.0;
        c2.fair = Prob::from_f64(0.30);
        let d = s.evaluate(&c2);
        assert_eq!(d.reason, Reason::RotationCooldown);
        assert_ne!(d.structure, Structure::DynamicRotation);
    }

    #[test]
    fn rotation_proceeds_once_the_cooldown_has_elapsed() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.z = 4.0;
        s.evaluate(&c);

        let mut c2 = ctx(9000); // well past the 2 s cooldown
        c2.z = 0.0;
        c2.fair = Prob::from_f64(0.30);
        let d = s.evaluate(&c2);
        assert_eq!(d.structure, Structure::DynamicRotation);
        assert_eq!(d.reason, Reason::FairValueCrossed);
        assert_eq!(d.target_dir, Some(Dir::Sell));
        assert_eq!(s.rotations(), 1);
    }

    #[test]
    fn rotation_is_capped_over_a_market_lifetime() {
        let cfg = SelectorConfig {
            max_rotations: 2,
            rotation_cooldown: Nanos::from_millis(1),
            min_dwell: Nanos::from_millis(1),
            ..Default::default()
        };
        let mut s = Selector::new(cfg);

        let mut t = 1000u64;
        let mut c = ctx(t);
        c.z = 4.0;
        s.evaluate(&c); // establishes Buy

        for i in 0..8 {
            t += 500;
            let mut c2 = ctx(t);
            c2.z = 0.0;
            c2.fair = Prob::from_f64(if i % 2 == 0 { 0.30 } else { 0.70 });
            s.evaluate(&c2);
        }
        assert!(s.rotations() <= 2, "rotations = {}", s.rotations());

        // And once capped, further flips fall back to plain quoting.
        t += 500;
        let mut c3 = ctx(t);
        c3.z = 0.0;
        c3.fair = Prob::from_f64(0.20);
        let d = s.evaluate(&c3);
        assert_eq!(d.reason, Reason::RotationCapped);
    }

    #[test]
    fn a_fair_value_hovering_on_the_fence_never_rotates() {
        // The classic whipsaw: p oscillating around 0.5. The rotation band
        // means neither side is ever requested.
        let cfg = SelectorConfig {
            rotation_cooldown: Nanos::ZERO,
            min_dwell: Nanos::ZERO,
            ..Default::default()
        };
        let mut s = Selector::new(cfg);
        for i in 0..200u64 {
            let mut c = ctx(1000 + i * 10);
            c.fair = Prob::from_f64(if i % 2 == 0 { 0.51 } else { 0.49 });
            let d = s.evaluate(&c);
            assert_ne!(d.structure, Structure::DynamicRotation);
        }
        assert_eq!(s.rotations(), 0);
    }

    #[test]
    fn locked_capital_is_released_near_expiry() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.matched = Qty::shares(500).0;
        c.tau = 10.0;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::InventoryRelease);
        assert_eq!(d.reason, Reason::CapitalLocked);
    }

    #[test]
    fn locked_capital_is_left_alone_when_there_is_time() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.matched = Qty::shares(500).0;
        c.tau = 200.0;
        assert_ne!(s.evaluate(&c).structure, Structure::InventoryRelease);
    }

    #[test]
    fn exhausted_quote_budget_stops_quoting_rather_than_queueing() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.quote_budget_ok = false;
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Idle);
        assert_eq!(d.reason, Reason::BudgetExhausted);
    }

    #[test]
    fn no_viable_quote_means_idle_not_a_worse_quote() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.best_make = Some(make(false));
        c.best_take = Some(take(false));
        let d = s.evaluate(&c);
        assert_eq!(d.structure, Structure::Idle);
        assert_eq!(d.reason, Reason::NoEdge);
    }

    #[test]
    fn reset_clears_per_market_history() {
        let mut s = Selector::new(SelectorConfig::default());
        let mut c = ctx(1000);
        c.z = 4.0;
        s.evaluate(&c);
        s.reset(Nanos::from_millis(2000));
        assert_eq!(s.current(), Structure::Idle);
        assert_eq!(s.rotations(), 0);
    }

    #[test]
    fn transitions_stay_bounded_under_a_noisy_feed() {
        // Whipsaw regression test: drive the selector with a rapidly oscillating
        // z-score and confirm the dwell lock keeps the transition count far
        // below the tick count.
        let mut s = Selector::new(SelectorConfig::default());
        let mut t = 1000u64;
        for i in 0..2000u64 {
            t += 5; // 5 ms apart; dwell is 250 ms
            let mut c = ctx(t);
            c.z = if i % 2 == 0 { 3.0 } else { 0.2 };
            s.evaluate(&c);
        }
        // 2000 ticks over 10 s, dwell 250 ms: at most ~40 legitimate flips.
        assert!(s.transitions < 100, "transitions = {}", s.transitions);
    }
}
