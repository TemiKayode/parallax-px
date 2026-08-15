//! `px-inventory` — inventory-adaptive quoting.
//!
//! # The reservation price
//!
//! The brief specifies
//!
//! ```text
//!   Working Price = Fair Value - Inventory Penalty
//!   Penalty       = q * lambda * sigma^2 * tau
//! ```
//!
//! which is the Avellaneda–Stoikov reservation price. The economics are worth
//! restating because they are the reason the bot survives a trend: a market
//! maker who is already long is not indifferent to buying more. The next unit
//! of the same exposure is worth strictly less to them than the last, because
//! it adds variance to a book that already carries variance. Quoting off fair
//! value regardless of position is how a maker ends up holding the entire
//! wrong side of a move.
//!
//! Skewing the *working price* rather than widening the spread is the important
//! detail. Widening makes us less competitive on both sides equally and forfeits
//! the liquidity reward on both. Skewing keeps us aggressive on the side that
//! reduces risk and passive on the side that adds it — we stay in the reward
//! programme while actively recruiting the flow we want.
//!
//! # A note on double-counting time
//!
//! `sigma^2 * tau` is total remaining variance. For a TWAP-settled contract
//! that quantity does *not* decay linearly in `tau` — see `px_alpha::twap` for
//! the cubic collapse. Passing a per-second variance rate together with `tau`
//! would be wrong twice over: once for the shape, once because the model
//! already knows the answer. `PenaltyEngine::from_model` therefore takes the
//! remaining variance directly from the fair-value computation, and the raw
//! `penalty` entry point exists only for spot-settled markets and for tests.
//!
//! # Complete sets
//!
//! Holding one YES and one NO of the same market is riskless: the pair merges
//! back into one dollar of collateral. So the position that matters for risk is
//! the *net*, and the matched portion is not exposure — it is capital waiting to
//! be released. The state machine distinguishes them, and `Flattening` prefers
//! merging a matched pair over paying a taker fee to sell a naked leg.

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

use px_core::{Px, Qty};

/// Which outcome token a fill landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Leg {
    Yes,
    No,
}

/// Signed position in one market, in YES-equivalent shares.
///
/// Buying NO is recorded as negative YES: economically identical, and it means
/// the whole risk calculation is one signed number instead of two that must be
/// kept in sync.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Position {
    pub yes: Qty,
    pub no: Qty,
    /// Cash spent (positive) or received (negative), micro-dollars.
    pub cost_basis: i64,
}

impl Position {
    /// Directional exposure: what actually carries risk.
    #[inline(always)]
    pub fn net(&self) -> i64 {
        self.yes.0 - self.no.0
    }

    /// Shares held as complete YES+NO sets. Riskless, and mergeable to release
    /// one dollar of collateral per set.
    #[inline(always)]
    pub fn matched(&self) -> Qty {
        Qty(self.yes.0.min(self.no.0))
    }

    /// Collateral currently locked in matched sets, micro-dollars.
    #[inline(always)]
    pub fn releasable(&self) -> i64 {
        self.matched().0
    }

    #[inline(always)]
    pub fn gross(&self) -> i64 {
        self.yes.0 + self.no.0
    }

    #[inline(always)]
    pub fn is_flat(&self) -> bool {
        self.net() == 0
    }

    /// Apply a fill on a named leg.
    ///
    /// # Why the signed API is not enough
    ///
    /// "Sell YES" and "buy NO" both reduce net YES exposure by the same amount,
    /// and are often described as equivalent. On this venue they are *not* the
    /// same operation:
    ///
    /// * Selling YES gives up a token you hold and returns cash.
    /// * Buying NO spends cash and acquires a second token, which — held
    ///   alongside a YES — forms a **complete set** that merges back to one
    ///   dollar of collateral.
    ///
    /// Only the second creates the mergeable pair that `matched()` reports and
    /// `InventoryRelease` acts on. A signed quantity cannot express the
    /// difference, so anything that cares about complete sets must use this.
    ///
    /// `px` is the price of the leg being traded, not of YES.
    pub fn apply_leg(&mut self, leg: Leg, buy: bool, qty: Qty, px: Px) {
        let px = px.clamp_unit();
        if qty.0 <= 0 {
            return;
        }
        let cash = Self::cash(qty.0, px.0);
        match (leg, buy) {
            (Leg::Yes, true) => {
                self.yes = Qty(self.yes.0 + qty.0);
                self.cost_basis += cash;
            }
            (Leg::Yes, false) => {
                // Cannot sell tokens we do not hold; the venue would reject it.
                let sold = self.yes.0.min(qty.0);
                self.yes = Qty(self.yes.0 - sold);
                self.cost_basis -= Self::cash(sold, px.0);
            }
            (Leg::No, true) => {
                self.no = Qty(self.no.0 + qty.0);
                self.cost_basis += cash;
            }
            (Leg::No, false) => {
                let sold = self.no.0.min(qty.0);
                self.no = Qty(self.no.0 - sold);
                self.cost_basis -= Self::cash(sold, px.0);
            }
        }
    }

    /// Apply a fill. `signed_qty` is positive for acquiring YES exposure and
    /// negative for shedding it.
    ///
    /// # Reduce before you acquire
    ///
    /// The first version routed every negative fill into the NO leg, on the
    /// reasoning that buying NO is economically identical to selling YES. The
    /// *net* was right, but nothing else was: selling 100 YES you already hold
    /// was recorded as holding 100 YES **and** 100 NO. `matched()` then reported
    /// 100 riskless complete sets that did not exist, `gross()` doubled, and
    /// `free_capital_available()` promised collateral the venue would not
    /// release. The selector consumes all three.
    ///
    /// A sell reduces the existing leg first and only opens the opposite leg
    /// with whatever is left over.
    ///
    /// Arithmetic is in `i128`. The old form computed `signed_qty * px` in
    /// `i64`, which overflows above roughly nine million shares — reachable on
    /// a large book, and silent when it happens.
    pub fn apply_fill(&mut self, signed_qty: i64, px: Px) {
        let px = px.clamp_unit();
        if signed_qty == 0 {
            return;
        }
        if signed_qty > 0 {
            // Acquiring YES: first retire any NO we hold, then add YES.
            let retire = self.no.0.min(signed_qty);
            self.no = Qty(self.no.0 - retire);
            let open = signed_qty - retire;
            self.yes = Qty(self.yes.0 + open);
            // Retiring a NO at price p costs (1 - p); opening YES costs p.
            self.cost_basis += Self::cash(retire, 1_000_000 - px.0);
            self.cost_basis += Self::cash(open, px.0);
        } else {
            let q = -signed_qty;
            let retire = self.yes.0.min(q);
            self.yes = Qty(self.yes.0 - retire);
            let open = q - retire;
            self.no = Qty(self.no.0 + open);
            // Selling YES at p returns p; opening NO costs (1 - p).
            self.cost_basis -= Self::cash(retire, px.0);
            self.cost_basis += Self::cash(open, 1_000_000 - px.0);
        }
    }

    /// `qty_micro * price_micro / 1e6`, in i128 so it cannot wrap.
    #[inline(always)]
    fn cash(qty_micro: i64, px_micro: i32) -> i64 {
        ((qty_micro as i128 * px_micro as i128) / 1_000_000) as i64
    }

    /// Merge matched sets back to collateral, removing them from the book.
    pub fn merge_sets(&mut self) -> Qty {
        let m = self.matched();
        self.yes = Qty(self.yes.0 - m.0);
        self.no = Qty(self.no.0 - m.0);
        self.cost_basis -= m.0;
        m
    }
}

/// Where the book stands, and therefore what the quoting engine is allowed to
/// do next.
///
/// ```text
///                       |net| grows  -->
///   Flat --> Balanced --> Skewed --> Reducing --> Flattening
///     <--       <--         <--         <--
///                       hysteresis: exit at 70% of entry
///
///   any state --(risk gate / feed fault)--> Frozen --(all clear)--> Flat
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InventoryState {
    /// No position. Quote both sides at full size.
    Flat,
    /// Small imbalance. Quote both sides, mild skew.
    Balanced,
    /// Meaningful imbalance. Skew hard; reduce size on the adding side.
    Skewed,
    /// Large imbalance. Stop quoting the adding side entirely.
    Reducing,
    /// At the limit. Pay the taker fee to get flat — the fee is cheaper than
    /// the tail.
    Flattening,
    /// Cancel-only. Set by the risk layer, cleared only by the risk layer.
    Frozen,
}

impl InventoryState {
    /// May we post a quote that would *increase* net exposure?
    #[inline(always)]
    pub fn may_add(&self) -> bool {
        matches!(
            self,
            InventoryState::Flat | InventoryState::Balanced | InventoryState::Skewed
        )
    }

    /// May we post any quote at all?
    #[inline(always)]
    pub fn may_quote(&self) -> bool {
        !matches!(self, InventoryState::Frozen)
    }

    /// Should we cross the spread to reduce?
    #[inline(always)]
    pub fn should_flatten(&self) -> bool {
        matches!(self, InventoryState::Flattening)
    }
}

/// Position thresholds, in shares, at which the state machine advances.
#[derive(Clone, Copy, Debug)]
pub struct InventoryLimits {
    pub balanced: i64,
    pub skewed: i64,
    pub reducing: i64,
    pub flattening: i64,
    /// Exit threshold as a fraction of the entry threshold. Below 1.0 this
    /// creates the hysteresis band that stops the machine oscillating on a
    /// position sitting exactly on a boundary.
    pub hysteresis: f64,
}

impl Default for InventoryLimits {
    fn default() -> Self {
        InventoryLimits {
            balanced: Qty::shares(50).0,
            skewed: Qty::shares(200).0,
            reducing: Qty::shares(500).0,
            flattening: Qty::shares(1000).0,
            hysteresis: 0.7,
        }
    }
}

/// The inventory penalty engine and its state machine.
#[derive(Clone, Copy, Debug)]
pub struct InventoryEngine {
    pub position: Position,
    pub state: InventoryState,
    pub limits: InventoryLimits,
    /// Risk aversion. Higher means a given position skews quotes further.
    pub lambda: f64,
    /// Number of state transitions so far — a whipsaw counter. If this climbs
    /// faster than the position changes, the thresholds are miscalibrated.
    pub transitions: u64,
}

impl InventoryEngine {
    pub fn new(lambda: f64, limits: InventoryLimits) -> Self {
        InventoryEngine {
            position: Position::default(),
            state: InventoryState::Flat,
            limits,
            lambda,
            transitions: 0,
        }
    }

    /// Raw penalty, micro-dollars per share.
    ///
    /// `var_rate` is the per-second variance of the *contract price* (in
    /// probability units), `tau` is seconds remaining. Only correct for
    /// spot-settled markets; TWAP markets must use `penalty_from_model`.
    #[inline(always)]
    pub fn penalty(&self, var_rate: f64, tau: f64) -> f64 {
        let q = self.position.net() as f64 / 1e6; // shares
        q * self.lambda * var_rate * tau.max(0.0) * 1e6
    }

    /// Penalty using the model's own remaining-variance estimate.
    ///
    /// `delta` is d(probability)/d(underlying) and `settle_sd` is the remaining
    /// standard deviation of the settlement value, both straight out of
    /// `px_alpha::FairValue`. Their product is the standard deviation of the
    /// contract price between now and expiry, which already carries the correct
    /// (cubic, for TWAP) time shape.
    #[inline(always)]
    pub fn penalty_from_model(&self, delta: f64, settle_sd: f64) -> f64 {
        let q = self.position.net() as f64 / 1e6;
        let sd_price = (delta * settle_sd).abs();
        q * self.lambda * sd_price * sd_price * 1e6
    }

    /// Fair value adjusted for what we are already carrying.
    #[inline]
    pub fn working_price(&self, fair: Px, delta: f64, settle_sd: f64) -> Px {
        let p = self.penalty_from_model(delta, settle_sd);
        Px(fair.0.saturating_sub(p as i32)).clamp_unit()
    }

    /// Two-sided quote centred on the working price.
    ///
    /// Returns `(bid, ask)` rounded onto the tick grid in the safe direction.
    /// `half_spread` is in micro-dollars.
    pub fn quote(
        &self,
        fair: Px,
        delta: f64,
        settle_sd: f64,
        half_spread: i32,
        tick: i32,
    ) -> (Px, Px) {
        let w = self.working_price(fair, delta, settle_sd);
        let bid = Px(w.0 - half_spread).clamp_unit().floor_to_tick(tick);
        let ask = Px(w.0 + half_spread).clamp_unit().ceil_to_tick(tick);
        (bid, ask)
    }

    /// Record a fill and re-evaluate the state machine.
    pub fn on_fill(&mut self, signed_qty: i64, px: Px) {
        self.position.apply_fill(signed_qty, px);
        self.reevaluate();
    }

    /// Advance the state machine from the current position, with hysteresis.
    pub fn reevaluate(&mut self) {
        if self.state == InventoryState::Frozen {
            return;
        }
        let a = self.position.net().unsigned_abs() as i64;
        let l = &self.limits;
        let h = l.hysteresis;

        // Entering a more constrained state uses the raw threshold; leaving it
        // requires falling to `hysteresis` times that threshold. Without this
        // band, a position parked on a boundary flips state on every fill and
        // the quoting engine cancels and replaces forever, burning the
        // rate-limit budget that the whole system depends on.
        let next = match self.state {
            InventoryState::Flat | InventoryState::Balanced => {
                if a >= l.flattening {
                    InventoryState::Flattening
                } else if a >= l.reducing {
                    InventoryState::Reducing
                } else if a >= l.skewed {
                    InventoryState::Skewed
                } else if a >= l.balanced {
                    InventoryState::Balanced
                } else if a == 0 {
                    InventoryState::Flat
                } else {
                    InventoryState::Balanced
                }
            }
            InventoryState::Skewed => {
                if a >= l.flattening {
                    InventoryState::Flattening
                } else if a >= l.reducing {
                    InventoryState::Reducing
                } else if (a as f64) < (l.skewed as f64) * h {
                    if a == 0 {
                        InventoryState::Flat
                    } else {
                        InventoryState::Balanced
                    }
                } else {
                    InventoryState::Skewed
                }
            }
            InventoryState::Reducing => {
                if a >= l.flattening {
                    InventoryState::Flattening
                } else if (a as f64) < (l.reducing as f64) * h {
                    InventoryState::Skewed
                } else {
                    InventoryState::Reducing
                }
            }
            InventoryState::Flattening => {
                if (a as f64) < (l.flattening as f64) * h {
                    InventoryState::Reducing
                } else {
                    InventoryState::Flattening
                }
            }
            InventoryState::Frozen => InventoryState::Frozen,
        };

        if next != self.state {
            self.state = next;
            self.transitions += 1;
        }
    }

    /// Trip the kill switch. Only the risk layer calls this.
    pub fn freeze(&mut self) {
        if self.state != InventoryState::Frozen {
            self.state = InventoryState::Frozen;
            self.transitions += 1;
        }
    }

    /// Release the kill switch and recompute from the position.
    pub fn thaw(&mut self) {
        if self.state == InventoryState::Frozen {
            self.state = if self.position.is_flat() {
                InventoryState::Flat
            } else {
                InventoryState::Balanced
            };
            self.transitions += 1;
            self.reevaluate();
        }
    }

    /// Size multiplier for a quote on the given side, in `[0, 1]`.
    ///
    /// `adds_exposure` is true when a fill on this side would move `|net|`
    /// further from zero.
    pub fn size_factor(&self, adds_exposure: bool) -> f64 {
        match self.state {
            InventoryState::Frozen => 0.0,
            InventoryState::Flat => 1.0,
            InventoryState::Balanced => {
                if adds_exposure {
                    0.8
                } else {
                    1.0
                }
            }
            InventoryState::Skewed => {
                if adds_exposure {
                    0.4
                } else {
                    1.2
                }
            }
            InventoryState::Reducing => {
                if adds_exposure {
                    0.0
                } else {
                    1.5
                }
            }
            InventoryState::Flattening => {
                if adds_exposure {
                    0.0
                } else {
                    2.0
                }
            }
        }
    }

    /// Capital that can be freed right now by merging complete sets, without
    /// touching the market or paying a fee.
    ///
    /// Checked before any flattening trade: releasing collateral from a matched
    /// pair costs nothing, whereas selling a naked leg costs the taker fee.
    #[inline(always)]
    pub fn free_capital_available(&self) -> i64 {
        self.position.releasable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> InventoryEngine {
        InventoryEngine::new(0.5, InventoryLimits::default())
    }

    #[test]
    fn flat_book_has_no_penalty_and_quotes_symmetrically() {
        let e = engine();
        assert_eq!(e.penalty_from_model(0.01, 100.0), 0.0);
        let (bid, ask) = e.quote(Px(500_000), 0.01, 100.0, 20_000, 10_000);
        assert_eq!(bid, Px(480_000));
        assert_eq!(ask, Px(520_000));
        assert_eq!(e.state, InventoryState::Flat);
    }

    #[test]
    fn a_long_book_lowers_both_quotes() {
        // The core behaviour the brief asks for: already long YES means less
        // willing to buy more YES, and keener to sell.
        let mut e = engine();
        e.on_fill(Qty::shares(300).0, Px(500_000));
        assert!(e.position.net() > 0);

        let flat = engine();
        let (fb, fa) = flat.quote(Px(500_000), 0.01, 100.0, 20_000, 10_000);
        let (lb, la) = e.quote(Px(500_000), 0.01, 100.0, 20_000, 10_000);

        assert!(lb < fb, "long bid {lb:?} should be below flat bid {fb:?}");
        assert!(la < fa, "long ask {la:?} should be below flat ask {fa:?}");
    }

    #[test]
    fn a_short_book_raises_both_quotes() {
        let mut e = engine();
        e.on_fill(-Qty::shares(300).0, Px(500_000));
        assert!(e.position.net() < 0);
        let (b, a) = e.quote(Px(500_000), 0.01, 100.0, 20_000, 10_000);
        let flat = engine();
        let (fb, fa) = flat.quote(Px(500_000), 0.01, 100.0, 20_000, 10_000);
        assert!(b > fb);
        assert!(a > fa);
    }

    #[test]
    fn penalty_scales_linearly_in_position_and_quadratically_in_sd() {
        let mut e = engine();
        e.on_fill(Qty::shares(100).0, Px(500_000));
        let p1 = e.penalty_from_model(0.01, 100.0);
        e.on_fill(Qty::shares(100).0, Px(500_000));
        let p2 = e.penalty_from_model(0.01, 100.0);
        assert!((p2 / p1 - 2.0).abs() < 1e-9);

        let double_sd = e.penalty_from_model(0.01, 200.0);
        assert!((double_sd / p2 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn penalty_vanishes_as_the_twap_window_closes() {
        // Because settle_sd collapses cubically, the penalty for holding a
        // position through the last seconds of a TWAP market collapses with it.
        // A position that is nearly certain is nearly riskless, and the engine
        // should stop paying to reduce it.
        let mut e = engine();
        e.on_fill(Qty::shares(500).0, Px(700_000));
        let early = e.penalty_from_model(0.01, 100.0);
        let late = e.penalty_from_model(0.01, 6.8); // sd at 10s of a 60s window
        assert!(late / early < 0.01, "ratio {}", late / early);
    }

    #[test]
    fn state_machine_advances_through_every_stage() {
        let mut e = engine();
        assert_eq!(e.state, InventoryState::Flat);
        e.on_fill(Qty::shares(60).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Balanced);
        e.on_fill(Qty::shares(150).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Skewed);
        e.on_fill(Qty::shares(300).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Reducing);
        e.on_fill(Qty::shares(500).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Flattening);
    }

    #[test]
    fn hysteresis_prevents_whipsaw_on_a_boundary() {
        let mut e = engine();
        // Park exactly on the `skewed` boundary.
        e.on_fill(Qty::shares(200).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Skewed);
        let t0 = e.transitions;

        // Oscillate by one share, fifty times. Without hysteresis this would
        // produce 100 transitions and 100 cancel/replace cycles.
        for _ in 0..50 {
            e.on_fill(-Qty::shares(1).0, Px(500_000));
            e.on_fill(Qty::shares(1).0, Px(500_000));
        }
        assert_eq!(
            e.transitions,
            t0,
            "state churned {} times",
            e.transitions - t0
        );
        assert_eq!(e.state, InventoryState::Skewed);
    }

    #[test]
    fn hysteresis_still_allows_genuine_de_escalation() {
        let mut e = engine();
        e.on_fill(Qty::shares(250).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Skewed);
        // Drop below 70% of the 200-share threshold.
        e.on_fill(-Qty::shares(120).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Balanced);
    }

    #[test]
    fn reducing_state_refuses_to_add_exposure() {
        let mut e = engine();
        e.on_fill(Qty::shares(600).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Reducing);
        assert!(!e.state.may_add());
        assert_eq!(e.size_factor(true), 0.0);
        assert!(e.size_factor(false) > 1.0);
        assert!(e.state.may_quote());
    }

    #[test]
    fn flattening_state_authorises_crossing() {
        let mut e = engine();
        e.on_fill(Qty::shares(1200).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Flattening);
        assert!(e.state.should_flatten());
        assert_eq!(e.size_factor(true), 0.0);
    }

    #[test]
    fn freeze_overrides_everything_and_thaw_restores() {
        let mut e = engine();
        e.on_fill(Qty::shares(300).0, Px(500_000));
        e.freeze();
        assert_eq!(e.state, InventoryState::Frozen);
        assert!(!e.state.may_quote());
        assert_eq!(e.size_factor(false), 0.0);
        // A fill while frozen must not silently un-freeze us.
        e.on_fill(Qty::shares(10).0, Px(500_000));
        assert_eq!(e.state, InventoryState::Frozen);
        e.thaw();
        assert_eq!(e.state, InventoryState::Skewed);
    }

    #[test]
    fn matched_sets_are_not_exposure() {
        // A genuine complete set requires *buying* both legs.
        let mut p = Position::default();
        p.apply_leg(Leg::Yes, true, Qty::shares(100), Px(400_000)); // 100 YES @ 40c
        p.apply_leg(Leg::No, true, Qty::shares(100), Px(600_000)); // 100 NO  @ 60c
        assert_eq!(p.net(), 0);
        assert_eq!(p.matched(), Qty::shares(100));
        assert_eq!(p.gross(), Qty::shares(200).0);
        // Paid 40c + 60c = $1.00 per pair for something worth exactly $1.
        assert_eq!(p.cost_basis, Qty::shares(100).0);
        assert!(p.is_flat());
    }

    #[test]
    fn selling_yes_is_not_the_same_as_buying_no() {
        // Regression for the accounting bug: routing every sell into the NO leg
        // conjured complete sets that did not exist, and with them collateral
        // the venue would never release.
        let mut sold = Position::default();
        sold.apply_leg(Leg::Yes, true, Qty::shares(100), Px(400_000));
        sold.apply_fill(-Qty::shares(100).0, Px(450_000)); // sell the YES back

        let mut bought = Position::default();
        bought.apply_leg(Leg::Yes, true, Qty::shares(100), Px(400_000));
        bought.apply_leg(Leg::No, true, Qty::shares(100), Px(550_000));

        // Identical net exposure...
        assert_eq!(sold.net(), 0);
        assert_eq!(bought.net(), 0);
        // ...but only one of them holds anything.
        assert_eq!(sold.matched(), Qty::ZERO);
        assert_eq!(sold.gross(), 0);
        assert_eq!(bought.matched(), Qty::shares(100));
        assert_eq!(bought.gross(), Qty::shares(200).0);

        // And the realised cash differs: sold at 45c on a 40c basis is +$5.
        assert_eq!(sold.cost_basis, -px_core::Usd::dollars(5).0);
    }

    #[test]
    fn a_sell_reduces_before_it_opens_the_opposite_leg() {
        let mut p = Position::default();
        p.apply_leg(Leg::Yes, true, Qty::shares(60), Px(500_000));
        // Shed 100 of YES exposure while holding only 60.
        p.apply_fill(-Qty::shares(100).0, Px(500_000));
        assert_eq!(p.yes, Qty::ZERO);
        assert_eq!(p.no, Qty::shares(40));
        assert_eq!(p.net(), -Qty::shares(40).0);
        assert_eq!(p.matched(), Qty::ZERO);
    }

    #[test]
    fn cannot_sell_tokens_not_held() {
        let mut p = Position::default();
        p.apply_leg(Leg::Yes, false, Qty::shares(100), Px(500_000));
        assert_eq!(p.yes, Qty::ZERO);
        assert_eq!(p.cost_basis, 0);
    }

    #[test]
    fn position_arithmetic_does_not_overflow_at_scale() {
        // The old i64 form wrapped above ~9M shares.
        let mut p = Position::default();
        p.apply_leg(Leg::Yes, true, Qty::shares(500_000_000), Px(990_000));
        assert!(p.cost_basis > 0, "cost basis wrapped: {}", p.cost_basis);
        assert_eq!(p.cost_basis, px_core::Usd::dollars(495_000_000).0);
    }

    #[test]
    fn merging_sets_releases_collateral() {
        let mut e = engine();
        e.position
            .apply_leg(Leg::Yes, true, Qty::shares(100), Px(400_000));
        e.position
            .apply_leg(Leg::No, true, Qty::shares(100), Px(600_000));
        e.reevaluate();
        assert_eq!(e.free_capital_available(), Qty::shares(100).0);
        let merged = e.position.merge_sets();
        assert_eq!(merged, Qty::shares(100));
        assert_eq!(e.position.yes, Qty::ZERO);
        assert_eq!(e.position.no, Qty::ZERO);
        assert_eq!(e.free_capital_available(), 0);
    }

    #[test]
    fn buying_no_is_recorded_as_negative_yes() {
        let mut p = Position::default();
        p.apply_fill(-Qty::shares(50).0, Px(300_000)); // sell YES at 30c == buy NO at 70c
        assert_eq!(p.net(), -Qty::shares(50).0);
        assert_eq!(p.no, Qty::shares(50));
        // Cost basis is 50 * 70c = $35.
        assert_eq!(p.cost_basis, px_core::Usd::dollars(35).0);
    }

    #[test]
    fn quotes_stay_inside_the_unit_interval() {
        let mut e = InventoryEngine::new(50.0, InventoryLimits::default());
        e.on_fill(Qty::shares(5000).0, Px(990_000));
        let (b, a) = e.quote(Px(990_000), 0.5, 500.0, 20_000, 10_000);
        assert!(b.0 >= 0 && b.0 <= 1_000_000);
        assert!(a.0 >= 0 && a.0 <= 1_000_000);
    }
}
