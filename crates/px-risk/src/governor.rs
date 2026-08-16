//! Rate-limit governor — the real latency budget.
//!
//! # The uncomfortable arithmetic
//!
//! The brief asks for a sub-10-microsecond critical path. We build one, and it
//! is worth building. But the binding constraint on this venue is not
//! nanoseconds, it is **orders per second**.
//!
//! Polymarket meters order and cancel requests through per-signer token buckets.
//! At the entry tier that is 40 order tokens per second with a burst of 60.
//! Orders cannot be modified in place, so changing a quote costs one cancel
//! token *and* one order token. A two-sided quote is two of each.
//!
//! So the sustainable requote rate across the entire strategy is:
//!
//! ```text
//!   requotes/sec = order_rate / (2 * markets_quoted)
//! ```
//!
//! Ten markets at the entry tier gives **two requotes per second per market**.
//! Not two thousand. Two.
//!
//! A decision path that takes 8 microseconds and one that takes 800 are
//! therefore indistinguishable in throughput terms; both are four orders of
//! magnitude faster than the venue will accept work. What the fast path
//! actually buys is *decision quality within a fixed budget*: when we are only
//! permitted two quote updates per second, each one must be computed from the
//! freshest possible state, and the microseconds are what let us defer the
//! decision to the last instant before sending.
//!
//! That reframing is the reason this module exists and why it sits in `px-risk`
//! rather than in a networking layer. Quote budget is a scarce resource that
//! must be *allocated* — to the markets with the most edge, at the moments when
//! repricing is worth the token.

use px_core::{Nanos, Px};

/// Volume-based rate-limit tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Standard,
    Copper,
    Bronze,
    Silver,
    Gold,
    Platinum,
    Diamond,
    Elite,
}

impl Tier {
    /// `(order_rate, order_burst, cancel_rate, cancel_burst, allows_negative_cancel)`
    pub fn params(self) -> (f64, f64, f64, f64, bool) {
        match self {
            Tier::Standard => (40.0, 60.0, 80.0, 120.0, true),
            Tier::Copper => (60.0, 90.0, 120.0, 180.0, true),
            Tier::Bronze => (80.0, 120.0, 160.0, 240.0, true),
            Tier::Silver => (200.0, 300.0, 400.0, 600.0, true),
            Tier::Gold => (400.0, 600.0, 800.0, 1200.0, true),
            Tier::Platinum => (450.0, 675.0, 900.0, 1350.0, false),
            Tier::Diamond => (525.0, 787.0, 1050.0, 1575.0, false),
            Tier::Elite => (600.0, 900.0, 1200.0, 1800.0, false),
        }
    }

    /// Sustainable requotes per second per market when quoting both sides.
    pub fn requotes_per_market_per_sec(self, markets: usize) -> f64 {
        if markets == 0 {
            return f64::INFINITY;
        }
        let (order_rate, _, _, _, _) = self.params();
        order_rate / (2.0 * markets as f64)
    }
}

/// Continuous-refill token bucket, matching the venue's semantics.
#[derive(Clone, Copy, Debug)]
pub struct TokenBucket {
    rate: f64,
    burst: f64,
    tokens: f64,
    last: Nanos,
    allow_negative: bool,
}

impl TokenBucket {
    pub fn new(rate: f64, burst: f64, allow_negative: bool, now: Nanos) -> Self {
        TokenBucket {
            rate,
            burst,
            tokens: burst,
            last: now,
            allow_negative,
        }
    }

    #[inline]
    fn refill(&mut self, now: Nanos) {
        let dt = now.since(self.last).as_secs_f64();
        if dt > 0.0 {
            self.tokens = (self.tokens + dt * self.rate).min(self.burst);
            self.last = now;
        }
    }

    /// Attempt to spend `n` tokens. All-or-nothing, matching the venue's rule
    /// that a batch is admitted only if every entry fits.
    pub fn try_take(&mut self, n: f64, now: Nanos) -> bool {
        self.refill(now);
        if n > self.burst {
            // Can never be admitted as one request; the caller must split it.
            return false;
        }
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Spend tokens even into debt. Only used to model `cancel-all`, whose cost
    /// is not known until after it has run.
    pub fn force_take(&mut self, n: f64, now: Nanos) {
        self.refill(now);
        self.tokens -= n;
        if !self.allow_negative && self.tokens < 0.0 {
            self.tokens = 0.0;
        }
    }

    #[inline]
    pub fn available(&mut self, now: Nanos) -> f64 {
        self.refill(now);
        self.tokens
    }

    /// Seconds until `n` tokens will be available.
    pub fn wait_for(&mut self, n: f64, now: Nanos) -> f64 {
        self.refill(now);
        if self.tokens >= n {
            0.0
        } else {
            (n - self.tokens) / self.rate.max(1e-9)
        }
    }
}

/// Outcome of asking whether a quote change is worth its token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RequoteVerdict {
    /// Send it.
    Send,
    /// The price barely moved; the existing quote is good enough.
    NotWorthIt,
    /// We would like to, but the budget is spoken for.
    NoBudget,
    /// Budget reserved for emergency cancellation must not be touched.
    ReserveOnly,
}

/// Allocates the scarce quote budget.
#[derive(Clone, Copy, Debug)]
pub struct QuoteGovernor {
    pub orders: TokenBucket,
    pub cancels: TokenBucket,
    pub tier: Tier,
    /// Minimum price move, in ticks, that justifies a cancel/replace.
    pub min_requote_ticks: i32,
    /// Fraction of the cancel bucket held back so that a kill-switch
    /// `cancel-all` can always be issued. Running out of cancel tokens while
    /// holding live quotes during a feed fault is the worst state this system
    /// can be in, and it is entirely preventable by arithmetic.
    pub cancel_reserve: f64,
    /// Live resting orders, for sizing the emergency reserve.
    pub live_orders: u32,
}

impl QuoteGovernor {
    pub fn new(tier: Tier, now: Nanos) -> Self {
        let (orate, oburst, crate_, cburst, neg) = tier.params();
        QuoteGovernor {
            orders: TokenBucket::new(orate, oburst, false, now),
            cancels: TokenBucket::new(crate_, cburst, neg, now),
            tier,
            min_requote_ticks: 1,
            cancel_reserve: 0.25,
            live_orders: 0,
        }
    }

    /// Tokens we refuse to spend on routine cancels.
    #[inline]
    fn reserve(&self) -> f64 {
        let (_, _, _, cburst, _) = self.tier.params();
        // Enough to cancel every live order, plus the one token `cancel-all`
        // costs up front, bounded by a fraction of the burst.
        (self.live_orders as f64 + 1.0).min(cburst * self.cancel_reserve)
    }

    /// Should we cancel and replace an existing quote?
    ///
    /// Three tests, in order of cheapness: is the move big enough to matter, is
    /// there budget, and would spending it eat the emergency reserve.
    pub fn should_requote(
        &mut self,
        current: Px,
        desired: Px,
        tick: i32,
        now: Nanos,
        urgent: bool,
    ) -> RequoteVerdict {
        // `desired.0`/`current.0` are `Px`, bounded to `[0, 1_000_000]` —
        // the difference is well inside `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        let moved = (desired.0 - current.0).abs() / tick.max(1);
        if !urgent && moved < self.min_requote_ticks {
            return RequoteVerdict::NotWorthIt;
        }

        let cancels_left = self.cancels.available(now);
        if !urgent && cancels_left - 1.0 < self.reserve() {
            return RequoteVerdict::ReserveOnly;
        }

        if self.orders.available(now) < 1.0 || cancels_left < 1.0 {
            return RequoteVerdict::NoBudget;
        }

        RequoteVerdict::Send
    }

    /// Commit a cancel/replace pair.
    ///
    /// Both tokens or neither. The first version was
    /// `cancels.try_take(1.0) && orders.try_take(1.0)`, which short-circuits:
    /// when the cancel bucket had a token and the order bucket did not, the
    /// cancel token was spent and `false` returned. The caller then did nothing,
    /// so the token bought nothing. Under sustained pressure — exactly when the
    /// budget matters — that leaks a cancel token per attempt and drains the
    /// bucket the kill switch depends on.
    pub fn commit_requote(&mut self, now: Nanos) -> bool {
        // Check both before spending either.
        if self.cancels.available(now) < 1.0 || self.orders.available(now) < 1.0 {
            return false;
        }
        let c = self.cancels.try_take(1.0, now);
        let o = self.orders.try_take(1.0, now);
        debug_assert!(c && o, "availability checked immediately prior");
        c && o
    }

    /// Commit a batch of new orders (the venue charges one token per order).
    pub fn commit_batch(&mut self, n: usize, now: Nanos) -> bool {
        if self.orders.try_take(n as f64, now) {
            self.live_orders = self.live_orders.saturating_add(n as u32);
            true
        } else {
            false
        }
    }

    /// Emergency: cancel everything. Always permitted, may go into debt.
    pub fn commit_cancel_all(&mut self, now: Nanos) {
        self.cancels.force_take(1.0 + self.live_orders as f64, now);
        self.live_orders = 0;
    }

    /// How many markets we can sustainably quote at a target requote rate.
    pub fn max_markets(&self, target_requotes_per_sec: f64) -> usize {
        if target_requotes_per_sec <= 0.0 {
            return usize::MAX;
        }
        let (order_rate, _, _, _, _) = self.tier.params();
        (order_rate / (2.0 * target_requotes_per_sec))
            .floor()
            .max(0.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_refills_at_the_stated_rate() {
        let mut b = TokenBucket::new(40.0, 60.0, false, Nanos::ZERO);
        assert!(b.try_take(60.0, Nanos::ZERO));
        assert!(!b.try_take(1.0, Nanos::ZERO));
        // One second later, 40 tokens have accrued.
        assert!((b.available(Nanos::from_millis(1000)) - 40.0).abs() < 1e-6);
        assert!(b.try_take(40.0, Nanos::from_millis(1000)));
    }

    #[test]
    // `refill` clamps `tokens` with `.min(self.burst)`; after a 100s gap the
    // sum term is far past `burst`, so `.min` returns `burst` (`60.0`)
    // exactly, not a computed value that happens to land there.
    #[allow(clippy::float_cmp)]
    fn bucket_never_exceeds_burst() {
        let mut b = TokenBucket::new(40.0, 60.0, false, Nanos::ZERO);
        assert_eq!(b.available(Nanos::from_millis(100_000)), 60.0);
    }

    #[test]
    fn a_batch_larger_than_burst_can_never_be_admitted() {
        // Matches the venue's rule: split it, do not retry it.
        let mut b = TokenBucket::new(40.0, 60.0, false, Nanos::ZERO);
        assert!(!b.try_take(61.0, Nanos::from_millis(100_000)));
    }

    #[test]
    fn batches_are_all_or_nothing() {
        let mut b = TokenBucket::new(40.0, 60.0, false, Nanos::ZERO);
        b.try_take(55.0, Nanos::ZERO);
        // 5 left; a batch of 10 must be refused entirely, not partially filled.
        assert!(!b.try_take(10.0, Nanos::ZERO));
        assert!((b.available(Nanos::ZERO) - 5.0).abs() < 1e-9);
    }

    #[test]
    // `wait_for(0.0, ..)` hits `if self.tokens >= n { 0.0 }` — `tokens`
    // (>=0 after any refill) is always `>= 0.0`, so this is the literal
    // branch, not a computed value.
    #[allow(clippy::float_cmp)]
    fn wait_for_reports_the_right_delay() {
        let mut b = TokenBucket::new(40.0, 60.0, false, Nanos::ZERO);
        b.try_take(60.0, Nanos::ZERO);
        // 20 tokens at 40/s is half a second.
        assert!((b.wait_for(20.0, Nanos::ZERO) - 0.5).abs() < 1e-9);
        assert_eq!(b.wait_for(0.0, Nanos::ZERO), 0.0);
    }

    #[test]
    fn the_headline_constraint_is_two_requotes_per_second() {
        // The number that reframes the whole design.
        let t = Tier::Standard;
        assert!((t.requotes_per_market_per_sec(10) - 2.0).abs() < 1e-9);
        assert!((t.requotes_per_market_per_sec(1) - 20.0).abs() < 1e-9);
        // Even at the top tier, twenty markets buys fifteen requotes a second.
        assert!((Tier::Elite.requotes_per_market_per_sec(20) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn max_markets_inverts_the_same_relation() {
        let g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        assert_eq!(g.max_markets(2.0), 10);
        assert_eq!(g.max_markets(1.0), 20);
        assert_eq!(g.max_markets(20.0), 1);
    }

    #[test]
    fn a_sub_tick_move_does_not_justify_a_requote() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        let v = g.should_requote(
            Px(500_000),
            Px(504_000),
            10_000,
            Nanos::from_millis(100),
            false,
        );
        assert_eq!(v, RequoteVerdict::NotWorthIt);
    }

    #[test]
    fn a_full_tick_move_does() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        let v = g.should_requote(
            Px(500_000),
            Px(510_000),
            10_000,
            Nanos::from_millis(100),
            false,
        );
        assert_eq!(v, RequoteVerdict::Send);
        assert!(g.commit_requote(Nanos::from_millis(100)));
    }

    #[test]
    fn the_cancel_reserve_is_protected() {
        // With 100 live orders we must always be able to cancel all of them.
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        g.live_orders = 100;
        // Drain the cancel bucket down toward the reserve.
        g.cancels.try_take(90.0, Nanos::ZERO);
        let v = g.should_requote(Px(500_000), Px(530_000), 10_000, Nanos::ZERO, false);
        assert_eq!(v, RequoteVerdict::ReserveOnly);
    }

    #[test]
    fn urgent_requotes_may_use_the_reserve() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        g.live_orders = 100;
        g.cancels.try_take(90.0, Nanos::ZERO);
        let v = g.should_requote(Px(500_000), Px(530_000), 10_000, Nanos::ZERO, true);
        assert_eq!(v, RequoteVerdict::Send);
    }

    #[test]
    fn exhausted_budget_is_reported_rather_than_queued() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        g.orders.try_take(60.0, Nanos::ZERO);
        let v = g.should_requote(Px(500_000), Px(530_000), 10_000, Nanos::ZERO, true);
        assert_eq!(v, RequoteVerdict::NoBudget);
    }

    #[test]
    // `before`/the final `available()` call are the same bucket in
    // identical state (nothing consumes a token across 50 failed
    // attempts, which is the property under test) — bit-identical
    // computation, not a coincidental rounding match.
    #[allow(clippy::float_cmp)]
    fn a_failed_requote_does_not_leak_a_cancel_token() {
        // Regression: `cancels.try_take() && orders.try_take()` short-circuits.
        // When the order bucket was empty the cancel token was already spent and
        // bought nothing — a leak, under exactly the pressure where the budget
        // matters, draining the bucket the kill switch depends on.
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        g.orders.try_take(60.0, Nanos::ZERO); // drain orders, leave cancels full
        let before = g.cancels.available(Nanos::ZERO);

        for _ in 0..50 {
            assert!(!g.commit_requote(Nanos::ZERO));
        }
        assert_eq!(
            g.cancels.available(Nanos::ZERO),
            before,
            "cancel tokens leaked on failed requotes"
        );
    }

    #[test]
    fn a_successful_requote_spends_exactly_one_of_each() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        let o = g.orders.available(Nanos::ZERO);
        let c = g.cancels.available(Nanos::ZERO);
        assert!(g.commit_requote(Nanos::ZERO));
        assert!((g.orders.available(Nanos::ZERO) - (o - 1.0)).abs() < 1e-9);
        assert!((g.cancels.available(Nanos::ZERO) - (c - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn cancel_all_always_succeeds_and_may_go_into_debt() {
        let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
        g.live_orders = 200;
        g.cancels.try_take(120.0, Nanos::ZERO);
        g.commit_cancel_all(Nanos::ZERO);
        assert_eq!(g.live_orders, 0);
        // Standard tier permits a negative cancel balance.
        assert!(g.cancels.available(Nanos::ZERO) < 0.0);
    }

    #[test]
    fn ten_markets_at_standard_tier_sustains_exactly_the_predicted_rate() {
        // Simulate one second of quoting ten markets, both sides, at 2 Hz, and
        // confirm the budget holds. Then confirm 3 Hz does not.
        for (hz, should_fit) in [(2.0f64, true), (3.0f64, false)] {
            let mut g = QuoteGovernor::new(Tier::Standard, Nanos::ZERO);
            // Start from a steady state rather than a full burst.
            g.orders.try_take(60.0, Nanos::ZERO);
            g.cancels.try_take(120.0, Nanos::ZERO);

            let mut ok = true;
            let updates = (hz * 10.0) as u64; // 10 markets
            let period_ms = (1000.0 / hz) as u64;
            for round in 1..=3u64 {
                let now = Nanos::from_millis(round * period_ms);
                for _ in 0..(updates / (hz as u64)) {
                    // two orders + two cancels per market per requote
                    if !(g.cancels.try_take(2.0, now) && g.orders.try_take(2.0, now)) {
                        ok = false;
                    }
                }
            }
            assert_eq!(ok, should_fit, "at {hz} Hz");
        }
    }
}
