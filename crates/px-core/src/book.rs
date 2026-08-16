//! Dense, allocation-free limit order book with a depth walker.
//!
//! Prediction-market prices are bounded to `[0, 1]` and quantised to a tick of
//! 0.01 or 0.001. That is at most 1001 distinct price levels — small enough to
//! hold the entire book in a flat array indexed directly by price. No hash map,
//! no tree, no pointer chasing, no allocation. A `price_change` delta is an
//! array store plus a bitmap update.
//!
//! Best bid / best ask come from an occupancy bitmap (`[u64; 16]`), so finding
//! the top of book after a level is emptied is a handful of `trailing_zeros`
//! calls rather than a linear scan.
//!
//! The whole struct is 16 KiB and fits comfortably in L1d, which is the point:
//! the walk that produces our expected average entry price touches only
//! contiguous memory we already own.

use crate::num::{Px, Qty, Usd};

/// Number of addressable price levels: 0.000 .. 1.000 inclusive, milli-dollar grid.
pub const LEVELS: usize = 1001;
const WORDS: usize = LEVELS.div_ceil(64);

/// Largest resting size we will accept at one level: one billion shares, in
/// micro-shares. Polymarket's largest markets rest a few hundred thousand
/// shares, so this is six orders of magnitude of headroom — but it is finite,
/// which is the point. Unbounded sizes from the wire flow into i64 accumulators
/// in the depth walker and the exposure book.
pub const MAX_LEVEL_QTY: i64 = 1_000_000_000 * 1_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Bid,
    Ask,
}

/// Result of walking the book for a target size. This is the single most
/// important number in the system: it is the *expected average entry*, not the
/// touch price. A 9-cent theoretical edge against a touch that holds 50 shares
/// is not a 9-cent edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Walk {
    /// How much we would actually get.
    pub filled: Qty,
    /// Total cash paid (buy) or received (sell).
    pub notional: Usd,
    /// Volume-weighted average price of the fill, `Px::ZERO` if nothing filled.
    pub avg_px: Px,
    /// Price of the last (worst) level touched.
    pub worst_px: Px,
    /// How many distinct price levels we had to eat through.
    pub levels: u16,
    /// True if the book ran dry before we reached the requested size.
    pub exhausted: bool,
}

impl Walk {
    /// Slippage versus the touch, in micro-dollars per share. Always >= 0.
    ///
    /// `avg_px` and `touch` are both `Px`, bounded to `[0, 1_000_000]` —
    /// the difference is well inside `i32`, and `.abs()` of it cannot
    /// reach `i32::MIN`'s unrepresentable positive counterpart.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn slippage_vs(&self, touch: Px) -> i32 {
        if self.filled.is_zero() {
            0
        } else {
            (self.avg_px.0 - touch.0).abs()
        }
    }

    #[inline(always)]
    pub fn fill_ratio(&self, requested: Qty) -> f64 {
        if requested.0 <= 0 {
            0.0
        } else {
            self.filled.0 as f64 / requested.0 as f64
        }
    }
}

#[derive(Clone)]
pub struct DenseBook {
    /// Resting size at each milli-dollar price level, per side.
    bid_qty: [i64; LEVELS],
    ask_qty: [i64; LEVELS],
    bid_occ: [u64; WORDS],
    ask_occ: [u64; WORDS],
    /// Venue tick size in micro-dollars (10_000 for 1c, 1_000 for 0.1c).
    pub tick: i32,
    /// Exchange timestamp of the last applied update, in milliseconds.
    pub exch_ts_ms: u64,
    /// Local receive timestamp, in nanoseconds since process start.
    pub recv_ts_ns: u64,
    /// Monotonic counter of applied updates; used to detect gaps on resync.
    pub seq: u64,
    /// Count of malformed updates dropped at the boundary. A non-zero and
    /// rising value is a data-quality fault: either the venue is sending us
    /// garbage or our parser is wrong, and both mean stop quoting.
    pub rejected: u64,
}

impl Default for DenseBook {
    fn default() -> Self {
        DenseBook::new(10_000)
    }
}

/// Summarised, not exhaustive. A derived `Debug` would print 2,002 array slots,
/// nearly all of them zero — useless in a log and expensive to format on a path
/// where formatting anything at all is already a mistake.
impl core::fmt::Debug for DenseBook {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DenseBook")
            .field("bid", &self.best_bid())
            .field("ask", &self.best_ask())
            .field("spread_ticks", &self.spread_ticks())
            .field("tick", &self.tick)
            .field("seq", &self.seq)
            .field("rejected", &self.rejected)
            .field("crossed", &self.is_crossed())
            .finish()
    }
}

impl DenseBook {
    pub fn new(tick: i32) -> Self {
        DenseBook {
            bid_qty: [0i64; LEVELS],
            ask_qty: [0i64; LEVELS],
            bid_occ: [0u64; WORDS],
            ask_occ: [0u64; WORDS],
            tick,
            exch_ts_ms: 0,
            recv_ts_ns: 0,
            seq: 0,
            rejected: 0,
        }
    }

    /// Map a price to a level index, or `None` if it is outside the tradable
    /// domain.
    ///
    /// # Why this is not a `debug_assert`
    ///
    /// It was, and that was a denial-of-service vector. Prices arrive as JSON
    /// strings from a network peer we do not control. A malformed or hostile
    /// `price` field — `"1.5"`, `"-0.2"`, a parser that dropped a decimal point
    /// — produced an index past the end of a 1001-element array. In release
    /// builds `debug_assert` is compiled out, so the result was an
    /// out-of-bounds index panic, and this workspace builds with
    /// `panic = "abort"`. One bad message from the venue would kill the process
    /// while holding live quotes.
    ///
    /// Untrusted input must be validated at the boundary, and the boundary is
    /// here. Bad levels are dropped and counted, never indexed.
    #[inline(always)]
    fn idx(px: Px) -> Option<usize> {
        if px.0 < 0 || px.0 > 1_000_000 {
            return None;
        }
        Some((px.0 as usize) / 1000)
    }

    /// Every caller passes an `idx` already bounded to `< LEVELS` (1001),
    /// so the product tops out at `1_000_000` — exactly `Px::ONE`, never
    /// past it.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn px_of(idx: usize) -> Px {
        Px((idx as i32) * 1000)
    }

    /// `i` is always a level index already validated by `idx` (`< LEVELS`
    /// = 1001), so `i >> 6` is at most 15 — inside `WORDS` = 16 — and
    /// `i & 63` is a bit position, never a second index. Both the word
    /// index and the arithmetic that builds it are bounded by that same
    /// invariant, not by anything this function itself checks.
    #[inline(always)]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn set_bit(occ: &mut [u64; WORDS], i: usize) {
        occ[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline(always)]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn clear_bit(occ: &mut [u64; WORDS], i: usize) {
        occ[i >> 6] &= !(1u64 << (i & 63));
    }

    /// `w` is bounded by the `while w < WORDS` guard on every read, and
    /// `(w << 6) + trailing_zeros` is at most `15*64 + 63 = 1023` —
    /// nowhere near overflowing `usize`.
    #[inline(always)]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn lowest_set(occ: &[u64; WORDS]) -> Option<usize> {
        let mut w = 0;
        while w < WORDS {
            let word = occ[w];
            if word != 0 {
                return Some((w << 6) + word.trailing_zeros() as usize);
            }
            w += 1;
        }
        None
    }

    /// `w` is decremented from `WORDS` *before* every read, so it is
    /// always `< WORDS`, and never reaches zero from below (the loop
    /// exits via `w > 0` first) — same bound as `lowest_set` above.
    #[inline(always)]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn highest_set(occ: &[u64; WORDS]) -> Option<usize> {
        let mut w = WORDS;
        while w > 0 {
            w -= 1;
            let word = occ[w];
            if word != 0 {
                return Some((w << 6) + (63 - word.leading_zeros() as usize));
            }
        }
        None
    }

    /// Apply an absolute level update. Polymarket's `price_change` events carry
    /// the *new aggregate size* at the level (0 removes it), so this is a store,
    /// not an increment — which is exactly what we want: it is idempotent, so a
    /// duplicated message is harmless.
    /// Returns `false` if the update was rejected as malformed.
    ///
    /// `i` comes only from `Self::idx`, which already returned `None` (and
    /// this function already returned) for anything outside `[0, LEVELS)`
    /// — see `idx`'s own doc comment for why that check exists and lives
    /// exactly there, at the untrusted-input boundary.
    #[inline]
    #[allow(clippy::indexing_slicing)]
    pub fn set_level(&mut self, side: Side, px: Px, size: Qty) -> bool {
        let i = match Self::idx(px) {
            Some(i) => i,
            None => {
                self.rejected = self.rejected.saturating_add(1);
                return false;
            }
        };
        // A negative resting size is nonsense; a size beyond any plausible book
        // is either a units error or an attack on our accumulators.
        if size.0 < 0 || size.0 > MAX_LEVEL_QTY {
            self.rejected = self.rejected.saturating_add(1);
            return false;
        }
        match side {
            Side::Bid => {
                self.bid_qty[i] = size.0;
                if size.0 > 0 {
                    Self::set_bit(&mut self.bid_occ, i);
                } else {
                    Self::clear_bit(&mut self.bid_occ, i);
                }
            }
            Side::Ask => {
                self.ask_qty[i] = size.0;
                if size.0 > 0 {
                    Self::set_bit(&mut self.ask_occ, i);
                } else {
                    Self::clear_bit(&mut self.ask_occ, i);
                }
            }
        }
        self.seq = self.seq.wrapping_add(1);
        true
    }

    /// Wipe both sides. Called when the venue sends a fresh `book` snapshot, or
    /// when we detect a sequence gap and must resynchronise from scratch.
    pub fn clear(&mut self) {
        self.bid_qty = [0i64; LEVELS];
        self.ask_qty = [0i64; LEVELS];
        self.bid_occ = [0u64; WORDS];
        self.ask_occ = [0u64; WORDS];
    }

    #[inline(always)]
    pub fn best_bid(&self) -> Option<Px> {
        Self::highest_set(&self.bid_occ).map(Self::px_of)
    }

    #[inline(always)]
    pub fn best_ask(&self) -> Option<Px> {
        Self::lowest_set(&self.ask_occ).map(Self::px_of)
    }

    /// Resting size at a price. An out-of-domain price holds nothing, by
    /// definition — reporting zero is both true and safe.
    ///
    /// `i` comes only from `Self::idx`, same guarantee as `set_level`.
    #[inline(always)]
    #[allow(clippy::indexing_slicing)]
    pub fn size_at(&self, side: Side, px: Px) -> Qty {
        let i = match Self::idx(px) {
            Some(i) => i,
            None => return Qty::ZERO,
        };
        Qty(match side {
            Side::Bid => self.bid_qty[i],
            Side::Ask => self.ask_qty[i],
        })
    }

    /// Midpoint, or `None` if either side is empty. Note we deliberately do not
    /// fall back to a one-sided proxy: a one-sided book is a data-quality event,
    /// and the caller should treat it as such rather than quote off a guess.
    ///
    /// `b.0` and `a.0` are each in `[0, 1_000_000]`, so the sum is bounded
    /// well inside `i32`.
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn mid(&self) -> Option<Px> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(Px((b.0 + a.0) / 2)),
            _ => None,
        }
    }

    /// `a.0` and `b.0` are each in `[0, 1_000_000]`, so the difference is
    /// bounded well inside `i32`; the division is by `self.tick`, which is
    /// always constructed positive (`DenseBook::new`'s only caller-supplied
    /// tick values are 10_000 or 1_000).
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn spread_ticks(&self) -> Option<i32> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((a.0 - b.0) / self.tick),
            _ => None,
        }
    }

    /// Total resting size no worse than `limit`. Answers "how much can I lift
    /// without paying more than X".
    ///
    /// Every index below is either the result of `lowest_set`/
    /// `highest_set` (already `< LEVELS`) or explicitly re-checked by the
    /// loop guard immediately before use — `i < LEVELS` on the ask side,
    /// `i == 0` breaking before the bid side's `i -= 1` can underflow. The
    /// running `total` is bounded by `LEVELS * MAX_LEVEL_QTY` (~1e18),
    /// inside `i64`'s ~9.2e18 range with room to spare.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub fn depth_to(&self, side: Side, limit: Px) -> Qty {
        let mut total = 0i64;
        match side {
            // Taking liquidity from asks: everything priced <= limit.
            Side::Ask => {
                let mut i = match Self::lowest_set(&self.ask_occ) {
                    Some(i) => i,
                    None => return Qty::ZERO,
                };
                // An out-of-domain limit is clamped rather than rejected: the
                // caller asked for "no worse than X", and X outside [0,1] just
                // means "no limit at all" on that side.
                let stop = Self::idx(limit.clamp_unit()).unwrap_or(0);
                while i <= stop && i < LEVELS {
                    total += self.ask_qty[i];
                    i += 1;
                }
            }
            // Hitting bids: everything priced >= limit.
            Side::Bid => {
                let mut i = match Self::highest_set(&self.bid_occ) {
                    Some(i) => i,
                    None => return Qty::ZERO,
                };
                // An out-of-domain limit is clamped rather than rejected: the
                // caller asked for "no worse than X", and X outside [0,1] just
                // means "no limit at all" on that side.
                let stop = Self::idx(limit.clamp_unit()).unwrap_or(0);
                loop {
                    total += self.bid_qty[i];
                    if i == 0 || i <= stop {
                        break;
                    }
                    i -= 1;
                }
            }
        }
        Qty(total)
    }

    /// Walk the ask side buying `want` shares, refusing to pay above `limit`.
    ///
    /// This is the expected-average-entry estimator. It reports partial fills
    /// honestly: if the book only holds 50 of the 500 shares we want inside our
    /// limit, `filled` is 50 and `exhausted` is true, and the edge calculator
    /// will size against 50, not 500.
    pub fn walk_buy(&self, want: Qty, limit: Px) -> Walk {
        let start = match Self::lowest_set(&self.ask_occ) {
            Some(i) => i,
            None => return Walk::default(),
        };
        self.walk(
            start,
            Self::idx(limit.clamp_unit()).unwrap_or(LEVELS - 1),
            1,
            want,
            &self.ask_qty,
        )
    }

    /// Walk the bid side selling `want` shares, refusing to accept below `limit`.
    pub fn walk_sell(&self, want: Qty, limit: Px) -> Walk {
        let start = match Self::highest_set(&self.bid_occ) {
            Some(i) => i,
            None => return Walk::default(),
        };
        self.walk(
            start,
            Self::idx(limit.clamp_unit()).unwrap_or(0),
            -1,
            want,
            &self.bid_qty,
        )
    }

    /// `u` is re-derived from `i` immediately after the `i < 0 || i as
    /// usize >= LEVELS` guard breaks the loop, so `qty[u]` is always in
    /// range. `cash` accumulates in `i128` specifically so the
    /// price*quantity products here have headroom regardless of `filled`;
    /// `filled`/`remaining` are bounded by `want.0`, and `levels: u16`
    /// increments at most `LEVELS` (1001) times, nowhere near `u16::MAX`.
    #[inline]
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn walk(&self, start: usize, stop: usize, step: isize, want: Qty, qty: &[i64; LEVELS]) -> Walk {
        let mut remaining = want.0;
        let mut cash = 0i128;
        let mut filled = 0i64;
        let mut levels = 0u16;
        let mut worst = Px::ZERO;
        let mut i = start as isize;

        while remaining > 0 {
            if i < 0 || i as usize >= LEVELS {
                break;
            }
            let u = i as usize;
            if (step > 0 && u > stop) || (step < 0 && u < stop) {
                break;
            }
            let avail = qty[u];
            if avail > 0 {
                let take = if avail < remaining { avail } else { remaining };
                let px = Self::px_of(u);
                cash += (px.0 as i128) * (take as i128);
                filled += take;
                remaining -= take;
                levels += 1;
                worst = px;
            }
            i += step;
        }

        let filled_q = Qty(filled);
        let notional_usd = Usd((cash / 1_000_000i128) as i64);
        let avg = if filled > 0 {
            Px((cash / (filled as i128)) as i32)
        } else {
            Px::ZERO
        };

        Walk {
            filled: filled_q,
            notional: notional_usd,
            avg_px: avg,
            worst_px: worst,
            levels,
            exhausted: remaining > 0,
        }
    }

    /// Convenience: walk with no price limit at all.
    #[inline]
    pub fn walk_buy_unbounded(&self, want: Qty) -> Walk {
        self.walk_buy(want, Px::ONE)
    }

    #[inline]
    pub fn walk_sell_unbounded(&self, want: Qty) -> Walk {
        self.walk_sell(want, Px::ZERO)
    }

    /// Visit each occupied level from the touch outward, best price first.
    /// Stops early when `f` returns `false`.
    ///
    /// Exists so `px-edge` can find the size that *maximises total edge* rather
    /// than assuming the requested size is the right one — walking deeper buys
    /// more shares at a worse average, and where that trade-off turns is a
    /// property of the book, not of our intent.
    ///
    /// Same bound as `depth_to`: the ask side's `i` is guarded by
    /// `i < LEVELS` before every read, and the bid side's `i` only ever
    /// decrements after an explicit `i == 0` check, so it cannot underflow.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub fn for_each_level<F>(&self, side: Side, mut f: F)
    where
        F: FnMut(Px, Qty) -> bool,
    {
        match side {
            Side::Ask => {
                let mut i = match Self::lowest_set(&self.ask_occ) {
                    Some(i) => i,
                    None => return,
                };
                while i < LEVELS {
                    let q = self.ask_qty[i];
                    if q > 0 && !f(Self::px_of(i), Qty(q)) {
                        return;
                    }
                    i += 1;
                }
            }
            Side::Bid => {
                let mut i = match Self::highest_set(&self.bid_occ) {
                    Some(i) => i,
                    None => return,
                };
                loop {
                    let q = self.bid_qty[i];
                    if q > 0 && !f(Self::px_of(i), Qty(q)) {
                        return;
                    }
                    if i == 0 {
                        return;
                    }
                    i -= 1;
                }
            }
        }
    }

    /// Sanity check used by the data-quality guard: a crossed or locked book
    /// means we have lost sequencing and must resync before quoting again.
    #[inline]
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => b.0 >= a.0,
            _ => false,
        }
    }

    /// Sum of resting size on both sides. Feeds the liquidity term of the
    /// fair-value model and the "is this market alive" check.
    ///
    /// `i` ranges over `0..LEVELS` directly, and `t` is bounded by
    /// `2 * LEVELS * MAX_LEVEL_QTY` (~2e18) — inside `i64`'s ~9.2e18 range.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub fn total_depth(&self) -> Qty {
        let mut t = 0i64;
        for i in 0..LEVELS {
            t += self.bid_qty[i] + self.ask_qty[i];
        }
        Qty(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book_with_ladder() -> DenseBook {
        let mut b = DenseBook::new(10_000);
        // Bids: 48c x 30, 47c x 100, 46c x 500
        b.set_level(Side::Bid, Px(480_000), Qty::shares(30));
        b.set_level(Side::Bid, Px(470_000), Qty::shares(100));
        b.set_level(Side::Bid, Px(460_000), Qty::shares(500));
        // Asks: 52c x 25, 53c x 60, 54c x 400
        b.set_level(Side::Ask, Px(520_000), Qty::shares(25));
        b.set_level(Side::Ask, Px(530_000), Qty::shares(60));
        b.set_level(Side::Ask, Px(540_000), Qty::shares(400));
        b
    }

    #[test]
    fn top_of_book_is_correct() {
        let b = book_with_ladder();
        assert_eq!(b.best_bid(), Some(Px(480_000)));
        assert_eq!(b.best_ask(), Some(Px(520_000)));
        assert_eq!(b.mid(), Some(Px(500_000)));
        assert_eq!(b.spread_ticks(), Some(4));
        assert!(!b.is_crossed());
    }

    #[test]
    fn top_of_book_updates_when_level_is_emptied() {
        let mut b = book_with_ladder();
        b.set_level(Side::Ask, Px(520_000), Qty::ZERO);
        assert_eq!(b.best_ask(), Some(Px(530_000)));
        b.set_level(Side::Ask, Px(530_000), Qty::ZERO);
        assert_eq!(b.best_ask(), Some(Px(540_000)));
        b.set_level(Side::Ask, Px(540_000), Qty::ZERO);
        assert_eq!(b.best_ask(), None);
        assert_eq!(b.mid(), None);
    }

    #[test]
    fn walk_small_order_stays_at_touch() {
        let b = book_with_ladder();
        let w = b.walk_buy_unbounded(Qty::shares(10));
        assert_eq!(w.filled, Qty::shares(10));
        assert_eq!(w.avg_px, Px(520_000));
        assert_eq!(w.levels, 1);
        assert!(!w.exhausted);
        assert_eq!(w.slippage_vs(Px(520_000)), 0);
    }

    #[test]
    fn walk_large_order_reveals_true_entry_price() {
        // This is the case the whole design exists for: the touch says 52c, but
        // 200 shares actually costs an average of 53.35c.
        let b = book_with_ladder();
        let w = b.walk_buy_unbounded(Qty::shares(200));
        assert_eq!(w.filled, Qty::shares(200));
        // 25@52 + 60@53 + 115@54 = 1300 + 3180 + 6210 = 10690 cents / 200 = 53.45c
        assert_eq!(w.avg_px, Px(534_500));
        assert_eq!(w.worst_px, Px(540_000));
        assert_eq!(w.levels, 3);
        assert_eq!(
            w.notional,
            crate::num::Usd::dollars(106) + crate::num::Usd(900_000)
        );
        // 1.45c of slippage against a touch-price model.
        assert_eq!(w.slippage_vs(Px(520_000)), 14_500);
    }

    #[test]
    fn walk_respects_price_limit_and_reports_exhaustion() {
        let b = book_with_ladder();
        // Refuse to pay above 53c: only 85 of the 200 shares are available.
        let w = b.walk_buy(Qty::shares(200), Px(530_000));
        assert_eq!(w.filled, Qty::shares(85));
        assert!(w.exhausted);
        assert_eq!(w.worst_px, Px(530_000));
        assert!((w.fill_ratio(Qty::shares(200)) - 0.425).abs() < 1e-9);
    }

    #[test]
    fn walk_empty_book_is_safe() {
        let b = DenseBook::new(10_000);
        let w = b.walk_buy_unbounded(Qty::shares(100));
        assert_eq!(w.filled, Qty::ZERO);
        assert!(w.exhausted || w.filled.is_zero());
        assert_eq!(w.avg_px, Px::ZERO);
    }

    #[test]
    fn walk_sell_mirrors_walk_buy() {
        let b = book_with_ladder();
        let w = b.walk_sell_unbounded(Qty::shares(200));
        // 30@48 + 100@47 + 70@46 = 1440 + 4700 + 3220 = 9360 / 200 = 46.8c
        assert_eq!(w.filled, Qty::shares(200));
        assert_eq!(w.avg_px, Px(468_000));
        assert_eq!(w.worst_px, Px(460_000));
    }

    #[test]
    fn depth_to_limit_matches_walk() {
        let b = book_with_ladder();
        assert_eq!(b.depth_to(Side::Ask, Px(530_000)), Qty::shares(85));
        assert_eq!(b.depth_to(Side::Ask, Px(540_000)), Qty::shares(485));
        assert_eq!(b.depth_to(Side::Bid, Px(470_000)), Qty::shares(130));
    }

    #[test]
    fn crossed_book_is_detected() {
        let mut b = DenseBook::new(10_000);
        b.set_level(Side::Bid, Px(550_000), Qty::shares(10));
        b.set_level(Side::Ask, Px(540_000), Qty::shares(10));
        assert!(b.is_crossed());
    }

    #[test]
    fn hostile_prices_are_rejected_not_indexed() {
        // Prices arrive as JSON from a network peer. Before this guard, a price
        // outside [0,1] indexed past a 1001-element array — an out-of-bounds
        // panic, and this workspace aborts on panic. One malformed message
        // would have killed the process while holding live quotes.
        let mut b = DenseBook::new(10_000);
        let hostile = [
            Px(1_500_000),
            Px(-1),
            Px(i32::MAX),
            Px(i32::MIN),
            Px(2_000_000_000),
        ];
        for px in hostile {
            assert!(
                !b.set_level(Side::Bid, px, Qty::shares(10)),
                "accepted {px:?}"
            );
            assert_eq!(b.size_at(Side::Bid, px), Qty::ZERO);
        }
        assert_eq!(b.rejected, hostile.len() as u64);
        assert_eq!(b.best_bid(), None);

        // Valid prices still work, and the boundaries are inclusive.
        assert!(b.set_level(Side::Bid, Px(0), Qty::shares(1)));
        assert!(b.set_level(Side::Ask, Px(1_000_000), Qty::shares(1)));
    }

    #[test]
    fn absurd_sizes_are_rejected() {
        let mut b = DenseBook::new(10_000);
        assert!(!b.set_level(Side::Bid, Px(500_000), Qty(-5)));
        assert!(!b.set_level(Side::Bid, Px(500_000), Qty(i64::MAX)));
        assert_eq!(b.rejected, 2);
        assert_eq!(b.best_bid(), None);
    }

    #[test]
    fn walks_survive_an_out_of_domain_limit() {
        let b = book_with_ladder();
        // Nonsense limits must clamp, not panic.
        let w = b.walk_buy(Qty::shares(50), Px(9_000_000));
        assert!(w.filled > Qty::ZERO);
        let w2 = b.walk_sell(Qty::shares(50), Px(-500_000));
        assert!(w2.filled > Qty::ZERO);
        assert_eq!(b.depth_to(Side::Ask, Px(i32::MAX)), Qty::shares(485));
    }

    #[test]
    fn set_level_is_idempotent() {
        // A duplicated venue message must not double-count.
        let mut b = DenseBook::new(10_000);
        b.set_level(Side::Ask, Px(520_000), Qty::shares(25));
        b.set_level(Side::Ask, Px(520_000), Qty::shares(25));
        assert_eq!(b.size_at(Side::Ask, Px(520_000)), Qty::shares(25));
    }
}
