//! `docs/GOING-LIVE.md` Stage 0: "Replay against recorded venue book
//! data, not a synthetic venue. Every parameter in a simulator config is
//! a guess until it has been checked against real flow." — and the
//! README's own honest admission: "the largest missing piece is replay
//! against recorded book data. Every parameter in `SimConfig` is a guess
//! until then."
//!
//! This module is the "replay" half. The recorder that produces the data
//! this reads lives outside this zero-dependency workspace on purpose —
//! polling a real HTTPS endpoint needs a TLS stack, and this repo's
//! std-only guarantee is a promise about the simulation and execution
//! path, not about a one-time data-capture tool. See `tools/px-record/`
//! for the recorder and `recordings/README.md` for how the sample
//! recording shipped in this repo was captured.
//!
//! # What gets replaced, and what does not
//!
//! `SimConfig::venue_quotes = VenueQuoteSource::Recorded(..)` replaces
//! exactly one thing: what the venue's own book is quoting at each
//! instant. Instead of `crude_fair(lagged_spot, ..)` plus synthetic
//! noise and a synthetic 3-level spread (`Venue::maybe_rebuild`), the
//! book is set directly from a real recorded best bid/ask and size —
//! no synthetic spread, no synthetic depth-thinning, no noise added on
//! top, because there is nothing left to simulate: this *is* the real
//! book, at the resolution a public REST snapshot gives (the touch, not
//! full depth).
//!
//! Deliberately **not** replaced: the reference price path driving the
//! model's own fair-value belief (`spot`, still the synthetic GBM
//! walk), and the informed/uninformed flow model. Recorded venue quotes
//! only exist for the token this repo records (the YES side); the NO
//! side is derived as the complementary price (`Px::complement`), the
//! same modelling choice the synthetic path already makes rather than
//! polling a second endpoint for a redundant view of the same market.
//! Both are real, scoped simplifications, stated here rather than
//! silently assumed — a fuller replay (a genuinely independent recorded
//! reference feed, both sides of the book recorded directly) is future
//! work, not something this claims to already do.

use px_core::{DenseBook, Px, Qty, Side};

/// One real, timestamped snapshot of a venue's touch (best bid/ask and
/// their resting size) on a single market's YES token.
#[derive(Clone, Copy, Debug)]
pub struct RecordedQuote {
    /// Seconds since the recording's first snapshot of this market — the
    /// same clock `SimConfig`'s `0..duration_s` runs on.
    pub t_s: f64,
    pub bid: f64,
    pub bid_size: f64,
    pub ask: f64,
    pub ask_size: f64,
}

/// Parses a recording. One snapshot per non-comment, non-blank line:
/// `t_unix,market,bid,bid_size,ask,ask_size` — comma-separated, `#` lines
/// are comments. A single recorder run can (and the shipped recorder
/// does) poll several markets in the same pass; only rows whose `market`
/// field equals `market` are kept, so one file can back several replays.
///
/// Timestamps are normalised so the earliest matching row is `t_s = 0`,
/// matching `SimConfig`'s clock — the recorder writes wall-clock Unix
/// time, which this file format has no other use for. A malformed row
/// (wrong column count, a field that does not parse as a finite `f64`)
/// is skipped, not fatal: a single corrupt line — a truncated write, a
/// transient parse hiccup at capture time — must not discard an
/// otherwise-good recording.
pub fn load_recording(text: &str, market: &str) -> Vec<RecordedQuote> {
    let mut rows: Vec<(f64, RecordedQuote)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 6 || cols[1] != market {
            continue;
        }
        let parse = |s: &str| s.trim().parse::<f64>().ok().filter(|v: &f64| v.is_finite());
        let (Some(t_unix), Some(bid), Some(bid_size), Some(ask), Some(ask_size)) = (
            parse(cols[0]),
            parse(cols[2]),
            parse(cols[3]),
            parse(cols[4]),
            parse(cols[5]),
        ) else {
            continue;
        };
        rows.push((
            t_unix,
            RecordedQuote {
                t_s: t_unix,
                bid,
                bid_size,
                ask,
                ask_size,
            },
        ));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some(&(t0, _)) = rows.first() else {
        return Vec::new();
    };
    rows.into_iter()
        .map(|(_, mut q)| {
            q.t_s -= t0;
            q
        })
        .collect()
}

/// The most recent recorded quote at or before `t` — `None` only if `t`
/// is before the first snapshot (or the recording is empty). Same "hold
/// the last known value forward" semantics as `replay`'s own `lookup`,
/// for the same reason: a resting real quote does not update every tick
/// either, and treating a gap as "unknown" rather than "unchanged" would
/// make every polling interval look like a stale-feed condition.
pub fn lookup(quotes: &[RecordedQuote], t: f64) -> Option<RecordedQuote> {
    if quotes.is_empty() || t < quotes[0].t_s {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = quotes.len();
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if quotes[mid].t_s <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    quotes.get(lo).copied()
}

#[inline]
fn qty_from_f64(shares: f64) -> Qty {
    Qty((shares.max(0.0) * 1_000_000.0).round() as i64)
}

/// Sets `book`'s touch directly from a real recorded quote (the YES
/// side) — see the module doc for why no synthetic spread or noise is
/// layered on top.
pub fn set_book_from_quote(book: &mut DenseBook, q: RecordedQuote) {
    book.clear();
    let bid = Px::from_f64(q.bid);
    let ask = Px::from_f64(q.ask);
    if bid.0 > 0 {
        book.set_level(Side::Bid, bid, qty_from_f64(q.bid_size));
    }
    if ask.0 < 1_000_000 {
        book.set_level(Side::Ask, ask, qty_from_f64(q.ask_size));
    }
}

/// Sets `book`'s touch from the *complementary* side of a recorded YES
/// quote — buying NO at `1 - p` is economically identical to selling YES
/// at `p` (`Px::complement`), and this repo records only the YES token,
/// so the NO book is derived rather than independently polled. Same
/// modelling choice `Venue::maybe_rebuild`'s synthetic path already
/// makes (`1.0 - venue_view`).
pub fn set_complementary_book_from_quote(book: &mut DenseBook, q: RecordedQuote) {
    book.clear();
    let no_bid = Px::from_f64(q.ask).complement(); // 1 - ask
    let no_ask = Px::from_f64(q.bid).complement(); // 1 - bid
    if no_bid.0 > 0 {
        book.set_level(Side::Bid, no_bid, qty_from_f64(q.ask_size));
    }
    if no_ask.0 < 1_000_000 {
        book.set_level(Side::Ask, no_ask, qty_from_f64(q.bid_size));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# comment line, ignored
1000.000,btc_4h,0.55,100.0,0.57,80.0
1000.500,btc_1h,0.20,40.0,0.22,30.0
1003.000,btc_4h,0.60,90.0,0.62,75.0
not,a,valid,line,at,all
1006.000,btc_4h,0.58,,0.61,70.0
1009.000,btc_4h,0.59,85.0,0.60,72.0
";

    #[test]
    fn loads_only_the_requested_market_normalised_to_t_zero() {
        let q = load_recording(SAMPLE, "btc_4h");
        // 4 candidate btc_4h rows; one has a missing bid_size field and
        // must be skipped, not crash the parse.
        assert_eq!(q.len(), 3);
        assert!(q[0].t_s.abs() < 1e-12);
        assert!((q[0].bid - 0.55).abs() < 1e-12);
        assert!((q[1].t_s - 3.0).abs() < 1e-9);
        assert!((q[2].t_s - 9.0).abs() < 1e-9);
    }

    #[test]
    fn a_different_market_gets_its_own_independent_t_zero() {
        let q = load_recording(SAMPLE, "btc_1h");
        assert_eq!(q.len(), 1);
        assert!(q[0].t_s.abs() < 1e-12);
        assert!((q[0].bid - 0.20).abs() < 1e-12);
    }

    #[test]
    fn an_unknown_market_yields_an_empty_recording_not_an_error() {
        assert!(load_recording(SAMPLE, "eth_4h").is_empty());
    }

    #[test]
    fn empty_input_yields_an_empty_recording() {
        assert!(load_recording("", "btc_4h").is_empty());
        assert!(load_recording("# only comments\n\n", "btc_4h").is_empty());
    }

    #[test]
    fn out_of_order_input_is_sorted_before_normalisation() {
        let shuffled = "\
1003.000,m,0.60,90.0,0.62,75.0
1000.000,m,0.55,100.0,0.57,80.0
";
        let q = load_recording(shuffled, "m");
        assert_eq!(q.len(), 2);
        assert!(q[0].t_s.abs() < 1e-12);
        assert!((q[1].t_s - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lookup_holds_the_last_known_quote_forward() {
        let q = load_recording(SAMPLE, "btc_4h");
        // Between the second and third snapshot (t_s 3 and 9): must
        // return the second, not the third or None.
        let at = lookup(&q, 5.0).unwrap();
        assert!((at.t_s - 3.0).abs() < 1e-9);
        // Past the last snapshot: holds the last value, does not panic.
        let past_end = lookup(&q, 1000.0).unwrap();
        assert!((past_end.t_s - 9.0).abs() < 1e-9);
    }

    #[test]
    fn lookup_before_the_first_snapshot_is_none() {
        let q = load_recording(SAMPLE, "btc_4h");
        assert!(lookup(&q, -1.0).is_none());
    }

    #[test]
    fn lookup_on_an_empty_recording_is_none_not_a_panic() {
        assert!(lookup(&[], 0.0).is_none());
    }

    #[test]
    fn set_book_from_quote_places_a_real_bid_and_ask() {
        let mut book = DenseBook::new(10_000);
        let q = RecordedQuote {
            t_s: 0.0,
            bid: 0.55,
            bid_size: 100.0,
            ask: 0.58,
            ask_size: 80.0,
        };
        set_book_from_quote(&mut book, q);
        assert_eq!(book.best_bid(), Some(Px::from_f64(0.55)));
        assert_eq!(book.best_ask(), Some(Px::from_f64(0.58)));
    }

    #[test]
    fn complementary_book_mirrors_around_one_half() {
        let mut book = DenseBook::new(10_000);
        let q = RecordedQuote {
            t_s: 0.0,
            bid: 0.55,
            bid_size: 100.0,
            ask: 0.58,
            ask_size: 80.0,
        };
        set_complementary_book_from_quote(&mut book, q);
        // NO's bid is 1 - YES's ask; NO's ask is 1 - YES's bid.
        assert_eq!(book.best_bid(), Some(Px::from_f64(0.42)));
        assert_eq!(book.best_ask(), Some(Px::from_f64(0.45)));
        // A book built this way must never be crossed — the complement of
        // a valid, non-crossed YES quote cannot cross either.
        assert!(!book.is_crossed());
    }

    #[test]
    fn a_zero_bid_or_one_ask_is_left_off_the_book_rather_than_placed_at_the_domain_edge() {
        let mut book = DenseBook::new(10_000);
        let q = RecordedQuote {
            t_s: 0.0,
            bid: 0.0,
            bid_size: 100.0,
            ask: 1.0,
            ask_size: 80.0,
        };
        set_book_from_quote(&mut book, q);
        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None);
    }
}
