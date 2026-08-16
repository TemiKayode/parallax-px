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
//!
//! # `run()`'s `outcome_yes` is still synthetic — score against it with care
//!
//! `Report::outcome_yes` is computed from the *synthetic* reference
//! path's settlement TWAP against `cfg.strike` (`replay.rs`,
//! `outcome_yes = settle_twap > cfg.strike`), regardless of
//! `venue_quotes`. When replaying a real recording of a market that has
//! actually settled, that is the wrong outcome to score forecasts
//! against — it answers "what would a random synthetic path have
//! settled to," not "what did this real market actually resolve." This
//! was not theoretical: the first attempt at scoring
//! `recordings/polymarket_sample.csv` this way produced a base rate of
//! 0% (both recorded markets reading as resolved NO) against real quotes
//! that were visibly trading up near 0.98-0.999 — the tell that the
//! *outcome*, not just the forecast, needs to come from outside `run()`
//! when the venue is real. `px-replay`'s "real recorded data" section
//! fetches each market's actual resolution from Polymarket's API after
//! the fact and overwrites `Resolved::outcome` with it before scoring;
//! any other caller replaying a settled recording needs to do the same.

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
    let rows = parse_quote_rows(text, market);
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

/// Shared parsing step behind `load_recording` and `complementarity_error`
/// — sorted `(t_unix, RecordedQuote)` pairs with `t_s` still equal to the
/// *absolute* recorded time, not yet normalised to start at 0. Kept
/// separate specifically so `complementarity_error` (and `calibration`'s
/// lag analysis) can compare two markets' recordings against a shared
/// clock; `load_recording` on its own has no reason to expose absolute
/// time, since every other caller only ever replays one market's
/// recording against its own start. `pub(crate)` rather than private so
/// `crate::calibration` can reuse it without duplicating the parser.
/// `cols.len() != 6` is checked before any of `cols[0]` through
/// `cols[5]` is read, so every index used below is in range whenever
/// execution reaches it.
#[allow(clippy::indexing_slicing)]
pub(crate) fn parse_quote_rows(text: &str, market: &str) -> Vec<(f64, RecordedQuote)> {
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
    rows
}

/// The most recent recorded quote at or before `t` — `None` only if `t`
/// is before the first snapshot (or the recording is empty). Same "hold
/// the last known value forward" semantics as `replay`'s own `lookup`,
/// for the same reason: a resting real quote does not update every tick
/// either, and treating a gap as "unknown" rather than "unchanged" would
/// make every polling interval look like a stale-feed condition.
/// `quotes.is_empty() || ...` short-circuits before `quotes[0]` runs on
/// an empty slice; `lo`/`hi`/`mid` are a standard binary search staying
/// within `[0, quotes.len())` throughout — same proof as
/// `calibration::lookup_value`.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
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

/// One real price level: what `RecordedQuote` reduces the whole book to
/// (the touch) is really the first entry of a list of these.
#[derive(Clone, Copy, Debug)]
pub struct RecordedLevel {
    pub price: f64,
    pub size: f64,
}

/// Both sides of a real venue's book at one instant, at full depth —
/// everything the public snapshot endpoint returned, not just the touch.
/// `RecordedQuote` (best bid/ask only) is what this repo's shipped
/// sample recording uses; this is for the queue-position-calibration
/// question `RecordedQuote` structurally cannot answer, since a resting
/// order's real fill probability depends on how much size sits ahead of
/// it, not just on where the touch is.
#[derive(Clone, Debug, Default)]
pub struct RecordedBookSnapshot {
    pub t_s: f64,
    /// Best price first (descending).
    pub bids: Vec<RecordedLevel>,
    /// Best price first (ascending).
    pub asks: Vec<RecordedLevel>,
}

/// Parses a full-depth recording. Two rows per snapshot, one per side:
/// `t_unix,market,side,price:size,price:size,...` (`side` is `bid` or
/// `ask`). Rows for the same `market` at the same `t_unix` (to
/// microsecond tolerance — both rows come from the same polling pass)
/// are merged into one `RecordedBookSnapshot`. Same tolerance as
/// `load_recording`: a malformed row, or a malformed individual level
/// within an otherwise-good row, is skipped rather than discarding the
/// whole recording.
pub fn load_recording_l2(text: &str, market: &str) -> Vec<RecordedBookSnapshot> {
    let mut rows: Vec<(f64, bool, Vec<RecordedLevel>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.splitn(4, ',');
        let (Some(t_str), Some(m), Some(side_str), Some(rest)) =
            (cols.next(), cols.next(), cols.next(), cols.next())
        else {
            continue;
        };
        if m != market {
            continue;
        }
        let Ok(t_unix) = t_str.trim().parse::<f64>() else {
            continue;
        };
        if !t_unix.is_finite() {
            continue;
        }
        let is_bid = match side_str.trim() {
            "bid" => true,
            "ask" => false,
            _ => continue,
        };
        let mut levels = Vec::new();
        for level_str in rest.split(',') {
            let mut parts = level_str.trim().splitn(2, ':');
            let (Some(p), Some(s)) = (parts.next(), parts.next()) else {
                continue;
            };
            let (Ok(price), Ok(size)) = (p.trim().parse::<f64>(), s.trim().parse::<f64>()) else {
                continue;
            };
            if price.is_finite() && size.is_finite() {
                levels.push(RecordedLevel { price, size });
            }
        }
        rows.push((t_unix, is_bid, levels));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some(&(t0, _, _)) = rows.first() else {
        return Vec::new();
    };

    let mut snapshots: Vec<RecordedBookSnapshot> = Vec::new();
    for (t_unix, is_bid, levels) in rows {
        let t_s = t_unix - t0;
        match snapshots.last_mut() {
            Some(last) if (last.t_s - t_s).abs() < 1e-3 => {
                if is_bid {
                    last.bids = levels;
                } else {
                    last.asks = levels;
                }
            }
            _ => {
                let mut snap = RecordedBookSnapshot {
                    t_s,
                    ..Default::default()
                };
                if is_bid {
                    snap.bids = levels;
                } else {
                    snap.asks = levels;
                }
                snapshots.push(snap);
            }
        }
    }
    snapshots
}

/// Same "hold the last known snapshot forward" semantics as `lookup`.
/// Same proof as `lookup` above.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn lookup_l2(snapshots: &[RecordedBookSnapshot], t: f64) -> Option<&RecordedBookSnapshot> {
    if snapshots.is_empty() || t < snapshots[0].t_s {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = snapshots.len();
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if snapshots[mid].t_s <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    snapshots.get(lo)
}

/// Sets every real recorded level on `book`'s YES side — full depth,
/// not just the touch `set_book_from_quote` places.
pub fn set_book_from_l2_snapshot(book: &mut DenseBook, snap: &RecordedBookSnapshot) {
    book.clear();
    for lvl in &snap.bids {
        let px = Px::from_f64(lvl.price);
        if px.0 > 0 {
            book.set_level(Side::Bid, px, qty_from_f64(lvl.size));
        }
    }
    for lvl in &snap.asks {
        let px = Px::from_f64(lvl.price);
        if px.0 < 1_000_000 {
            book.set_level(Side::Ask, px, qty_from_f64(lvl.size));
        }
    }
}

/// Sets `book`'s touch directly from a real recorded quote (the YES
/// side) — see the module doc for why no synthetic spread or noise is
/// layered on top.
///
/// Deliberately leaves `book.exch_ts_ms`/`recv_ts_ns` at whatever they
/// were — `0` on a freshly constructed book. Polymarket's public REST
/// book endpoint (what `tools/px-record` polls) returns a full snapshot
/// with no server-side timestamp of its own; `RecordedQuote::t_s` is our
/// own local capture time, normalised per market, not an exchange
/// timestamp. Stamping `exch_ts_ms` from it would misrepresent a recorded
/// replay as having a real feed-latency measurement it does not have —
/// see `DenseBook::measured_latency_s`, which the synthetic venue path in
/// `replay::Venue` *does* stamp honestly, because there the exchange's
/// "as of" instant is a real, known quantity (`t - venue_lag_s`).
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

/// Checks the assumption `set_complementary_book_from_quote` has to make
/// against two *independently recorded* real quote series instead of
/// assuming it: real complementarity would mean `no.bid == 1 - yes.ask`
/// and `no.ask == 1 - yes.bid` at every matched instant. `yes_market` and
/// `no_market` are two `market` values recorded in the same `text` (the
/// shipped recorder writes the NO side as `{market}-no`) — matched by
/// *absolute* recorded time, deliberately not through `load_recording`'s
/// already-normalised `t_s`, which zeroes each market's own first row
/// independently and would silently compare the wrong instants whenever
/// the two series' recordings did not start on the exact same poll.
///
/// Each returned pair is `(bid_error, ask_error)`, the absolute deviation
/// from perfect complementarity in price units, one entry per
/// `no_market` snapshot matched against the most recent `yes_market`
/// quote at or before it (same "hold forward" semantics as `lookup`). A
/// `no_market` snapshot with nothing in `yes_market` yet to compare
/// against is skipped, not treated as zero error.
///
/// `yes_rows.is_empty() || ...` short-circuits before indexing on an
/// empty slice; the inner `lo`/`hi`/`mid` walk is the same bounded
/// binary search as `lookup`.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn complementarity_error(text: &str, yes_market: &str, no_market: &str) -> Vec<(f64, f64)> {
    let yes_rows = parse_quote_rows(text, yes_market);
    let no_rows = parse_quote_rows(text, no_market);
    let mut out = Vec::with_capacity(no_rows.len());
    for (t_unix, n) in &no_rows {
        if yes_rows.is_empty() || *t_unix < yes_rows[0].0 {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = yes_rows.len();
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if yes_rows[mid].0 <= *t_unix {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let Some((_, y)) = yes_rows.get(lo) else {
            continue;
        };
        let bid_error = (n.bid - (1.0 - y.ask)).abs();
        let ask_error = (n.ask - (1.0 - y.bid)).abs();
        out.push((bid_error, ask_error));
    }
    out
}

#[cfg(test)]
// Test-only: a wrong index or an unwrap on `None` is the test failing,
// which is the correct, intended outcome here.
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
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

    const L2_SAMPLE: &str = "\
# comment
1000.000,btc_4h,bid,0.55:100.0,0.54:200.0,0.53:50.0
1000.000,btc_4h,ask,0.57:80.0,0.58:120.0
1000.500,btc_1h,bid,0.20:40.0
1000.500,btc_1h,ask,0.22:30.0
not,a,valid,row
1003.000,btc_4h,bid,0.60:90.0,malformed,0.58:20.0
1003.000,btc_4h,ask,0.62:75.0
";

    #[test]
    fn l2_loads_only_the_requested_market_with_full_depth() {
        let snaps = load_recording_l2(L2_SAMPLE, "btc_4h");
        assert_eq!(snaps.len(), 2);
        assert!(snaps[0].t_s.abs() < 1e-9);
        assert_eq!(snaps[0].bids.len(), 3);
        assert_eq!(snaps[0].asks.len(), 2);
        assert!((snaps[0].bids[0].price - 0.55).abs() < 1e-12);
        assert!((snaps[0].bids[2].price - 0.53).abs() < 1e-12);
        // Second snapshot's bid row has one malformed level ("malformed")
        // between two good ones — skipped, not fatal to the row.
        assert_eq!(snaps[1].bids.len(), 2);
        assert!((snaps[1].t_s - 3.0).abs() < 1e-9);
    }

    #[test]
    fn l2_a_different_market_gets_its_own_snapshots() {
        let snaps = load_recording_l2(L2_SAMPLE, "btc_1h");
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].bids.len(), 1);
        assert_eq!(snaps[0].asks.len(), 1);
    }

    #[test]
    fn l2_unknown_market_is_empty_not_an_error() {
        assert!(load_recording_l2(L2_SAMPLE, "eth_4h").is_empty());
    }

    #[test]
    fn l2_lookup_holds_the_last_known_snapshot_forward() {
        let snaps = load_recording_l2(L2_SAMPLE, "btc_4h");
        let at = lookup_l2(&snaps, 5.0).unwrap();
        assert!((at.t_s - 3.0).abs() < 1e-9);
        let before_first = lookup_l2(&snaps, -1.0);
        assert!(before_first.is_none());
    }

    #[test]
    fn l2_set_book_places_every_level_not_just_the_touch() {
        let mut book = DenseBook::new(10_000);
        let snap = RecordedBookSnapshot {
            t_s: 0.0,
            bids: vec![
                RecordedLevel {
                    price: 0.55,
                    size: 100.0,
                },
                RecordedLevel {
                    price: 0.54,
                    size: 200.0,
                },
            ],
            asks: vec![RecordedLevel {
                price: 0.57,
                size: 80.0,
            }],
        };
        set_book_from_l2_snapshot(&mut book, &snap);
        assert_eq!(book.best_bid(), Some(Px::from_f64(0.55)));
        assert_eq!(
            book.size_at(Side::Bid, Px::from_f64(0.54)),
            qty_from_f64(200.0)
        );
        assert_eq!(book.best_ask(), Some(Px::from_f64(0.57)));
    }

    #[test]
    fn a_real_l2_recording_loads_and_populates_a_book_with_real_depth() {
        // Real full-depth data captured 2026-08-16 with tools/px-record's
        // L2 mode — see recordings/README.md. Proves `load_recording_l2`
        // and `set_book_from_l2_snapshot` work end to end against actual
        // venue responses, not just the hand-built fixture above.
        let raw = include_str!("../../../recordings/polymarket_l2_sample.csv");
        let snaps = load_recording_l2(raw, "btc-updown-5m-1786860300");
        assert!(
            snaps.len() > 10,
            "expected a substantial real L2 recording, got {} snapshots",
            snaps.len()
        );
        // Real depth, not just a touch: this market's book runs dozens of
        // levels deep on both sides (see recordings/README.md).
        let deepest = snaps.iter().map(|s| s.bids.len()).max().unwrap_or(0);
        assert!(deepest > 10, "expected multi-level depth, got {deepest}");

        let mut book = DenseBook::new(10_000);
        for snap in &snaps {
            set_book_from_l2_snapshot(&mut book, snap);
            assert!(book.best_bid().is_some());
            assert!(book.best_ask().is_some());
            assert!(!book.is_crossed(), "real book crossed at t_s={}", snap.t_s);
            // Depth beyond the touch must actually be there, not just the
            // first level — the whole point of L2 over `RecordedQuote`.
            let last_bid = snap.bids.last().unwrap();
            assert!(
                book.size_at(Side::Bid, Px::from_f64(last_bid.price)) > Qty::ZERO,
                "deepest recorded bid level did not make it into the book"
            );
        }
    }

    #[test]
    fn perfectly_complementary_quotes_have_zero_error() {
        let text = "\
1000.000,m,0.49,80,0.50,30
1000.000,m-no,0.50,30,0.51,80
1003.000,m,0.55,10,0.57,20
1003.000,m-no,0.43,20,0.45,10
";
        let errors = complementarity_error(text, "m", "m-no");
        assert_eq!(errors.len(), 2);
        for (bid_error, ask_error) in errors {
            assert!(bid_error < 1e-12, "bid_error {bid_error}");
            assert!(ask_error < 1e-12, "ask_error {ask_error}");
        }
    }

    #[test]
    fn a_real_deviation_from_complementarity_is_not_hidden() {
        // NO's book is 2c wider than YES's complement would predict on
        // both sides — the two venue-side books are not, in this
        // fixture, perfectly mirrored.
        let text = "\
1000.000,m,0.49,80,0.50,30
1000.000,m-no,0.48,30,0.53,80
";
        let errors = complementarity_error(text, "m", "m-no");
        assert_eq!(errors.len(), 1);
        let (bid_error, ask_error) = errors[0];
        assert!((bid_error - 0.02).abs() < 1e-9, "bid_error {bid_error}");
        assert!((ask_error - 0.02).abs() < 1e-9, "ask_error {ask_error}");
    }

    #[test]
    fn complementarity_error_matches_on_absolute_time_not_normalised_t_s() {
        // `no`'s single row is genuinely complementary to `yes`'s row at
        // the same real instant (t=1200) — but `no`'s recording starts
        // later than `yes`'s, so if this matched on each series' own
        // independently-zeroed `t_s` (as it would if it reused
        // `load_recording`'s output directly, where both series' first
        // row is `t_s = 0`) it would instead compare `no` against
        // `yes`'s *first* row (t=1000, `t_s = 0` in `yes`'s own
        // numbering) and report a large, entirely artefactual error.
        let text = "\
1000.000,m,0.90,1,0.91,1
1100.000,m,0.80,1,0.81,1
1200.000,m,0.50,1,0.51,1
1300.000,m,0.10,1,0.11,1
1250.000,m-no,0.49,1,0.50,1
";
        let errors = complementarity_error(text, "m", "m-no");
        assert_eq!(errors.len(), 1);
        // Correctly matched against the t=1200 YES row (0.50/0.51) — a
        // real, near-zero complementarity error — not the t=1000 row
        // (0.90/0.91) a `t_s`-based match would wrongly pick, which would
        // read as ~0.40 of error where none exists.
        let (bid_error, ask_error) = errors[0];
        assert!(bid_error < 1e-9, "bid_error {bid_error}");
        assert!(ask_error < 1e-9, "ask_error {ask_error}");
    }

    #[test]
    fn complementarity_error_before_any_yes_quote_is_skipped() {
        let text = "\
1000.000,m-no,0.50,30,0.51,80
1500.000,m,0.49,80,0.50,30
";
        assert!(complementarity_error(text, "m", "m-no").is_empty());
    }
}
