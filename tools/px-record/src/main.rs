//! Real, minimal Polymarket order-book recorder for
//! `px_engine::recording`.
//!
//! Polls Polymarket's public, unauthenticated Gamma + CLOB REST endpoints
//! for the currently active "Up or Down" crypto markets and appends a
//! best-bid/best-ask/size snapshot per market per poll to a CSV file, in
//! exactly the `t_unix,market,bid,bid_size,ask,ask_size` format
//! `px_engine::recording::load_recording` parses. It also polls
//! Bitstamp's public BTC/USD ticker in the same loop — Binance/Coinbase/
//! Kraken are unreachable from the sandbox this was built in, but
//! Bitstamp and CoinGecko are not — so a long recording session captures
//! a real reference price series *overlapping in time* with the real
//! venue quotes, which a calibration check on `SimConfig::venue_lag_s`
//! needs and no earlier capture in this repo had.
//!
//! Built for unattended, multi-hour runs, not just a short foreground
//! session: `duration_secs = 0` runs until Ctrl+C; the market list is
//! re-discovered periodically so a 5-minute market that expires mid-run
//! gets dropped and whatever replaced it gets picked up, rather than
//! polling a dead market for the rest of the session; a target that
//! fails several polls in a row is skipped until the next re-discovery
//! refreshes it, rather than hammering an endpoint that is not answering.
//!
//! Read-only: this never places an order and needs no credentials — every
//! endpoint polled here is a public market-data read.
//!
//! Usage: `cargo run --release -- [output_path] [duration_secs] [poll_interval_secs] [rediscover_interval_secs]`
//! Defaults: `recording.csv`, 300, 3, 180. `duration_secs = 0` means "run
//! until Ctrl+C".

use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GAMMA_EVENTS_URL: &str = "https://gamma-api.polymarket.com/events?active=true&closed=false&tag_slug=crypto&order=volume24hr&ascending=false&limit=100";
const CLOB_BOOK_URL: &str = "https://clob.polymarket.com/book";
const BITSTAMP_TICKER_URL: &str = "https://www.bitstamp.net/api/v2/ticker/btcusd/";
/// A target that has failed this many consecutive polls is skipped until
/// the next re-discovery cycle — see `FailureTracker`.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// One market worth polling: a short, stable name for the CSV `market`
/// column (its Polymarket event slug), and the CLOB token ids of both its
/// YES/"Up" and NO/"Down" outcomes. `no_token` is polled and written
/// under `{name}-no` in the same output files — see `main`'s comment on
/// why, this is what item 5 of the "what's left" list needs: checking
/// whether the two sides' independently-quoted books are actually
/// complementary, rather than assuming it the way `recording.rs`'s
/// `set_complementary_book_from_quote` currently has to.
#[derive(Debug, Clone)]
struct Target {
    name: String,
    yes_token: String,
    no_token: Option<String>,
}

/// Extracts every currently active "Up or Down" crypto market from a
/// Gamma `/events` response. Pure and unit-testable against a saved
/// response, unlike the live HTTP call around it — the same split
/// `parallax-venues::parse_kalshi_orderbook` uses for the same reason.
fn discover_targets(events: &Value) -> Vec<Target> {
    let mut out = Vec::new();
    let Some(events) = events.as_array() else {
        return out;
    };
    for event in events {
        let title = event.get("title").and_then(Value::as_str).unwrap_or("");
        if !title.to_lowercase().contains("up or down") {
            continue;
        }
        let slug = event.get("slug").and_then(Value::as_str).unwrap_or("event");
        let Some(markets) = event.get("markets").and_then(Value::as_array) else {
            continue;
        };
        let Some(market) = markets.first() else {
            continue;
        };
        let Some(token_ids_raw) = market.get("clobTokenIds").and_then(Value::as_str) else {
            continue;
        };
        let Ok(token_ids) = serde_json::from_str::<Vec<String>>(token_ids_raw) else {
            continue;
        };
        let Some(yes_token) = token_ids.first() else {
            continue;
        };
        out.push(Target {
            name: slug.to_string(),
            yes_token: yes_token.clone(),
            no_token: token_ids.get(1).cloned(),
        });
    }
    out
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Every level on one side, sorted best-first (descending for bids,
/// ascending for asks) — the CLOB response's own order is not documented
/// as sorted, so this sorts explicitly rather than trusting it. A
/// malformed individual level is dropped, not fatal to the rest.
fn all_levels(levels: &Value, descending: bool) -> Vec<(f64, f64)> {
    let Some(arr) = levels.as_array() else {
        return Vec::new();
    };
    let mut out: Vec<(f64, f64)> = arr
        .iter()
        .filter_map(|level| {
            let price: f64 = level.get("price")?.as_str()?.parse().ok()?;
            let size: f64 = level.get("size")?.as_str()?.parse().ok()?;
            if price.is_finite() && size.is_finite() {
                Some((price, size))
            } else {
                None
            }
        })
        .collect();
    out.sort_by(|a, b| {
        if descending {
            b.0.partial_cmp(&a.0)
        } else {
            a.0.partial_cmp(&b.0)
        }
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// `px_engine::recording::load_recording_l2`'s level-list format:
/// `price:size,price:size,...`.
fn format_levels(levels: &[(f64, f64)]) -> String {
    levels
        .iter()
        .map(|(p, s)| format!("{p}:{s}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Derives the L2-depth output path from the touch-only one: `x.csv` ->
/// `x.l2.csv`, or `x.l2.csv` appended if there is no extension to split
/// on. Keeps the CLI to one output argument rather than two.
fn l2_path(base: &str) -> String {
    match base.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.l2.{ext}"),
        None => format!("{base}.l2.csv"),
    }
}

/// Same derivation as `l2_path`, for the BTC/USD reference-price file.
fn ref_path(base: &str) -> String {
    match base.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.ref.{ext}"),
        None => format!("{base}.ref.csv"),
    }
}

/// Tracks a per-target consecutive-failure streak, same "streak, not a
/// single blip" shape as `parallax-cli::FeedHealthMonitor` in the sibling
/// repo: a fetch that fails once in a while is ordinary internet, and a
/// target that fails `MAX_CONSECUTIVE_FAILURES` times in a row is either
/// gone (expired, delisted) or genuinely down — either way, not worth
/// spending a request on every 3-second tick until the next re-discovery
/// cycle either drops it (if it is really gone) or confirms it is still
/// listed and worth retrying.
#[derive(Debug, Default)]
struct FailureTracker(HashMap<String, u32>);

impl FailureTracker {
    fn record_failure(&mut self, key: &str) {
        *self.0.entry(key.to_string()).or_insert(0) += 1;
    }

    fn record_success(&mut self, key: &str) {
        self.0.remove(key);
    }

    fn should_skip(&self, key: &str) -> bool {
        self.0
            .get(key)
            .is_some_and(|&n| n >= MAX_CONSECUTIVE_FAILURES)
    }

    /// Called on every re-discovery: a fresh target list deserves a
    /// fresh chance, not a streak carried over from whatever the old
    /// list's targets did.
    fn reset(&mut self) {
        self.0.clear();
    }
}

/// Bitstamp's public ticker: real bid/ask/last for BTC/USD, no API key.
/// `None` on a malformed response — the caller treats this the same as a
/// transient fetch failure, not a reason to stop recording the venue side.
fn parse_reference_ticker(v: &Value) -> Option<(f64, f64, f64)> {
    let bid: f64 = v.get("bid")?.as_str()?.parse().ok()?;
    let ask: f64 = v.get("ask")?.as_str()?.parse().ok()?;
    let last: f64 = v.get("last")?.as_str()?.parse().ok()?;
    if bid.is_finite() && ask.is_finite() && last.is_finite() {
        Some((bid, ask, last))
    } else {
        None
    }
}

/// Fetches one token's real book and appends it to both output files
/// under `market_label` — the one piece of per-token logic `main`'s loop
/// needs twice (YES, and optionally NO). Returns `Ok(true)` if a touch
/// row was written (both sides had at least one level), `Ok(false)` if
/// the book was one-sided (routine, not an error) or malformed, and
/// `Err` only on an actual request failure.
fn fetch_and_record(
    client: &reqwest::blocking::Client,
    token_id: &str,
    market_label: &str,
    file: &mut File,
    l2_file: &mut File,
) -> Result<bool, reqwest::Error> {
    let url = format!("{CLOB_BOOK_URL}?token_id={token_id}");
    let book: Value = client.get(&url).send()?.json()?;
    let ts = unix_now();
    let bids = all_levels(book.get("bids").unwrap_or(&Value::Null), true);
    let asks = all_levels(book.get("asks").unwrap_or(&Value::Null), false);
    let mut wrote_touch = false;
    if let (Some(&(bp, bs)), Some(&(ap, asz))) = (bids.first(), asks.first()) {
        let row = format!("{ts:.3},{market_label},{bp},{bs},{ap},{asz}\n");
        wrote_touch = file.write_all(row.as_bytes()).is_ok();
    }
    if !bids.is_empty() {
        let row = format!("{ts:.3},{market_label},bid,{}\n", format_levels(&bids));
        let _ = l2_file.write_all(row.as_bytes());
    }
    if !asks.is_empty() {
        let row = format!("{ts:.3},{market_label},ask,{}\n", format_levels(&asks));
        let _ = l2_file.write_all(row.as_bytes());
    }
    Ok(wrote_touch)
}

/// Fetches and parses the active-markets list. `Vec::new()` on any
/// failure (network error, malformed JSON) — the caller decides what an
/// empty discovery means (fatal at startup, "keep the old list" on a
/// periodic re-discovery), this just reports what happened.
fn discover(client: &reqwest::blocking::Client) -> Vec<Target> {
    match client.get(GAMMA_EVENTS_URL).send().and_then(|r| r.json()) {
        Ok(events) => discover_targets(&events),
        Err(e) => {
            eprintln!("failed to fetch events: {e}");
            Vec::new()
        }
    }
}

/// How many markets in `fresh` are new relative to `old`, and how many
/// of `old`'s have dropped out of `fresh` — pure and testable, unlike
/// the live re-discovery call around it. Compares by `name` only: a
/// market whose token ids changed while its slug stayed the same (should
/// not happen in practice) is not treated as added-and-dropped.
fn diff_targets(old: &[Target], fresh: &[Target]) -> (usize, usize) {
    let old_names: std::collections::HashSet<&str> = old.iter().map(|t| t.name.as_str()).collect();
    let fresh_names: std::collections::HashSet<&str> =
        fresh.iter().map(|t| t.name.as_str()).collect();
    let added = fresh_names.difference(&old_names).count();
    let dropped = old_names.difference(&fresh_names).count();
    (added, dropped)
}

fn open_append(path: &str, header: &str) -> File {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| {
            eprintln!("failed to open {path}: {e}");
            std::process::exit(1);
        });
    let _ = writeln!(f, "{header}");
    f
}

/// Polls both sides of one target (YES, and NO if listed), skipping a
/// side that has already failed `MAX_CONSECUTIVE_FAILURES` times in a
/// row this cycle — see `FailureTracker`.
#[allow(clippy::too_many_arguments)]
fn poll_target(
    client: &reqwest::blocking::Client,
    t: &Target,
    file: &mut File,
    l2_file: &mut File,
    failures: &mut FailureTracker,
    n: &mut u64,
    errors: &mut u64,
) {
    let mut sides: Vec<(String, &str)> = vec![(t.name.clone(), t.yes_token.as_str())];
    if let Some(no_token) = &t.no_token {
        sides.push((format!("{}-no", t.name), no_token.as_str()));
    }
    for (label, token) in sides {
        if failures.should_skip(&label) {
            continue;
        }
        match fetch_and_record(client, token, &label, file, l2_file) {
            Ok(true) => {
                *n += 1;
                failures.record_success(&label);
            }
            Ok(false) => failures.record_success(&label),
            Err(e) => {
                *errors += 1;
                failures.record_failure(&label);
                eprintln!("fetch failed for {label}: {e}");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "recording.csv".to_string());
    let duration_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let interval_secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rediscover_secs: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(180);
    let run_forever = duration_secs == 0;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("px-record/0.1 (research tool, read-only)")
        .build()
        .expect("failed to build HTTP client");

    println!("Discovering active Polymarket 'Up or Down' crypto markets...");
    let mut targets = discover(&client);
    if targets.is_empty() {
        eprintln!("no active 'Up or Down' crypto markets found right now — nothing to record");
        std::process::exit(1);
    }
    println!("Recording {} market(s):", targets.len());
    for t in &targets {
        println!("  {} ({})", t.name, t.yes_token);
    }

    let l2_out_path = l2_path(&out_path);
    let ref_out_path = ref_path(&out_path);
    let duration_desc = if run_forever {
        "until Ctrl+C".to_string()
    } else {
        format!("for {duration_secs}s")
    };
    println!(
        "Writing touch quotes to {out_path}, full depth to {l2_out_path},\nBTC/USD reference to {ref_out_path}, every {interval_secs}s {duration_desc}.\nRe-discovering the market list every {rediscover_secs}s. Read-only — no order\nis ever placed, no credentials required."
    );

    let mut file = open_append(&out_path, "# t_unix,market,bid,bid_size,ask,ask_size");
    let mut l2_file = open_append(
        &l2_out_path,
        "# t_unix,market,side,price:size,price:size,...",
    );
    let mut ref_file = open_append(&ref_out_path, "# t_unix,bid,ask,last");

    // Ctrl+C sets a flag the loop checks between ticks, rather than
    // killing the process outright — an unattended multi-hour run should
    // get to flush and print a summary on the way out, the same
    // reasoning `parallax-cli::record_main`'s `tokio::signal::ctrl_c()`
    // handling uses in the sibling repo.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        if let Err(e) = ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)) {
            eprintln!(
                "warning: could not install a Ctrl+C handler ({e}) — stop with a process kill instead"
            );
        }
    }

    let start = Instant::now();
    let mut last_discovery = Instant::now();
    let mut failures = FailureTracker::default();
    let mut n = 0u64;
    let mut ref_n = 0u64;
    let mut errors = 0u64;

    while !stop.load(Ordering::SeqCst)
        && (run_forever || start.elapsed() < Duration::from_secs(duration_secs))
    {
        let tick_start = Instant::now();

        if last_discovery.elapsed() >= Duration::from_secs(rediscover_secs) {
            let fresh = discover(&client);
            if fresh.is_empty() {
                eprintln!("re-discovery found nothing active — keeping the current target list");
            } else {
                let (added, dropped) = diff_targets(&targets, &fresh);
                if added > 0 || dropped > 0 {
                    println!(
                        "re-discovery: {added} market(s) added, {dropped} expired or dropped, {} unchanged",
                        fresh.len().saturating_sub(added)
                    );
                }
                targets = fresh;
                // A freshly re-discovered list deserves a fresh chance —
                // do not carry a failure streak from the old list's
                // targets onto whatever now occupies the same slot.
                failures.reset();
            }
            last_discovery = Instant::now();
        }

        for t in &targets {
            poll_target(
                &client,
                t,
                &mut file,
                &mut l2_file,
                &mut failures,
                &mut n,
                &mut errors,
            );
        }

        match client
            .get(BITSTAMP_TICKER_URL)
            .send()
            .and_then(|r| r.json::<Value>())
        {
            Ok(v) => {
                if let Some((bid, ask, last)) = parse_reference_ticker(&v) {
                    let row = format!("{:.3},{bid},{ask},{last}\n", unix_now());
                    if ref_file.write_all(row.as_bytes()).is_ok() {
                        ref_n += 1;
                    }
                }
            }
            Err(e) => eprintln!("reference-price fetch failed: {e}"),
        }

        let _ = file.flush();
        let _ = l2_file.flush();
        let _ = ref_file.flush();
        println!(
            "recorded so far: {n} venue snapshots, {ref_n} reference ticks, {errors} fetch errors"
        );

        let elapsed = tick_start.elapsed();
        if elapsed < Duration::from_secs(interval_secs) {
            std::thread::sleep(Duration::from_secs(interval_secs) - elapsed);
        }
    }

    println!(
        "Done. {n} venue snapshots and {ref_n} reference ticks written ({errors} fetch errors)."
    );
    println!(
        "Load venue quotes with px_engine::recording::load_recording, full depth from\n{l2_out_path} with load_recording_l2, BTC/USD reference from {ref_out_path} as\nplain (t_unix, bid, ask, last) rows — filter venue files on the market slug\nprinted above."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events() -> Value {
        serde_json::json!([
            {
                "title": "Bitcoin Up or Down - August 15, 8:00PM-12:00AM ET",
                "slug": "btc-updown-4h-1786838400",
                "markets": [
                    {"clobTokenIds": "[\"111\",\"222\"]"}
                ]
            },
            {
                "title": "Will X win the election?",
                "slug": "x-election",
                "markets": [
                    {"clobTokenIds": "[\"333\",\"444\"]"}
                ]
            },
            {
                "title": "Ethereum Up or Down - August 15, 8:00PM-12:00AM ET",
                "slug": "eth-updown-4h-1786838400",
                "markets": [
                    {"clobTokenIds": "[\"555\",\"666\"]"}
                ]
            }
        ])
    }

    #[test]
    fn discovers_only_up_or_down_markets_and_their_yes_token() {
        let targets = discover_targets(&sample_events());
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].name, "btc-updown-4h-1786838400");
        assert_eq!(targets[0].yes_token, "111");
        assert_eq!(targets[0].no_token.as_deref(), Some("222"));
        assert_eq!(targets[1].name, "eth-updown-4h-1786838400");
        assert_eq!(targets[1].yes_token, "555");
        assert_eq!(targets[1].no_token.as_deref(), Some("666"));
    }

    #[test]
    fn a_single_outcome_token_leaves_no_token_absent_not_fatal() {
        let events = serde_json::json!([
            {
                "title": "BTC Up or Down",
                "slug": "solo",
                "markets": [{"clobTokenIds": "[\"only-one\"]"}]
            }
        ]);
        let targets = discover_targets(&events);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].yes_token, "only-one");
        assert!(targets[0].no_token.is_none());
    }

    #[test]
    fn missing_or_malformed_fields_are_skipped_not_fatal() {
        let events = serde_json::json!([
            {"title": "BTC Up or Down", "slug": "a", "markets": []},
            {"title": "BTC Up or Down", "slug": "b"},
            {"title": "BTC Up or Down", "slug": "c", "markets": [{"clobTokenIds": "not json"}]},
            {"title": "BTC Up or Down", "slug": "d", "markets": [{}]},
        ]);
        assert!(discover_targets(&events).is_empty());
    }

    #[test]
    fn empty_events_array_yields_no_targets() {
        assert!(discover_targets(&serde_json::json!([])).is_empty());
    }

    #[test]
    fn all_levels_of_an_empty_side_is_empty() {
        assert!(all_levels(&serde_json::json!([]), true).is_empty());
        assert!(all_levels(&Value::Null, true).is_empty());
    }

    #[test]
    fn all_levels_sorts_bids_descending_and_asks_ascending() {
        let levels = serde_json::json!([
            {"price": "0.52", "size": "5"},
            {"price": "0.60", "size": "20"},
            {"price": "0.55", "size": "10"},
        ]);
        assert_eq!(
            all_levels(&levels, true),
            vec![(0.60, 20.0), (0.55, 10.0), (0.52, 5.0)]
        );
        assert_eq!(
            all_levels(&levels, false),
            vec![(0.52, 5.0), (0.55, 10.0), (0.60, 20.0)]
        );
    }

    #[test]
    fn all_levels_drops_malformed_entries_not_the_whole_side() {
        let levels = serde_json::json!([
            {"price": "0.52", "size": "5"},
            {"price": "not a number", "size": "10"},
            {"size": "10"},
            {"price": "0.55", "size": "10"},
        ]);
        assert_eq!(all_levels(&levels, true), vec![(0.55, 10.0), (0.52, 5.0)]);
    }

    #[test]
    fn format_levels_matches_load_recording_l2s_expected_syntax() {
        assert_eq!(
            format_levels(&[(0.55, 100.0), (0.54, 200.0)]),
            "0.55:100,0.54:200"
        );
        assert_eq!(format_levels(&[]), "");
    }

    #[test]
    fn l2_path_inserts_before_the_extension() {
        assert_eq!(l2_path("recording.csv"), "recording.l2.csv");
        assert_eq!(l2_path("dir/recording.csv"), "dir/recording.l2.csv");
        assert_eq!(l2_path("no_extension"), "no_extension.l2.csv");
    }

    #[test]
    fn ref_path_inserts_before_the_extension() {
        assert_eq!(ref_path("recording.csv"), "recording.ref.csv");
        assert_eq!(ref_path("no_extension"), "no_extension.ref.csv");
    }

    fn target(name: &str) -> Target {
        Target {
            name: name.to_string(),
            yes_token: "t".to_string(),
            no_token: None,
        }
    }

    #[test]
    fn diff_targets_counts_additions_and_drops() {
        let old = vec![target("a"), target("b"), target("c")];
        let fresh = vec![target("b"), target("c"), target("d")];
        // "a" dropped, "d" added, "b"/"c" unchanged.
        assert_eq!(diff_targets(&old, &fresh), (1, 1));
    }

    #[test]
    fn diff_targets_of_identical_lists_is_zero_and_zero() {
        let list = vec![target("a"), target("b")];
        assert_eq!(diff_targets(&list, &list.clone()), (0, 0));
    }

    #[test]
    fn diff_targets_from_empty_counts_everything_as_added() {
        let fresh = vec![target("a"), target("b")];
        assert_eq!(diff_targets(&[], &fresh), (2, 0));
    }

    #[test]
    fn failure_tracker_skips_only_after_the_threshold() {
        let mut f = FailureTracker::default();
        for _ in 0..MAX_CONSECUTIVE_FAILURES - 1 {
            f.record_failure("m");
        }
        assert!(!f.should_skip("m"));
        f.record_failure("m");
        assert!(f.should_skip("m"));
    }

    #[test]
    fn failure_tracker_success_resets_the_streak() {
        let mut f = FailureTracker::default();
        for _ in 0..MAX_CONSECUTIVE_FAILURES {
            f.record_failure("m");
        }
        assert!(f.should_skip("m"));
        f.record_success("m");
        assert!(!f.should_skip("m"));
    }

    #[test]
    fn failure_tracker_reset_clears_every_target() {
        let mut f = FailureTracker::default();
        for _ in 0..MAX_CONSECUTIVE_FAILURES {
            f.record_failure("m");
        }
        f.reset();
        assert!(!f.should_skip("m"));
    }

    #[test]
    fn failure_tracker_tracks_targets_independently() {
        let mut f = FailureTracker::default();
        for _ in 0..MAX_CONSECUTIVE_FAILURES {
            f.record_failure("a");
        }
        assert!(f.should_skip("a"));
        assert!(!f.should_skip("b"));
    }

    #[test]
    fn parse_reference_ticker_reads_bid_ask_last() {
        let v = serde_json::json!({"bid": "63026.91", "ask": "63026.92", "last": "63026.91"});
        assert_eq!(
            parse_reference_ticker(&v),
            Some((63026.91, 63026.92, 63026.91))
        );
    }

    #[test]
    fn parse_reference_ticker_rejects_malformed_fields() {
        assert_eq!(parse_reference_ticker(&serde_json::json!({})), None);
        let v = serde_json::json!({"bid": "not a number", "ask": "1", "last": "1"});
        assert_eq!(parse_reference_ticker(&v), None);
    }
}
