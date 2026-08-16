//! Real, minimal Polymarket order-book recorder for
//! `px_engine::recording`.
//!
//! Polls Polymarket's public, unauthenticated Gamma + CLOB REST endpoints
//! for the currently active "Up or Down" crypto markets and appends a
//! best-bid/best-ask/size snapshot per market per poll to a CSV file, in
//! exactly the `t_unix,market,bid,bid_size,ask,ask_size` format
//! `px_engine::recording::load_recording` parses.
//!
//! Read-only: this never places an order and needs no credentials — both
//! endpoints polled here are public market-data reads.
//!
//! Usage: `cargo run --release -- [output_path] [duration_secs] [poll_interval_secs]`
//! Defaults: `recording.csv`, 300, 3.

use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GAMMA_EVENTS_URL: &str = "https://gamma-api.polymarket.com/events?active=true&closed=false&tag_slug=crypto&order=volume24hr&ascending=false&limit=100";
const CLOB_BOOK_URL: &str = "https://clob.polymarket.com/book";

/// One market worth polling: a short, stable name for the CSV `market`
/// column (its Polymarket event slug), and the CLOB token id of its
/// YES/"Up" outcome.
#[derive(Debug, Clone)]
struct Target {
    name: String,
    yes_token: String,
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

/// Best bid (max price) / best ask (min price) from a CLOB `/book`
/// response's `bids`/`asks` level arrays. `None` if the side is empty or
/// malformed — a momentarily one-sided book is routine on a real live
/// book, not an error to abort a whole polling pass over.
fn best_level(levels: &Value, want_max: bool) -> Option<(f64, f64)> {
    let arr = levels.as_array()?;
    let mut best: Option<(f64, f64)> = None;
    for level in arr {
        let price: f64 = level.get("price")?.as_str()?.parse().ok()?;
        let size: f64 = level.get("size")?.as_str()?.parse().ok()?;
        best = match best {
            None => Some((price, size)),
            Some((bp, _)) if want_max && price > bp => Some((price, size)),
            Some((bp, _)) if !want_max && price < bp => Some((price, size)),
            other => other,
        };
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "recording.csv".to_string());
    let duration_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let interval_secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("px-record/0.1 (research tool, read-only)")
        .build()
        .expect("failed to build HTTP client");

    println!("Discovering active Polymarket 'Up or Down' crypto markets...");
    let events: Value = match client.get(GAMMA_EVENTS_URL).send().and_then(|r| r.json()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to fetch events: {e}");
            std::process::exit(1);
        }
    };
    let targets = discover_targets(&events);
    if targets.is_empty() {
        eprintln!("no active 'Up or Down' crypto markets found right now — nothing to record");
        std::process::exit(1);
    }
    println!("Recording {} market(s):", targets.len());
    for t in &targets {
        println!("  {} ({})", t.name, t.yes_token);
    }
    println!(
        "Writing to {out_path} every {interval_secs}s for {duration_secs}s. Read-only — no order is ever placed, no credentials required."
    );

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap_or_else(|e| {
            eprintln!("failed to open {out_path}: {e}");
            std::process::exit(1);
        });
    let _ = writeln!(file, "# t_unix,market,bid,bid_size,ask,ask_size");

    let start = Instant::now();
    let mut n = 0u64;
    let mut errors = 0u64;
    while start.elapsed() < Duration::from_secs(duration_secs) {
        let tick_start = Instant::now();
        for t in &targets {
            let url = format!("{CLOB_BOOK_URL}?token_id={}", t.yes_token);
            match client.get(&url).send().and_then(|r| r.json::<Value>()) {
                Ok(book) => {
                    let bid = best_level(book.get("bids").unwrap_or(&Value::Null), true);
                    let ask = best_level(book.get("asks").unwrap_or(&Value::Null), false);
                    if let (Some((bp, bs)), Some((ap, asz))) = (bid, ask) {
                        let row = format!(
                            "{:.3},{},{},{},{},{}\n",
                            unix_now(),
                            t.name,
                            bp,
                            bs,
                            ap,
                            asz
                        );
                        if file.write_all(row.as_bytes()).is_ok() {
                            n += 1;
                        }
                    }
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("fetch failed for {}: {e}", t.name);
                }
            }
        }
        let _ = file.flush();
        println!("recorded so far: {n} snapshots, {errors} fetch errors");
        let elapsed = tick_start.elapsed();
        if elapsed < Duration::from_secs(interval_secs) {
            std::thread::sleep(Duration::from_secs(interval_secs) - elapsed);
        }
    }
    println!("Done. {n} snapshots written to {out_path} ({errors} fetch errors).");
    println!(
        "Load with px_engine::recording::load_recording, filtering on the market slug printed above."
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
        assert_eq!(targets[1].name, "eth-updown-4h-1786838400");
        assert_eq!(targets[1].yes_token, "555");
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
    fn best_level_picks_max_for_bids_and_min_for_asks() {
        let levels = serde_json::json!([
            {"price": "0.55", "size": "10"},
            {"price": "0.60", "size": "20"},
            {"price": "0.52", "size": "5"},
        ]);
        assert_eq!(best_level(&levels, true), Some((0.60, 20.0)));
        assert_eq!(best_level(&levels, false), Some((0.52, 5.0)));
    }

    #[test]
    fn best_level_of_an_empty_side_is_none() {
        assert_eq!(best_level(&serde_json::json!([]), true), None);
        assert_eq!(best_level(&Value::Null, true), None);
    }
}
