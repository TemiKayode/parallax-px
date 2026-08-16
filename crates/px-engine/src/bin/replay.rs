//! Scenario replay runner.
//!
//! ```text
//!   cargo run --release --bin px-replay
//! ```
//!
//! Runs a battery of scenarios, then sweeps the three parameters that decide
//! whether the strategy is viable at all:
//!
//!   * **informed fraction** — how much of the counterparty flow knows more
//!     than we do. This is a property of the venue's participant mix, and it is
//!     the hard viability condition. No amount of engineering beats it.
//!   * **venue lag** — how stale the resting book is. This is our edge source.
//!   * **execution latency** — how long our own round trip takes. This says
//!     what a faster path is *worth*, in dollars, instead of assuming.
//!
//! Finally it reports a seed distribution, because one backtest is one draw
//! from a wide distribution and reporting the draw as the result is the oldest
//! mistake in the field.

use px_engine::replay::{
    quantiles, run, seed_distribution, sweep_informed_fraction, sweep_latency, sweep_venue_lag,
    Outage, Report, Shock, SimConfig, VenueQuoteSource,
};

fn header() {
    println!(
        "{:<20} {:>10} {:>9} {:>6} {:>11} {:>11} {:>6} {:>6}",
        "scenario", "pnl($)", "fees($)", "fee/L", "aggr shrs", "pasv shrs", "uptime", "endPos"
    );
    println!("{}", "-".repeat(96));
}

fn line(name: &str, r: &Report) {
    println!(
        "{:<20} {:>10.2} {:>9.2} {:>6.2} {:>11.0} {:>11.0} {:>5.1}% {:>6.0}",
        name,
        r.pnl_dollars(),
        r.fees_paid as f64 / 1e6,
        r.fee_share_of_loss(),
        r.aggressive_shares as f64 / 1e6,
        r.passive_shares as f64 / 1e6,
        r.quote_uptime * 100.0,
        r.final_position as f64 / 1e6,
    );
}

fn bar(v: f64, scale: f64) -> String {
    if v >= 0.0 {
        format!("{:>20}|{}", "", "#".repeat(((v / scale) as usize).min(40)))
    } else {
        let n = (((-v) / scale) as usize).min(20);
        format!("{:>20}|", "-".repeat(n))
    }
}

fn main() {
    println!("Parallax replay harness\n");
    println!("P&L in dollars. Mark-out in micro-dollars per share (positive = adverse).");
    println!("PF = profit factor. miss = aggressive orders that arrived to find the price gone.\n");

    header();

    let base = SimConfig::default();
    line("calm market", &run(&base));

    let mut busy = SimConfig::default();
    busy.flow_per_s = 5.0;
    line("heavy flow", &run(&busy));

    let mut shocked = SimConfig::default();
    shocked.shocks = vec![Shock {
        at_s: 150.0,
        magnitude: 0.006,
    }];
    line("news shock +60bp", &run(&shocked));

    let mut hostile = SimConfig::default();
    hostile.shocks = (0..10)
        .map(|i| Shock {
            at_s: 30.0 + i as f64 * 25.0,
            magnitude: if i % 2 == 0 { 0.004 } else { -0.004 },
        })
        .collect();
    line("whipsaw tape", &run(&hostile));

    let mut outage = SimConfig::default();
    outage.outages = vec![Outage {
        feed_index: 0,
        start_s: 100.0,
        end_s: 160.0,
    }];
    line("60s feed outage", &run(&outage));

    let mut fast = SimConfig::default();
    fast.venue_lag_s = 0.0;
    fast.venue_noise = 0.0005;
    line("efficient venue", &run(&fast));

    let mut endgame = SimConfig::default();
    endgame.venue_lag_s = 0.8;
    line("twap endgame", &run(&endgame));

    let mut geo = SimConfig::default();
    geo.category = px_core::Category::Geopolitics;
    line("no-fee category", &run(&geo));

    let mut benign = SimConfig::default();
    benign.informed_fraction = 0.15;
    benign.flow_per_s = 4.0;
    line("mostly noise flow", &run(&benign));

    // ---------------------------------------------------------------
    // Sweep 1: informed fraction. The viability condition.
    // ---------------------------------------------------------------
    println!("\n\nInformed fraction of counterparty flow — the viability condition\n");
    let sweep_base = SimConfig {
        flow_per_s: 4.0,
        ..Default::default()
    };
    for (f, pnl) in sweep_informed_fraction(&sweep_base, &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    {
        println!(
            "  informed {:>4.0}%   pnl {:>9.2}  {}",
            f * 100.0,
            pnl,
            bar(pnl, 20.0)
        );
    }

    // ---------------------------------------------------------------
    // Sweep 2: venue lag. Our edge source.
    // ---------------------------------------------------------------
    println!("\n\nVenue lag — how stale the resting book is\n");
    for (lag, pnl) in sweep_venue_lag(&sweep_base, &[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]) {
        println!(
            "  lag {:>5.2}s      pnl {:>9.2}  {}",
            lag,
            pnl,
            bar(pnl, 20.0)
        );
    }

    // ---------------------------------------------------------------
    // Sweep 3: our own latency. What speed is worth.
    // ---------------------------------------------------------------
    println!("\n\nOur execution latency — what a faster path is actually worth\n");
    for (l, pnl) in sweep_latency(&sweep_base, &[0.000_01, 0.001, 0.010, 0.025, 0.100, 0.500]) {
        println!(
            "  latency {:>7.2}ms  pnl {:>9.2}  {}",
            l * 1000.0,
            pnl,
            bar(pnl, 20.0)
        );
    }

    // ---------------------------------------------------------------
    // Seed distribution. One run is one draw.
    // ---------------------------------------------------------------
    println!("\n\nSeed distribution (40 seeds, default config)\n");
    let dist = seed_distribution(&SimConfig::default(), 40);
    let (p10, p50, p90) = quantiles(&dist);
    println!("  worst  {:>9.2}", dist[0]);
    println!("  p10    {:>9.2}", p10);
    println!("  median {:>9.2}", p50);
    println!("  p90    {:>9.2}", p90);
    println!("  best   {:>9.2}", dist[dist.len() - 1]);
    let positive = dist.iter().filter(|&&x| x > 0.0).count();
    println!("  seeds profitable: {}/{}", positive, dist.len());

    // ---------------------------------------------------------------
    // The question that decides everything.
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("DOES THE MODEL BEAT THE VENUE MID?");
    println!("{}\n", "=".repeat(72));
    println!("Scored on forecasts alone — no orders, no execution, no capital.");
    println!("If the skill score is not positive here, no amount of execution");
    println!("engineering downstream can manufacture an edge.\n");

    // Pool forecasts across many seeds: one session is far too few resolved
    // forecasts to say anything, and 60 clusters barely clears has_edge()'s
    // own n_clusters >= 20 gate — nowhere near enough to resolve a genuinely
    // inconclusive result (clustered |t| ~1, well short of the |t| > 2 bar
    // in either direction). 500 clusters tightens the cluster-robust SE by
    // roughly sqrt(500/60) ~ 2.9x over the original run.
    const N_SEEDS: u64 = 500;
    let mut pooled = px_score::Scorer::new();
    for i in 0..N_SEEDS {
        let mut c = SimConfig::default();
        c.seed = 0x5EED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for r in run(&c).scorecard_inputs {
            pooled.record(r);
        }
    }
    let card = pooled.score();
    print!("{}", px_score::report(&card));

    // ---------------------------------------------------------------
    // DOES THE MODEL BEAT THE VENUE MID? (real recorded data, not synthetic)
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("DOES THE MODEL BEAT THE VENUE MID? (real recorded data)");
    println!("{}\n", "=".repeat(72));
    println!(
        "Same question, scored against tools/px-record's real captured venue quotes\ninstead of a synthetic venue — see recordings/README.md for what was captured."
    );
    println!(
        "The reference price feeding the model's own belief is still synthetic GBM\n(recording.rs's module doc explains why); only the venue side is real here.\n"
    );
    {
        // Each entry is one genuinely independent real market instance —
        // the actual unit of evidence, not a seed. `cluster_id` is
        // assigned per *market*, not per run, specifically so pooling
        // several recordings' worth of real data does not repeat the
        // mistake this whole feature exists to catch: treating repeated
        // or correlated readings as independent ones.
        //
        // `real_outcome` is the market's *actual* resolution — fetched
        // from Polymarket's Gamma API after both markets closed
        // (`umaResolutionStatus: "resolved"`, `outcomePrices: ["1","0"]`
        // for `["Up","Down"]` on both). This matters more than it looks:
        // `run()`'s own `outcome_yes` is computed from the *synthetic*
        // reference path's settlement TWAP against `cfg.strike`, which
        // has nothing to do with what a replayed real market actually
        // did. Scoring against that synthetic outcome instead of the real
        // one was tried first here and produced a base rate of 0% (both
        // markets reading as resolved NO) against real quotes that were
        // visibly trading up near 0.98-0.999 — the giveaway that the
        // outcome, not just the forecast, needs to be real too.
        let real_markets: &[(&str, u64, bool)] = &[("btc_4h", 1, true), ("btc_1h", 2, true)];
        let raw = include_str!("../../../../recordings/polymarket_sample.csv");
        let mut pooled_real = px_score::Scorer::new();
        let mut n_markets = 0usize;
        for (market, cluster_id, real_outcome) in real_markets {
            let quotes = px_engine::recording::load_recording(raw, market);
            if quotes.len() < 20 {
                println!(
                    "  skipping {market}: only {} recorded rows, too thin to replay",
                    quotes.len()
                );
                continue;
            }
            let span_s = quotes.last().map(|q| q.t_s).unwrap_or(0.0);
            let mut cfg = SimConfig::default();
            cfg.duration_s = span_s;
            cfg.expiry_s = span_s;
            cfg.venue_quotes = VenueQuoteSource::Recorded(quotes);
            let r = run(&cfg);
            for mut f in r.scorecard_inputs {
                f.forecast.cluster_id = *cluster_id;
                f.outcome = *real_outcome;
                pooled_real.record(f);
            }
            n_markets += 1;
            println!(
                "  {market}: {:.0}s replayed, {} fills, {} forecasts logged, real outcome = {}",
                span_s,
                r.fills.len(),
                r.stats.ticks,
                if *real_outcome { "Up" } else { "Down" }
            );
        }
        println!("\n{n_markets} real market instance(s) pooled below.\n");
        let real_card = pooled_real.score();
        print!("{}", px_score::report(&real_card));
        println!(
            "\n(n_clusters here is the count of *distinct recorded market instances*, not\nseeds — {n_markets} is nowhere near has_edge()'s own n_clusters >= 20 bar, so this\nis a real, honestly-reported measurement, not a significance claim. It is what\nrecording more independent real markets — the actual open item — would build on.)"
        );
        if real_card.skill_score < -1.0 {
            println!(
                "\nRead this skill score carefully: it is NOT evidence the strategy has no edge.\nThe model's own belief here is still computed from the synthetic GBM reference\npath, which has no relationship to what BTC actually did during this real\nwindow — this number is what happens when you score a forecast against a real\noutcome after disconnecting it from the real world first. It quantifies why\nthe synthetic reference feed is not a nice-to-have: until it is replaced with\na real one, no comparison against real venue quotes or real outcomes means\nwhat it looks like it means."
            );
        }
    }

    // ---------------------------------------------------------------
    // Is SimConfig::venue_depth calibrated against a real book?
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("IS venue_depth CALIBRATED AGAINST A REAL BOOK?");
    println!("{}\n", "=".repeat(72));
    {
        let raw_l2 = include_str!("../../../../recordings/polymarket_l2_sample.csv");
        let snaps = px_engine::recording::load_recording_l2(raw_l2, "btc-updown-5m-1786860300");
        if snaps.is_empty() {
            println!("  no L2 recording available — skipped.");
        } else {
            // Depth resting within the assumed half-spread's reach of the
            // touch, each side, averaged over every real snapshot — the
            // real-world number `SimConfig::default().venue_depth`
            // (400 shares) stands in for.
            let half_spread_px = SimConfig::default().venue_half_spread as f64 / 1_000_000.0;
            let mut bid_depth_sum = 0.0;
            let mut ask_depth_sum = 0.0;
            for snap in &snaps {
                let Some(best_bid) = snap.bids.first().map(|l| l.price) else {
                    continue;
                };
                let Some(best_ask) = snap.asks.first().map(|l| l.price) else {
                    continue;
                };
                bid_depth_sum += snap
                    .bids
                    .iter()
                    .filter(|l| best_bid - l.price <= half_spread_px)
                    .map(|l| l.size)
                    .sum::<f64>();
                ask_depth_sum += snap
                    .asks
                    .iter()
                    .filter(|l| l.price - best_ask <= half_spread_px)
                    .map(|l| l.size)
                    .sum::<f64>();
            }
            let n = snaps.len() as f64;
            let assumed = SimConfig::default().venue_depth.as_f64();
            println!(
                "  {} real snapshots of btc-updown-5m-1786860300 (see recordings/README.md)",
                snaps.len()
            );
            println!(
                "  assumed venue_depth (SimConfig::default):  {assumed:>10.1} shares per side"
            );
            println!(
                "  observed, within {:.0}c of the touch:  bid {:>10.1}   ask {:>10.1}  (mean per snapshot)",
                half_spread_px * 100.0,
                bid_depth_sum / n,
                ask_depth_sum / n
            );
            println!(
                "\n  Real resting size near the touch on a genuinely thin, low-volume market\n  varies enormously snapshot to snapshot (a handful of shares to tens of\n  thousands) — a single scalar `venue_depth` cannot represent that range,\n  which is itself the finding: queue-position calibration needs a\n  distribution, not a constant, and now there is real data to build one\n  from instead of guessing at it."
            );
        }
    }

    // ---------------------------------------------------------------
    // Are YES and NO books actually complementary in practice?
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("ARE YES AND NO REAL BOOKS ACTUALLY COMPLEMENTARY?");
    println!("{}\n", "=".repeat(72));
    println!(
        "recording.rs's set_complementary_book_from_quote assumes NO's book is exactly\n1 minus YES's — checked here against tools/px-record's -no output, both sides\npolled independently for the same markets (recordings/README.md).\n"
    );
    {
        let raw = include_str!("../../../../recordings/polymarket_yesno_sample.csv");
        let markets = [
            "btc-updown-4h-1786852800",
            "eth-updown-4h-1786852800",
            "sol-updown-4h-1786852800",
            "xrp-updown-4h-1786852800",
            "btc-updown-5m-1786860300",
            "btc-updown-15m-1786858200",
            "bitcoin-up-or-down-august-16-2026-1am-et",
            "ethereum-up-or-down-august-16-2026-1am-et",
            "solana-up-or-down-august-16-2026-1am-et",
            "xrp-up-or-down-august-16-2026-1am-et",
        ];
        let mut all_errors: Vec<(f64, f64)> = Vec::new();
        for market in markets {
            let no_market = format!("{market}-no");
            let errors = px_engine::recording::complementarity_error(raw, market, &no_market);
            if !errors.is_empty() {
                all_errors.extend(errors);
            }
        }
        if all_errors.is_empty() {
            println!("  no matched YES/NO pairs available — skipped.");
        } else {
            let n = all_errors.len() as f64;
            let mean_bid = all_errors.iter().map(|(b, _)| b).sum::<f64>() / n;
            let mean_ask = all_errors.iter().map(|(_, a)| a).sum::<f64>() / n;
            let max_bid = all_errors.iter().map(|(b, _)| *b).fold(0.0, f64::max);
            let max_ask = all_errors.iter().map(|(_, a)| *a).fold(0.0, f64::max);
            println!(
                "  {} matched real YES/NO snapshot pairs across {} markets",
                all_errors.len(),
                markets.len()
            );
            println!(
                "  mean |no.bid - (1-yes.ask)| = {:>7.4}   mean |no.ask - (1-yes.bid)| = {:>7.4}",
                mean_bid, mean_ask
            );
            println!(
                "  max  |no.bid - (1-yes.ask)| = {:>7.4}   max  |no.ask - (1-yes.bid)| = {:>7.4}",
                max_bid, max_ask
            );
            let verdict = if mean_bid < 0.005 && mean_ask < 0.005 {
                "The assumption holds well in this sample — mean error under half a tick."
            } else if mean_bid < 0.02 && mean_ask < 0.02 {
                "Close but not exact — small, real friction between the two independently\n  quoted sides, not just rounding."
            } else {
                "The assumption does NOT hold cleanly here — treating NO as YES's exact\n  complement would misprice it by more than a rounding error."
            };
            println!("\n  {verdict}");
        }
    }

    // ---------------------------------------------------------------
    // Charts. SVG, zero dependencies, renders anywhere.
    // ---------------------------------------------------------------
    {
        use std::io::Write;
        let baseline = run(&SimConfig::default());

        let charts: [(&str, String); 4] = [
            ("reliability.svg", px_plot::reliability(&card).0),
            ("variance-shape.svg", px_plot::variance_shape(60.0).0),
            (
                "fee-curve.svg",
                px_plot::fee_curve(0.07, 0.20, "Polymarket crypto").0,
            ),
            (
                "equity.svg",
                px_plot::equity_curve(&baseline.equity_curve, "calm market, default seed").0,
            ),
        ];

        let dir = std::path::Path::new("charts");
        let _ = std::fs::create_dir_all(dir);
        let mut written = Vec::new();
        for (name, svg) in charts {
            let path = dir.join(name);
            if let Ok(mut fh) = std::fs::File::create(&path) {
                if fh.write_all(svg.as_bytes()).is_ok() {
                    written.push(name);
                }
            }
        }
        println!("\n\nCharts written to ./charts/ : {}", written.join(", "));
        println!("\nCalibration, in the terminal:\n");
        print!("{}", px_plot::reliability_ascii(&card));
        println!(
            "\n  equity  {}",
            px_plot::sparkline(&baseline.equity_curve, 64)
        );
    }

    println!("\nHow to read this:");
    println!("  * The informed-fraction curve is a claim about the venue, not");
    println!("    about our code. If the real market sits right of the crossing,");
    println!("    no amount of engineering makes passive quoting pay.");
    println!("  * Positive mean mark-out on passive fills is expected and");
    println!("    correct. The rebate and reward accrual have to cover it.");
    println!("  * Judge on the p10, not the median. The median is what you");
    println!("    tell people; the p10 is what you actually live through.");
}
