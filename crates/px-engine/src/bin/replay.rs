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

/// Parses `tools/px-record`'s reference-price format —
/// `t_unix,bid,ask,last`, `#`-prefixed comments skipped — into
/// `(t_unix, mid)` pairs for `px_engine::calibration::lag_correlation`.
/// Lives here rather than in `px-engine` itself since it is specific to
/// this one file format, not a general recording concept the library
/// needs to expose.
fn parse_reference_csv(text: &str) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 4 {
            continue;
        }
        let parse = |s: &str| s.trim().parse::<f64>().ok().filter(|v: &f64| v.is_finite());
        let (Some(t_unix), Some(bid), Some(ask)) = (parse(cols[0]), parse(cols[1]), parse(cols[2]))
        else {
            continue;
        };
        out.push((t_unix, (bid + ask) / 2.0));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

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
    // Two businesses, same market, same seeds.
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(78));
    println!("EDGE-SEEKING vs REWARD-HARVESTING");
    println!("{}\n", "=".repeat(78));
    println!("Same markets, same seeds, same model. The only difference is what the");
    println!("engine is trying to do: beat the mid, or be present near it.\n");
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>7} {:>9}",
        "mode", "pnl($)", "trading($)", "rewards($)", "uptime", "aggr shrs"
    );
    println!("{}", "-".repeat(78));
    for (label, mode) in [
        ("edge-seeking", px_engine::QuoteMode::EdgeSeeking),
        (
            "reward-harvest d=3",
            px_engine::QuoteMode::RewardHarvest { defensive_ticks: 3 },
        ),
        (
            "reward-harvest d=6",
            px_engine::QuoteMode::RewardHarvest { defensive_ticks: 6 },
        ),
    ] {
        let mut tot = 0.0;
        let mut rew = 0.0;
        let mut up = 0.0;
        let mut aggr = 0.0;
        const N: u64 = 40;
        for i in 0..N {
            let mut c = SimConfig::default();
            c.mode = mode;
            c.flow_per_s = 4.0;
            c.seed = 0x5EED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let r = run(&c);
            tot += r.pnl_dollars();
            rew += r.reward_income as f64 / 1e6;
            up += r.quote_uptime;
            aggr += r.aggressive_shares as f64 / 1e6;
        }
        let n = N as f64;
        println!(
            "{:<22} {:>10.2} {:>10.2} {:>10.4} {:>6.1}% {:>9.0}",
            label,
            tot / n,
            (tot - rew) / n,
            rew / n,
            up / n * 100.0,
            aggr / n
        );
    }
    println!("\n  (means over 40 seeds; trading = pnl minus reward income)");
    println!(
        "\n  Read the reward figure with care: RewardModel::est_total_q (this venue's\n  estimated total qualifying volume, used to size each participant's share of\n  the pool) is currently a placeholder constant, not a measured one — see\n  crates/px-edge/src/fee.rs. It is the single biggest unverified input to the\n  dollar figures above; the ratio between modes is more trustworthy than\n  either mode's absolute number until it is replaced with something measured."
    );

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
    // Is SimConfig calibrated against real data? All four parameters
    // GOING-LIVE.md's own "every parameter in a simulator config is a
    // guess until it has been checked against real flow" was written
    // about.
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("IS SimConfig CALIBRATED AGAINST REAL DATA?");
    println!("{}\n", "=".repeat(72));

    // --- venue_half_spread, venue_noise: recordings/polymarket_calibration_sample.csv ---
    {
        let raw = include_str!("../../../../recordings/polymarket_calibration_sample.csv");
        let market = "btc-updown-4h-1786852800";
        let quotes = px_engine::recording::load_recording(raw, market);
        if quotes.is_empty() {
            println!("  no calibration recording available for {market} — skipped.");
        } else {
            let assumed_half_spread = SimConfig::default().venue_half_spread as f64 / 1_000_000.0;
            let observed = px_engine::calibration::observed_half_spreads(&quotes);
            if let Some((mean, std)) = px_engine::calibration::mean_std(&observed) {
                println!(
                    "venue_half_spread — {} real snapshots of {market}",
                    quotes.len()
                );
                println!(
                    "  assumed (SimConfig::default): {:>8.4}   observed: mean {:>8.4}  std {:>8.4}",
                    assumed_half_spread, mean, std
                );
            }

            let assumed_noise = SimConfig::default().venue_noise;
            let noise = px_engine::calibration::observed_noise(&quotes, 5);
            if let Some((mean, std)) = px_engine::calibration::mean_std(&noise) {
                println!(
                    "\nvenue_noise — deviation of each mid from a trailing 5-snapshot rolling mean"
                );
                println!(
                    "  assumed (SimConfig::default): {:>8.4}   observed: mean {:>8.4}  std {:>8.4}",
                    assumed_noise, mean, std
                );
                println!(
                    "  (this is an analogue, not the same quantity: SimConfig::venue_noise perturbs\n  a *synthetic* venue's rebuilt quote around its own lagged \"true\" view; this\n  measures how much a *real* mid actually jitters around its own recent past.\n  Comparable in spirit — both describe short-term quote instability — not\n  a like-for-like unit conversion.)"
                );
            }
        }
    }

    // --- venue_lag_s: does the book's own exch_ts_ms/recv_ts_ns actually
    // reflect the configured lag? A self-consistency check, not a
    // real-data one — the calibration question below (does the *assumed*
    // 0.4s match reality) is separate from this one (does the simulator
    // internally do what its own config says it does). ---
    println!("\nvenue_lag_s — is the simulator internally consistent with its own knob?");
    {
        let mut cfg = SimConfig::default();
        cfg.flow_per_s = 3.0;
        let measured = run(&cfg).mean_measured_latency_s;
        println!(
            "  configured venue_lag_s: {:.2}s   measured (DenseBook::measured_latency_s, averaged\n  over the run): {:.2}s",
            cfg.venue_lag_s, measured
        );
        println!(
            "  measured can only be >= configured — it also includes however long the book\n  sits unmoved between repositions, and `Venue::maybe_rebuild` only repositions\n  once the view has moved a full tick. A gap this size says the synthetic\n  venue reprices far less often than a per-tick config knob suggests: on a\n  quiet 300s session the TWAP-shaped fair-value view can sit inside one tick\n  for tens of seconds at a time, so most of {measured:.1}s is dead time between\n  real repositions, not the {:.2}s of modelled lag itself. Below the\n  configured value would mean the stamping is broken; this size of gap above\n  it is a real property of the simulated venue, not a bug.",
            cfg.venue_lag_s
        );
    }

    // --- venue_lag_s: cross-correlate real venue returns against the real
    // BTC/USD reference feed, captured in the same session so the two
    // series genuinely overlap in time. ---
    println!("\nvenue_lag_s — lagged correlation, real BTC/USD reference vs real venue mid");
    {
        let raw_ref = include_str!("../../../../recordings/btc_reference_calibration_sample.csv");
        let raw_venue = include_str!("../../../../recordings/polymarket_calibration_sample.csv");
        let reference_series = parse_reference_csv(raw_ref);
        let venue_series =
            px_engine::calibration::venue_mid_series(raw_venue, "btc-updown-4h-1786852800");
        if reference_series.len() < 2 || venue_series.len() < 2 {
            println!("  not enough overlapping data — skipped.");
        } else {
            let lags: [f64; 7] = [-6.0, -3.0, 0.0, 0.4, 3.0, 6.0, 9.0];
            let results =
                px_engine::calibration::lag_correlation(&reference_series, &venue_series, &lags);
            println!(
                "  assumed venue_lag_s (SimConfig::default): {:.2}s",
                SimConfig::default().venue_lag_s
            );
            print!("  lag(s)  ");
            for (lag, _) in &results {
                print!("{lag:>7.1}");
            }
            println!();
            print!("  corr    ");
            for (_, c) in &results {
                match c {
                    Some(c) => print!("{c:>7.3}"),
                    None => print!("{:>7}", "n/a"),
                }
            }
            println!();
            // Negative lags (`reference(t + |lag|)`) are in the tried set
            // on purpose, as a control: real information cannot flow
            // backwards, so a genuine "venue lags the reference" effect
            // can only show up at lag >= 0. The strongest hit landing at
            // a negative lag is not just weak evidence, it is a direct
            // tell that whatever the number is, it is not the effect
            // being measured — a coincidence in a small sample, not a
            // venue that predicts the future.
            let best_causal = results
                .iter()
                .filter(|&&(lag, _)| lag >= 0.0)
                .filter_map(|&(lag, c)| c.map(|c| (lag, c.abs())))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let best_overall = results
                .iter()
                .filter_map(|&(lag, c)| c.map(|c| (lag, c.abs())))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if let (Some((best_lag, best_corr)), Some((causal_lag, causal_corr))) =
                (best_overall, best_causal)
            {
                if best_lag < 0.0 && best_corr > causal_corr {
                    println!(
                        "\n  The strongest |correlation| in the table is at lag {best_lag:.1}s (|r| = {best_corr:.3})\n  — a NEGATIVE lag, which is not physically possible as a real \"venue lags\n  reference\" effect (information cannot flow backwards). That is itself the\n  finding: this is small-sample noise, not signal. Restricted to lags >= 0,\n  the actual candidates for venue_lag_s, the best is only {causal_lag:.1}s at\n  |r| = {causal_corr:.3} — too weak to read as a real measurement.\n  Honest conclusion: {} real venue observations over an ~8-minute, largely\n  quiet window is not enough statistical power to calibrate venue_lag_s\n  either way. Not evidence the assumed 0.4s is wrong — evidence that this\n  measurement needs a longer, more volatile session\n  (tools/px-record's new unattended mode makes that practical) before it\n  can say anything at all.",
                        venue_series.len()
                    );
                } else if causal_corr > 0.3 {
                    println!(
                        "\n  Strongest causal (lag >= 0) |correlation| at lag {causal_lag:.1}s (|r| = {causal_corr:.3}).\n  Read this as a weak signal, not a confident calibration: {} real venue\n  observations over an ~8-minute, largely quiet window is not much\n  statistical power for a lead-lag estimate, and the reference feed\n  (Bitstamp spot) is not necessarily what this venue's own book actually\n  prices off internally.",
                        venue_series.len()
                    );
                } else {
                    println!(
                        "\n  No lag >= 0 in the range tried shows a real correlation (best |r| = {causal_corr:.3},\n  {} real venue observations). Honest reading: this ~8-minute real capture\n  did not contain enough genuine BTC price movement to estimate\n  venue_lag_s with any confidence either way — not evidence the assumed\n  0.4s is wrong, just that this measurement cannot yet say. A longer, more\n  volatile session (tools/px-record's new unattended mode makes that\n  practical) is the actual next step, not a number forced out of a quiet\n  sample.",
                        venue_series.len()
                    );
                }
            } else {
                println!("\n  Not enough usable lag/correlation pairs to say anything.");
            }
        }
    }

    // --- venue_depth: recordings/polymarket_l2_sample.csv (full depth) ---
    println!("\nvenue_depth — real resting size near the touch");
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
                "  assumed (SimConfig::default): {assumed:>10.1} shares per side   observed, within {:.0}c of\n  the touch: bid {:>10.1}   ask {:>10.1}  (mean per snapshot)",
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
    // Can recalibration fix it? Honest train/test, split by session.
    // ---------------------------------------------------------------
    println!("\n\n{}", "=".repeat(72));
    println!("CAN RECALIBRATION FIX IT?");
    println!("{}\n", "=".repeat(72));
    {
        let mut all = Vec::new();
        for i in 0..120u64 {
            let mut c = SimConfig::default();
            c.seed = 0x5EED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            // Tag every forecast with its session so the split cannot leak.
            for mut r in run(&c).scorecard_inputs {
                r.forecast.horizon_s = i as f64;
                all.push(r);
            }
        }
        let res = px_score::calibrate::train_test_split(&all, |r| r.forecast.horizon_s as u64);
        print!("{}", px_score::calibrate::report(&res));
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
