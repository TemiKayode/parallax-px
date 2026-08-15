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
    Outage, Report, Shock, SimConfig,
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
    {
        // Pool forecasts across many seeds: one session is far too few
        // resolved forecasts to say anything.
        let mut pooled = px_score::Scorer::new();
        for i in 0..60u64 {
            let mut c = SimConfig::default();
            c.seed = 0x5EED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for r in run(&c).scorecard_inputs {
                pooled.record(r);
            }
        }
        print!("{}", px_score::report(&pooled.score()));
    }

    // ---------------------------------------------------------------
    // Charts. SVG, zero dependencies, renders anywhere.
    // ---------------------------------------------------------------
    {
        use std::io::Write;
        let mut pooled = px_score::Scorer::new();
        for i in 0..60u64 {
            let mut c = SimConfig::default();
            c.seed = 0x5EED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for r in run(&c).scorecard_inputs {
                pooled.record(r);
            }
        }
        let card = pooled.score();
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
