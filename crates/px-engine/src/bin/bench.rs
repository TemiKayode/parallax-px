//! Latency benchmark for the critical path.
//!
//! Measures `Engine::on_market_tick` end to end: fair probability, edge check,
//! inventory penalty, order construction, risk gate.
//!
//! Run with:
//! ```text
//!   cargo run --release --bin px-bench
//! ```
//!
//! Debug builds are 20-40x slower and tell you nothing useful; the harness
//! prints a warning if it detects one.
//!
//! Read the p99.9 and the max, not the mean. A market-making engine lives or
//! dies on its tail: the mean latency is what happens when nothing is
//! happening, and nothing is happening is not when the money is made or lost.

use px_core::{
    Category, MarketId, MarketSpec, Nanos, Px, Qty, Settlement, Side, TokenId, Underlying, Usd,
};
use px_edge::RewardModel;
use px_engine::{Engine, EngineConfig, MarketCtx};
use px_risk::Feed;
use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 20_000;
const SAMPLES: usize = 200_000;

fn spec(id: u32, underlying: Underlying, expiry_s: f64, window_s: f64) -> MarketSpec {
    MarketSpec {
        market: MarketId(id),
        yes: TokenId(id * 2),
        no: TokenId(id * 2 + 1),
        underlying,
        category: Category::Crypto,
        settlement: Settlement::Twap { window_s },
        strike: 65_000.0,
        expiry: Nanos::from_secs_f64(expiry_s),
        tick: 10_000,
        min_size: Qty::shares(5),
        reward_max_spread_ticks: 3,
        reward_min_size: Qty::shares(50),
    }
}

fn rewards() -> RewardModel {
    RewardModel {
        pool_per_day: 300_000.0 * 1e6,
        est_total_q: 5e6,
        max_spread_ticks: 3,
        one_sided_divisor: 3.0,
        min_qualifying_size: Qty::shares(50),
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn report(name: &str, mut ns: Vec<u64>) {
    ns.sort_unstable();
    let sum: u128 = ns.iter().map(|&x| x as u128).sum();
    let mean = sum as f64 / ns.len() as f64;
    println!(
        "{:<26} n={:>7}  mean={:>7.0}  p50={:>6}  p90={:>6}  p99={:>6}  p99.9={:>7}  max={:>8}",
        name,
        ns.len(),
        mean,
        percentile(&ns, 0.50),
        percentile(&ns, 0.90),
        percentile(&ns, 0.99),
        percentile(&ns, 0.999),
        ns[ns.len() - 1],
    );
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("!! debug build: these numbers are meaningless. use --release.\n");
    }

    println!("Parallax critical-path benchmark");
    println!("all figures in nanoseconds\n");

    // ---------------------------------------------------------------
    // Set up a realistic engine: 12 markets across 4 underlyings.
    // ---------------------------------------------------------------
    let mut eng = Engine::new(
        EngineConfig::default(),
        Usd::dollars(100_000),
        px_risk::Tier::Silver,
        Nanos::ZERO,
    );

    let underlyings = [
        Underlying::Btc,
        Underlying::Eth,
        Underlying::Sol,
        Underlying::Xrp,
    ];
    let durations = [(300.0, 60.0), (900.0, 60.0), (14_400.0, 60.0)];
    let mut id = 1u32;
    for u in underlyings {
        for (exp, win) in durations {
            eng.add_market(MarketCtx::new(spec(id, u, exp, win), rewards()));
            id += 1;
        }
    }
    let n_markets = eng.markets.len();

    // Warm the volatility and cross-asset estimators.
    let mut px = [65_000.0, 3_200.0, 150.0, 0.60];
    let mut t = 0.0f64;
    for i in 0..4000 {
        for (a, p) in px.iter_mut().enumerate() {
            *p *= if (i + a) % 2 == 0 { 1.00004 } else { 0.99996 };
            eng.on_reference_tick(a, *p, Nanos::from_secs_f64(t));
        }
        t += 0.03;
    }
    let now = Nanos::from_secs_f64(t + 0.01);
    for f in Feed::ALL {
        eng.risk.health.touch(f, now);
    }

    // Populate every book.
    for i in 0..n_markets {
        for k in 0..3i32 {
            eng.on_book_delta(
                i,
                true,
                Side::Bid,
                Px(480_000 - k * 10_000),
                Qty::shares(300),
                now,
            );
            eng.on_book_delta(
                i,
                true,
                Side::Ask,
                Px(520_000 + k * 10_000),
                Qty::shares(300),
                now,
            );
            eng.on_book_delta(
                i,
                false,
                Side::Bid,
                Px(480_000 - k * 10_000),
                Qty::shares(300),
                now,
            );
            eng.on_book_delta(
                i,
                false,
                Side::Ask,
                Px(520_000 + k * 10_000),
                Qty::shares(300),
                now,
            );
        }
    }

    // ---------------------------------------------------------------
    // 0. Clock overhead. Every figure below includes one `Instant::now()`
    //    plus one `elapsed()`; subtract this to get the true cost of the
    //    work being measured. For the sub-100 ns components it is the
    //    dominant term.
    // ---------------------------------------------------------------
    {
        let mut samples = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let s = Instant::now();
            black_box(());
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("(clock overhead)", samples);
    }

    // ---------------------------------------------------------------
    // 1. Order book delta application.
    //
    //    Prices stay strictly below the best ask so we do not cross our own
    //    book — which would (correctly) trip the data-quality guard and
    //    silence the engine for the rest of the run.
    // ---------------------------------------------------------------
    {
        let mut samples = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let px = Px(400_000 + ((i as i32 * 7919) % 80) * 1000);
            let s = Instant::now();
            eng.on_book_delta(
                black_box(0),
                true,
                Side::Bid,
                black_box(px),
                black_box(Qty::shares(100)),
                now,
            );
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("book delta", samples);
        // Restore a clean two-sided book for the sections that follow.
        for k in 0..80i32 {
            eng.on_book_delta(0, true, Side::Bid, Px(400_000 + k * 1000), Qty::ZERO, now);
        }
        for k in 0..3i32 {
            eng.on_book_delta(
                0,
                true,
                Side::Bid,
                Px(480_000 - k * 10_000),
                Qty::shares(300),
                now,
            );
        }
    }

    // ---------------------------------------------------------------
    // 2. Depth walk (the expected-average-entry estimate).
    // ---------------------------------------------------------------
    {
        let book = &eng.markets[0].yes_book;
        let mut samples = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let want = Qty::shares(50 + (i as i64 % 500));
            let s = Instant::now();
            let w = book.walk_buy_unbounded(black_box(want));
            black_box(w);
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("depth walk", samples);
    }

    // ---------------------------------------------------------------
    // 3. Fair value alone.
    // ---------------------------------------------------------------
    {
        use px_alpha::FairModel;
        let m = &eng.markets[0];
        let model = eng.model;
        let mut samples = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let n = Nanos(now.0 + (i as u64 % 1000) * 1_000_000);
            let s = Instant::now();
            let fv = model.fair(&m.spec, &eng.refs, &m.alpha, black_box(n));
            black_box(fv);
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("fair value", samples);
    }

    // ---------------------------------------------------------------
    // 4. The whole critical path.
    // ---------------------------------------------------------------
    {
        // Clear any sticky fault left by the earlier sections so we measure the
        // full path rather than the early-out.
        eng.risk.health.clear();
        let mut samples = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let idx = i % n_markets;
            let n = Nanos(now.0 + (i as u64) * 1_000);
            eng.risk.health.touch(Feed::Reference, n);
            eng.risk.health.touch(Feed::MarketData, n);
            eng.risk.health.touch(Feed::OrderUpdates, n);
            eng.risk.health.touch(Feed::Oracle, n);

            let s = Instant::now();
            let a = eng.on_market_tick(black_box(idx), black_box(n));
            black_box(a);
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("CRITICAL PATH", samples);
    }

    // ---------------------------------------------------------------
    // 5. Reference tick (the heavier, less frequent loop).
    // ---------------------------------------------------------------
    {
        let mut samples = Vec::with_capacity(SAMPLES / 4);
        let mut p = 65_000.0f64;
        let mut tt = t + 1.0;
        for i in 0..(WARMUP + SAMPLES / 4) {
            p *= if i % 2 == 0 { 1.00002 } else { 0.99998 };
            tt += 0.01;
            let n = Nanos::from_secs_f64(tt);
            let s = Instant::now();
            eng.on_reference_tick(black_box(0), black_box(p), n);
            let d = s.elapsed().as_nanos() as u64;
            if i >= WARMUP {
                samples.push(d);
            }
        }
        report("reference tick", samples);
    }

    println!("\nengine stats: {:?}", eng.stats);
    println!("markets: {n_markets}");
    println!(
        "\nsustainable requotes/market/sec at this tier and market count: {:.2}",
        px_risk::Tier::Silver.requotes_per_market_per_sec(n_markets)
    );
    println!(
        "  ... which is one decision every {:.0} ms. The critical path above is",
        1000.0 / px_risk::Tier::Silver.requotes_per_market_per_sec(n_markets)
    );
    println!("  the budget for deciding, not the budget for acting.");
}
