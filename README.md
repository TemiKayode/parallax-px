# Parallax (px-*)

A research engine for TWAP-settled prediction markets. Seven library crates and
two command-line binaries. **No web UI, no HTTP server, no venue connectors, no
Python, and zero external dependencies** — `std` only.

> ### Not the same project as `parallax-ui` / `parallax-book`
>
> If you have a checkout with a `parallax-ui` crate, a dashboard on port 7878,
> Kalshi adapters or Python tests, that is a **different codebase** that happens
> to share the name. Nothing here will make that one work, and its README does
> not apply here. Crates in this workspace are all prefixed `px-`.
>
> | | this repo | `parallax-ui` repo |
> |---|---|---|
> | crates | `px-core`, `px-alpha`, `px-edge`, `px-inventory`, `px-selector`, `px-risk`, `px-engine` | `parallax-ui`, `parallax-book`, … |
> | run with | `cargo run --bin px-replay` | `cargo run -p parallax-ui` |
> | web dashboard | none | yes, `127.0.0.1:7878` |
> | venues | none — Polymarket modelled, not connected | Kalshi + Polymarket read-only |
> | dependencies | zero | axum, reqwest, … |
> | output | terminal reports | browser |

---

## Run it

```bash
cargo test --workspace              # 248 tests
cargo run --release --bin px-bench  # critical-path latency percentiles
cargo run --release --bin px-replay # scenarios, parameter sweeps, seed distribution
```

`--release` matters for the benchmark: a debug build is 20–40× slower and the
numbers mean nothing. The binary warns you if it detects one.

There is no server to start and nothing to open in a browser. Both binaries
print to stdout and exit.

---

## What it is

An engine that continuously asks *what should this outcome be worth right now,
and can I build a position before that price disappears* — for Polymarket's
crypto 5-minute, 15-minute and 4-hour markets specifically.

Three findings from the venue's documentation shaped it more than any design
preference:

**The taker fee is bigger than the edge.** `fee = C × feeRate × p × (1−p)`, and
on crypto `feeRate = 0.07` — 1.75¢ per share at the money. Makers pay zero and
collect a 20% rebate. The gap between the two sides of the same trade is 2.1¢,
two full ticks. An aggressive bot that crosses whenever its model disagrees will
lose money on most of its *correct* calls. So this is a quoting engine that
occasionally takes, not a sniper.

**Crypto markets settle on a TWAP, and variance collapses cubically.** Not on
the spot print. Inside the settlement window `Var ∝ r³`, so standard deviation
decays as `r^1.5`, not `√r`. With ten seconds left on a sixty-second window the
remaining uncertainty is 0.68% of what it was at window open — a `√τ` model says
41%. That gap is the largest single source of edge here.

**The real latency budget is orders per second.** Per-signer token buckets: 40
order tokens/s at entry tier, and a requote costs one cancel plus one order. Ten
markets two-sided works out to **two requotes per second per market**. The
sub-10 µs critical path is real and measured (p50 ≈ 1.0 µs, p99 ≈ 1.9 µs) but it
does not buy throughput — it buys the right to defer each decision to the last
possible instant.

`DESIGN.md` covers all of this properly, including the state machines, the full
latency table, and what was deliberately *not* taken from Nautilus, Hummingbot,
Freqtrade, Lean, Jesse and CCXT.

---

## Layout

```
px-core       fixed-point types, dense order book, deterministic math, clock
px-alpha      fair probability — TWAP closed form, volatility, cross-asset factor
px-edge       venue fee model, depth walker, tradable-edge calculator
px-inventory  Avellaneda–Stoikov penalty, position state machine
px-selector   relative value, 10-rule structure-selection ladder
px-risk       fractional Kelly, limits, kill switch, rate-limit governor
px-engine     the critical path, replay harness, performance statistics
```

Dependencies point one way; there are no cycles. `px-alpha` is never handed an
order book — a model that reads the venue's price cannot tell you that price is
wrong.

---

## Status: research prototype

310 tests pass (plus 19 more in the standalone `tools/px-record`), debug and
release. It is **not profitable in simulation** — median −$232 across 40
seeds, 3/40 profitable. It has never touched a live venue and there is no
code path that could.

Do not point this at money. What it is good for is the modelling and the
harness, which between them have caught eleven real bugs, including a sign flip
in `exp` reachable through `norm_pdf` in the TWAP endgame, a NaN that failed
*open* into zero safety margin, and an aggressive-churn loop that crossed 1.2
million shares against 48,000 filled passively.

**Clippy and `cargo fmt`, honestly.** `px-core` — the crate everything else in
the workspace depends on — is now fully clean: `cargo clippy -p px-core
--all-targets -- -D warnings` and `cargo fmt --check` on every file in it both
pass, including fixing a genuine rustfmt performance pathology in `exp`'s
Horner-form polynomial (restructured into an equivalent loop, verified
bit-exact against the existing test suite, not just reformatted around). The
rest of the workspace (`px-alpha`, `px-edge`, `px-inventory`, `px-selector`,
`px-risk`, and most of `px-engine`'s own pre-existing code) still does not
pass `cargo clippy --workspace --all-targets -- -D warnings` on this toolchain
(rustc/clippy 1.96.0) — several hundred more `clippy::indexing_slicing` /
`clippy::arithmetic_side_effects` / `clippy::needless_range_loop` diagnostics,
present on the very first commit of this repo, before any work in this repo
touched them. Almost certainly toolchain-version drift from whenever this was
originally validated clean, not a regression — `cargo build`/`cargo test
--workspace` both pass without warnings regardless, and every file actually
touched in this repo's work (`px-score`, `px-plot`, and the specific files
changed in `px-engine`) was separately verified clippy-clean by temporarily
holding the upstream crates' blocking lints aside for the check. Fully fixing
`px-alpha` etc. is real, substantial, separate work — scoped out here rather
than either quietly ignored or rushed through without the same bit-exactness
care `px-core`'s fix required.

### What changed since the 248-test snapshot

- **Replay against recorded venue book data**, the thing the README used to
  call the largest missing piece. `px_engine::recording::load_recording`
  parses a real captured order-book series; `SimConfig::venue_quotes =
  VenueQuoteSource::Recorded(..)` replaces the venue's synthetic
  noise-plus-spread book with real recorded best bid/ask/size at each
  instant — no synthetic spread layered on top, because there is nothing left
  to simulate. `tools/px-record` (a separate, non-workspace Cargo project —
  see its own `Cargo.toml` for why a networked recorder can't live inside a
  zero-dependency workspace) is the recorder: it auto-discovers whichever
  Polymarket "Up or Down" crypto markets are live and polls their real public
  book, both YES and NO, touch and full depth (see below). What this does
  *not* replace: the reference price path (`spot`) is still the synthetic GBM
  walk — `recording.rs`'s module doc says so plainly, and it matters more
  than it sounds: the first attempt at scoring the model against a *real*
  settled market's *real* outcome produced skill **−4.47** (Brier 0.613 vs
  the real venue's 0.112) — not evidence of no edge, but a direct measurement
  of how disconnected a synthetic-reference forecast is from an outcome that
  actually happened. `px-replay`'s "real recorded data" section runs this
  today, against `recordings/polymarket_sample.csv` (both markets' real
  resolutions fetched from Polymarket's API after they closed) and prints
  that caveat inline rather than leaving the number to misread.
- **Full L2 depth**, not just the touch. `tools/px-record` now writes every
  price level the venue's public book endpoint returns to a second file
  (`{output}.l2.{ext}`); `px_engine::recording::load_recording_l2` and
  `set_book_from_l2_snapshot` load and replay it. `recordings/
  polymarket_l2_sample.csv` is a real 210-snapshot capture across 7 markets;
  `px-replay`'s "IS venue_depth CALIBRATED AGAINST A REAL BOOK?" section
  compares real resting size near the touch against `SimConfig::default()`'s
  assumed 400-share `venue_depth` — real depth ran higher (mean ~580-850
  shares in the sample) and, more to the point, varied by orders of
  magnitude snapshot to snapshot, which a single scalar cannot represent.
- **Are YES and NO books actually complementary?** `set_complementary_book_
  from_quote` has always assumed NO's real book is exactly `1 - YES`'s,
  because this repo only ever polled one side. `tools/px-record` now polls
  both; `px_engine::recording::complementarity_error` checks the assumption
  against `recordings/polymarket_yesno_sample.csv` (330 real snapshots, 10
  markets, both sides). Verdict: the assumption holds — mean deviation 0.0001
  (a fraction of a tick) across 150 matched pairs, max exactly one tick. A
  confirmatory finding, not just a corrective one: not every real measurement
  in this pass overturned an assumption, and this is the one that didn't.
- **Is `SimConfig` actually calibrated against real data?** All four of
  `venue_lag_s` / `venue_noise` / `venue_half_spread` / `venue_depth`, checked
  in one place now — `px_engine::calibration` (unit-tested against synthetic
  fixtures with a known injected lag before ever touching real data) plus a
  `px-replay` section that runs it for real. `venue_half_spread`: assumed
  0.015, observed mean 0.0077 — real spreads on the sampled market ran about
  half as wide. `venue_noise`: assumed 0.004, observed std 0.0125 on an
  analogous (not identical) real quantity — see the module doc on why it
  isn't a literal unit match. `venue_lag_s` needed something no earlier
  recording had: a *real reference feed*, captured *overlapping in time*
  with real venue quotes. Binance/Coinbase/Kraken are unreachable from this
  sandbox, but Bitstamp isn't — `tools/px-record` now polls it every tick
  alongside the venue. Run for real: the strongest correlation in the lag
  table landed at a *negative* lag, which is diagnostic of small-sample noise
  on its own (information cannot flow backwards) rather than a usable
  measurement — reported as exactly that, an honest "cannot say yet," not a
  number forced out of eight quiet minutes.
- **`tools/px-record`, rebuilt for unattended, multi-hour runs.**
  `duration_secs = 0` now runs until Ctrl+C; the market list re-discovers on
  an interval so a 5-minute market expiring mid-run gets dropped for whatever
  replaced it instead of being polled dead for hours; a target that fails
  `MAX_CONSECUTIVE_FAILURES` polls in a row is skipped until the next
  re-discovery refreshes it, so one dead endpoint can't be hammered for the
  rest of a long session. This is what made the `venue_lag_s` capture above
  possible in the first place — real overlapping reference + venue data
  needs a recorder built to run longer than a foreground demo.
- **Clustered standard errors in `px-score`.** The paired t-test used to treat
  every logged forecast as independent, but `px-replay` pools ~200 per-second
  forecasts from each simulated seed — correlated within a seed, not across
  independent draws. `Scorecard::t_stat_clustered` (Cameron & Miller 2015
  CR1, specialised to a simple mean, one cluster per seed) is what
  `has_edge()` actually gates on now; `t_stat` (the naive one) is kept
  alongside it specifically so the gap is visible. At 60 seeds the clustered
  result was genuinely inconclusive (t = +1.00, underpowered rather than
  negative); at 500 seeds — cheap to run, ~10s — it resolves: clustered
  t = **+4.73**, skill **−0.0898**, clearly significant and clearly on the
  "worse than the venue" side. The naive statistic (t = +44.40 at 500 seeds)
  was overstating that same conclusion's confidence by roughly 10x throughout.
- **A LICENSE** — MIT, matching the sibling `parallax-ui` repo.

---

## Licence

MIT — see `LICENSE`.
