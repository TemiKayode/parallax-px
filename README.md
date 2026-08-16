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

289 tests pass, debug and release. It is **not profitable in simulation** —
median −$232 across 40 seeds, 3/40 profitable. It has never touched a live
venue and there is no code path that could.

Do not point this at money. What it is good for is the modelling and the
harness, which between them have caught eleven real bugs, including a sign flip
in `exp` reachable through `norm_pdf` in the TWAP endgame, a NaN that failed
*open* into zero safety margin, and an aggressive-churn loop that crossed 1.2
million shares against 48,000 filled passively.

**Clippy, honestly:** `cargo clippy --workspace --all-targets -- -D warnings`
does not currently pass on this toolchain (rustc/clippy 1.96.0) — `px-core`
alone has 60+ pre-existing `clippy::arithmetic_side_effects`/`indexing_slicing`/
`float_cmp` diagnostics under it, present on the very first commit of this
repo, before any of the work below. Almost certainly a toolchain-version drift
from whenever this was originally validated clean, not a regression introduced
here — `cargo build`/`cargo test --workspace` both pass without warnings
regardless, and the three crates touched in this pass (`px-score`, `px-engine`,
`px-plot`) were separately verified clean against the same clippy with
`px-core`'s blocking lints held aside for the check. Fixing `px-core` itself —
bounds-checking every book index, deciding what fixed-point arithmetic actually
needs overflow checking versus a scoped `#[allow]` — is real, separate work,
stated here rather than left for someone to discover the hard way.

**`cargo fmt`, same story.** `cargo fmt --all -- --check` does not complete in
practical time on this toolchain (rustfmt 1.9.0) either — `px-core/src/lib.rs`
and `px-core/src/math.rs` specifically (the deeply nested Horner-form
polynomials in `exp`/`norm_ppf`) hit a known rustfmt performance pathology on
deeply nested expressions and were still running after several minutes,
standalone, with no other process involved. Every other file in the workspace
(and the standalone `px-record` tool) formats cleanly and near-instantly on
its own — checked file by file with plain `rustfmt --check`, since the `cargo
fmt` wrapper hangs the moment it reaches either of those two files. Restructuring
bit-exact numerical code to dodge a formatter's performance edge case is not
something to do casually, so it is named here rather than either silently
skipped or hacked around.

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
  book. `recordings/polymarket_sample.csv` is a genuine sample it captured —
  13 minutes across two then-live BTC markets, through actual settlement,
  including a one-sided book with no ask at all in its last few rows as
  liquidity pulled — and `crates/px-engine/src/replay.rs` has an integration
  test that loads it and runs a full simulated session against it end to end.
  What this does *not* yet replace: the reference price path (`spot`) is
  still the synthetic GBM walk, so this checks "does the strategy behave
  sanely against a real venue book," not yet "was the GBM assumption itself
  right" — stated plainly in `recording.rs`'s module doc, not left implicit.
- **Clustered standard errors in `px-score`.** The paired t-test used to treat
  every logged forecast as independent, but `px-replay` pools ~200 per-second
  forecasts from each of 60 simulated seeds — correlated within a seed, not
  across 12,000 independent draws. `Scorecard::t_stat_clustered` (Cameron &
  Miller 2015 CR1, specialised to a simple mean, one cluster per seed) is what
  `has_edge()` actually gates on now; `t_stat` (the naive one) is kept
  alongside it specifically so the gap is visible. Run for real against the
  pooled 60-seed scorecard: naive t = **+10.21** (reads as confidently worse
  than the venue), clustered t = **+1.00** (genuinely inconclusive, once you
  count independent evidence instead of correlated readings of it) — the
  naive statistic was overstating confidence by roughly 10x.
- **A LICENSE** — MIT, matching the sibling `parallax-ui` repo.

---

## Licence

MIT — see `LICENSE`.
