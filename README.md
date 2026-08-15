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

248 tests pass, zero warnings, debug and release. It is **not profitable in
simulation** — median −$232 across 40 seeds, 3/40 profitable. It has never
touched a live venue and there is no code path that could.

Do not point this at money. What it is good for is the modelling and the
harness, which between them have caught eleven real bugs, including a sign flip
in `exp` reachable through `norm_pdf` in the TWAP endgame, a NaN that failed
*open* into zero safety margin, and an aggressive-churn loop that crossed 1.2
million shares against 48,000 filled passively.

The largest missing piece is replay against recorded book data. Every parameter
in `SimConfig` is a guess until then.

---

## Licence

Unlicensed / all rights reserved unless you add one.
