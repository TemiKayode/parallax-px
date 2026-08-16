# Recordings

Real Polymarket order-book snapshots, captured with `tools/px-record` and
loaded by `px_engine::recording`. See that module's doc comment for the
file formats and exactly what replaying one does and does not replace in
the simulation.

## `polymarket_sample.csv`

Captured 2026-08-16, 03:45–03:58 UTC, against two real, then-live markets:

- `btc_4h` — "Bitcoin Up or Down - August 15, 8:00PM-12:00AM ET" (a
  4-hour window), recorded through its final ~13 minutes and into
  settlement — the book's ask side disappears in the last few rows
  (`None,None`) as the market resolves and liquidity providers pull their
  offers, which `load_recording` skips as malformed rather than
  misreading as a real quote.
- `btc_1h` — "Bitcoin Up or Down - August 15, 11PM ET" (a 1-hour window),
  recorded over the same span, including a real, fast move from ~0.46 to
  ~0.95 in its last few minutes as the underlying broke one direction.

521 lines, 244 `btc_4h` rows and 245 `btc_1h` rows after filtering (the
rest are the `#`-prefixed header and ~31 transient fetch-error comment
lines the recorder logged and continued past, real network noise from an
unattended ~13-minute run against a live public endpoint).

## `polymarket_l2_sample.csv`

Captured 2026-08-16, ~05:35–05:37 UTC, `tools/px-record`'s L2 mode (every
run now writes full depth to `{output}.l2.{ext}` alongside the touch-only
file — see `px_engine::recording::load_recording_l2`). 210 real
full-depth snapshots across 7 live markets, up to dozens of price levels
per side on a given snapshot — real resting size at each level, not the
touch-only best bid/ask `polymarket_sample.csv` reduces the book to.
`crates/px-engine/src/bin/replay.rs`'s "IS venue_depth CALIBRATED AGAINST
A REAL BOOK?" section uses this to compare `SimConfig::default()`'s
assumed `venue_depth` against what a real book near the touch actually
holds.

## `polymarket_yesno_sample.csv`

Captured 2026-08-16, ~05:35–05:36 UTC. Same touch-only format as
`polymarket_sample.csv`, but for every market it also polled and
recorded the NO/"Down" outcome token, written under `{market}-no` in the
same file — `tools/px-record` now polls both sides of every market it
discovers, not just YES. 330 snapshots across 10 markets, both sides
each.

Built to check `recording.rs`'s `set_complementary_book_from_quote`
assumption (that NO's real book is exactly `1 - YES`'s) against actual
independently-polled data instead of just asserting it —
`px_engine::recording::complementarity_error` does the comparison, and
`px-replay`'s "ARE YES AND NO REAL BOOKS ACTUALLY COMPLEMENTARY?" section
runs it against this file. Verdict on this sample: yes, closely — mean
deviation from perfect complementarity across 150 matched snapshot pairs
is 0.0001 (a fraction of a tick), max is exactly one tick (0.01). The
assumption holds.

## `polymarket_calibration_sample.csv` + `btc_reference_calibration_sample.csv`

Captured together, 2026-08-16, ~06:12–06:20 UTC — the first pair in this
directory to capture Polymarket venue quotes and a genuinely independent
real reference price (Bitstamp's public BTC/USD ticker) *overlapping in
time*, from `tools/px-record`'s extended unattended mode (Ctrl+C-capable,
periodic market re-discovery, per-target failure backoff — see its module
doc). 1,380 real venue snapshots (13 markets, YES and NO) and 112 real
BTC/USD reference ticks over 8 minutes, zero fetch errors.

Built specifically to check `SimConfig::venue_lag_s` — the one parameter
`polymarket_l2_sample.csv`/`polymarket_yesno_sample.csv` structurally
couldn't reach, since checking whether the venue's book lags a *reference*
feed needs a real reference feed recorded alongside it, not just more
venue data. `px_engine::calibration::lag_correlation` does the
cross-correlation; `px-replay`'s "IS SimConfig CALIBRATED AGAINST REAL
DATA?" section runs the full check (also `venue_half_spread` and
`venue_noise`, both measurable from `polymarket_calibration_sample.csv`
alone). Honest verdict on this sample: inconclusive by construction — an
8-minute, largely quiet window doesn't contain enough genuine BTC
movement to estimate a lead-lag relationship with any confidence, and the
tool says so rather than reporting a number anyway (the strongest
correlation in the table lands at a *negative* lag, which is not
physically possible as a real "venue lags reference" effect and is
therefore diagnostic of noise, not signal, on its own). `venue_half_spread`
and `venue_noise` came back with real, usable numbers from the same
capture: observed half-spread (mean 0.0077) ran about half the assumed
0.015, and observed quote jitter (std 0.0125) ran roughly 3x the assumed
`venue_noise` of 0.004 — though that comparison is an analogue, not a
literal unit match; see the module doc on `observed_noise` for why.

## Re-capturing

```bash
cd tools/px-record
cargo run --release -- ../../recordings/my_recording.csv 780 3
```

(`780` seconds, `3` second poll interval, `180`s re-discovery interval by
default — pass a 4th argument to change it. `duration_secs = 0` runs
until Ctrl+C, for a genuinely long unattended capture. Writes
`my_recording.csv` (touch, YES and `-no`), `my_recording.l2.csv` (full
depth), and `my_recording.ref.csv` (real BTC/USD reference ticks). The
recorder auto-discovers whichever "Up or Down" crypto markets are active
when it starts — and re-discovers periodically as markets expire — so
the market names it finds will not match the ones already shipped here;
use whatever `market` slug it prints.)
