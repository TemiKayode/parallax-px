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

## Re-capturing

```bash
cd tools/px-record
cargo run --release -- ../../recordings/my_recording.csv 780 3
```

(`780` seconds, `3` second poll interval — adjust to taste. Writes both
`my_recording.csv` (touch, YES and `-no`) and `my_recording.l2.csv`
(full depth). The recorder auto-discovers whichever "Up or Down" crypto
markets are active when it starts, so the market names it finds will not
match the ones already shipped here; use whatever `market` slug it
prints.)
