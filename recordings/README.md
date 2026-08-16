# Recordings

Real Polymarket order-book snapshots, captured with `tools/px-record` and
loaded by `px_engine::recording::load_recording`. See that module's doc
comment for the file format and exactly what replaying one does and does
not replace in the simulation.

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

Re-capture a fresh sample any time with:

```bash
cd tools/px-record
cargo run --release -- ../../recordings/my_recording.csv 780 3
```

(`780` seconds, `3` second poll interval — adjust to taste. The recorder
auto-discovers whichever "Up or Down" crypto markets are active when it
starts, so the market names it finds will not match `btc_4h`/`btc_1h`
exactly; use whatever `market` slug it prints.)
