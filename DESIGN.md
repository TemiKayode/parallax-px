# Parallax — System Design

A cross-venue, event-driven mispricing engine for prediction markets.

**Status: research prototype.** 220 tests pass; the critical path runs in 832 ns
at the median. It is **not** yet profitable in simulation — median −$211 across
40 seeds, 4/40 profitable. Section 9 explains what that means and what would
have to change. Do not point this at real money.

---

## 0. The three findings that reshaped the design

The original brief specified an ultra-low-latency taker: colocated hardware,
FPGA parsing, FIX cross-connects, sub-10 µs tick-to-trade, crossing the spread
whenever the model disagrees with the book. Reading the venue's actual
documentation changed almost all of it.

### 0.1 The taker fee is larger than the edge

Polymarket charges takers `fee = C × feeRate × p × (1−p)`. On crypto markets
`feeRate = 0.07`, so a taker crossing at 50¢ pays **1.75¢ per share**. Makers pay
zero and receive 20% of the counterparty's fee.

| At 50¢, per share | Taker | Maker |
|---|---:|---:|
| Fee | −1.75¢ | 0 |
| Rebate | 0 | +0.35¢ |
| **Net** | **−1.75¢** | **+0.35¢** |

The gap between the two sides of the same trade is **2.1¢ — two full ticks**.
Almost no mispricing in a liquid five-minute book survives that. An aggressive
bot that crosses whenever its model disagrees will lose money on the *majority
of its correct calls*.

So the architecture inverted: this is a **quoting engine that occasionally
takes**, not a sniper. Taking is reserved for edges wide enough to clear 1.75¢
(they occur, briefly, after a shock), for model-free arbitrage, and for
inventory reduction where the fee is the price of not carrying risk.

Fees fall toward the extremes — at 97¢ the fee is 0.20¢, an order of magnitude
smaller. That is the entire reason the near-resolution structure survives the
schedule while at-the-money taking does not.

### 0.2 Crypto markets settle on a TWAP, and the variance collapses cubically

Polymarket's crypto 5-minute, 15-minute and 4-hour markets settle on a Chainlink
**time-weighted average price** over a window of `W` seconds ending at expiry —
not the spot print. Modelling this correctly is the single largest source of edge
in the system.

Model the underlying as arithmetic Brownian motion with volatility `σ`.
Settlement is `V = (1/W)∫S(u)du` over `[T−W, T]`.

**Before the window opens** (`a = (T−W) − t` seconds away):

```
E[V]   = S(t)
Var[V] = σ²(a + W/3)
```

**Inside the window** (`r = T − t` remaining, `φ = (W−r)/W` elapsed, `A` = the
average already observed):

```
E[V]   = φ·A + (r/W)·S(t)
Var[V] = σ²·r³ / (3W²)
```

The two agree at `r = W`, as they must.

The consequence is the part that matters. Variance inside the window decays as
**`r³`**, so standard deviation decays as **`r^1.5`**, not `√r`:

| Time left on a 60 s window | Correct sd (relative) | `√τ` model says |
|---|---:|---:|
| 60 s (window opens) | 1.000 | 1.000 |
| 30 s | 0.354 | 0.707 |
| 10 s | 0.068 | 0.408 |
| 3 s | 0.011 | 0.224 |

A worked case from the test suite: BTC $30 above the strike, 10 seconds left,
observed window average $12 above. The correct model gives `p > 0.999999` —
decided. A spot-settled `√τ` model gives `p = 0.980`, and quotes 98¢ into a
market worth 100¢. Everything else in this system — the depth walker, the
inventory penalty, the rate-limit governor — exists to convert that two-cent
difference into filled size before the resting liquidity is repriced.

Also note `Var = σ²W/3` at window open: a TWAP-settled contract carries **one
third** the variance of a spot-settled one. Pricing it as spot-settled is
systematically too close to 50¢.

### 0.3 The real latency budget is orders per second, not nanoseconds

Polymarket meters order and cancel requests through **per-signer token buckets**.
Orders cannot be modified in place, so changing a quote costs one cancel token
*and* one order token. A two-sided quote is two of each.

| Tier | 30-day volume | Order rate/s | Order burst |
|---|---:|---:|---:|
| Standard | — | 40 | 60 |
| Silver | $100k+ | 200 | 300 |
| Elite | $10M+ | 600 | 900 |

```
sustainable requotes/sec/market = order_rate / (2 × markets_quoted)
```

**Ten markets at the entry tier gives two requotes per second per market.** Not
two thousand. Two.

There is also no FIX, no cross-connect, and no colocation. An order is an
EIP-712 signature over HTTPS to a hosted matching engine: signing alone is tens
of microseconds, and the network round trip is milliseconds.

So the sub-10 µs critical path in this design is real and measured — but it is
not what buys the edge. What it buys is **decision quality within a fixed
budget**: when you get two quote updates per second, each one must be computed
from the freshest possible state, and microseconds are what let you defer the
decision to the last instant before sending. Quote budget is a scarce resource
that must be *allocated*, which is why the governor lives in `px-risk` rather
than in a networking layer.

---

## 1. Architecture

Seven crates. Dependencies point one way; there are no cycles.

```
                 ┌───────────┐
                 │  px-core  │  types, dense book, deterministic math, clock
                 └─────┬─────┘
        ┌──────────┬───┴────┬───────────┬──────────┐
        │          │        │           │          │
   ┌────▼────┐ ┌───▼───┐ ┌──▼────────┐ ┌▼───────┐  │
   │px-alpha │ │px-edge│ │px-inventory│ │px-risk │  │
   │fair prob│ │ fees, │ │  penalty,  │ │ Kelly, │  │
   │  TWAP,  │ │ edge, │ │  position  │ │limits, │  │
   │ x-asset │ │ walk  │ │   FSM      │ │governor│  │
   └────┬────┘ └───┬───┘ └──┬─────────┘ └┬───────┘  │
        │          │        │            │          │
        │     ┌────▼────────▼────┐       │          │
        │     │   px-selector    │       │          │
        │     │ structure FSM,   │       │          │
        │     │ relative value   │       │          │
        │     └────────┬─────────┘       │          │
        └──────────────┼─────────────────┴──────────┘
                  ┌────▼─────┐
                  │px-engine │  critical path, replay harness, stats
                  └──────────┘
```

### The one rule

**Nothing in `px-alpha` may read a Polymarket price.** Not the mid, not the
touch, not the last trade. Fair value comes from the reference asset, its
volatility, the settlement mechanics, and the clock.

This is enforced by the type system: `FairModel::fair` is not handed a book. A
model that anchors on the venue's own price cannot, even in principle, tell you
that price is wrong — it converges on whatever the book says and reports a
comfortable, useless zero edge.

Book state *is* used, by `px-edge` to compute realisable entry and by `px-risk`
to size. It enters after fair value exists, never before.

### Two loops, not one

| Loop | Trigger | Frequency | Work |
|---|---|---|---|
| Reference tick | spot print | ~20/s per underlying | volatility, cross-asset factor, TWAP integral |
| Market tick | book delta | ~1000/s per market | fair value, edge, penalty, selection, risk |

Splitting them keeps the expensive statistical work off the path that runs most
often. Fair value on the market tick is a handful of multiplies and one
`norm_cdf`, because everything else was computed when the reference last printed.

---

## 2. Latency budget — measured, not aspirational

Release build, 12 markets across 4 underlyings, 200,000 samples, shared cloud VM.
All figures nanoseconds; subtract ~14 ns of clock overhead.

| Stage | p50 | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|
| Book delta apply | 16 | 20 | 41 | 182 |
| Depth walk (avg entry) | 35 | 44 | 61 | 443 |
| Fair value (TWAP + greeks) | 128 | 169 | 547 | 1,346 |
| **Full critical path** | **832** | **1,058** | **2,252** | **12,213** |
| Reference tick | 222 | 296 | 806 | 3,067 |

The critical path is market data → fair probability → edge check → inventory
penalty → order construction → risk gate → order intent. **p99 = 2.25 µs**,
comfortably inside the 10 µs target.

The p99.9 of 12 µs and the 321 µs max are host scheduler preemption on a shared
VM, not code. On a tuned box — isolated cores, `nohz_full`, huge pages, no
frequency scaling — that tail collapses. Read p99 here.

**Where the time goes.** Fair value is 128 ns of the 832. The remainder is the
depth walks (four of them, including the complete-set pair pricing), the
relative-value tracker, the selector ladder, and the risk gate. Nothing
allocates, locks, or syscalls.

### What speed is actually worth

The harness sweeps our own execution latency against P&L:

| Round-trip latency | P&L |
|---|---:|
| 0.01 ms | −$230 |
| 1 ms | −$230 |
| 10 ms | −$230 |
| 25 ms | −$130 |
| 100 ms | −$1,228 |
| 500 ms | −$33,601 |

Below the venue's own quantisation, latency is free — 10 µs and 10 ms are
indistinguishable. Beyond ~100 ms it is catastrophic. The engineering target is
therefore "reliably under a few tens of milliseconds", not "as fast as
physically possible", and effort beyond that is better spent on the model.

---

## 3. Fair probability (`px-alpha`)

### Volatility: two estimators, three uses

A time-aware EWMA (decay by elapsed time, estimate the variance *rate* `r²/dt`)
at two half-lives: fast (3 s) and slow (180 s). Irregular tick arrival makes a
per-sample EWMA silently weight ten ticks in a millisecond the same as ten ticks
in a second.

Three separate quantities come out, and conflating them was a real bug:

| Quantity | Definition | Used by |
|---|---|---|
| `sigma_rel()` | precision-weighted blend | **pricing** |
| `sigma_rel_conservative()` | `max(fast, slow)` | inventory penalty, limits |
| `rel_err()` | sampling error + *excess* disagreement | safety margin, spread width |

**Why not just take the maximum.** That was the first implementation, and the
harness showed it losing money. A conservative volatility is *not* a
conservative price: overstating σ pulls fair probability toward 50¢, which away
from the money is a systematic **directional** error. When the true value is 72¢
and we say 66¢ because σ is too high, we are not being careful — we are quoting
an offer at 67¢ that the market will lift all day. Conservatism belongs in the
error bar, not the point estimate.

**Why "excess" disagreement.** The second attempt added `burst_ratio − 1`
straight into `rel_err`, and shut the strategy off entirely. A fast estimator
with a 3-second half-life sees ~60 samples, so its own standard error is 10–30%;
on a stationary random walk it reads 1.3× the slow estimator much of the time *by
construction*. Only the part exceeding two standard errors is information.

Robustness: 1 ms floor on `dt` (duplicate timestamps), 8σ winsorisation
(fat-finger prints), rejection of non-finite and non-positive prices.

### Cross-asset: redundancy measured, not assumed

Seven underlyings, EWMA covariance on a synchronised 250 ms sampling grid (async
sampling crushes measured correlation via the Epps effect). First eigenvector by
power iteration gives the common factor; `R²ᵢ = vᵢ²λ₁` is each asset's
redundancy.

Effective degrees of freedom uses a trick that avoids eigendecomposition
entirely — for a correlation matrix, `trace = N`, so

```
dof = (Σλ)² / Σλ² = N² / ‖C‖²_F
```

which is a sum of squared entries, O(N²). It runs from 1 (everything is one
trade) to N (all independent), and it scales the portfolio gross limit by
`√(dof/N)`.

What correlated assets actually give us is **not direction** — crypto is close
enough to a martingale over five minutes. It is:

1. **Nowcasting a stale feed.** If BTC goes quiet for 300 ms while ETH and SOL
   print and move together, the factor says where BTC almost certainly is now.
   This is the genuine low-latency edge: it buys back the gap *between feeds*
   rather than trying to beat the market.
2. **Correct risk aggregation.** BTC 5m and ETH 5m are not two independent bets
   during a systematic move.

### Uncertainty propagation

```
σ_p² = (∂p/∂S · σ · √age)²  +  (∂p/∂σ · σ · rel_err)²
        ↑ feed staleness         ↑ volatility estimation error
```

Both greeks are analytic and verified against central finite differences to
1e-6 in the test suite. `σ_p` then drives the safety margin *and* the quote
width — so a cold estimator or a stale feed automatically widens the spread and
eventually stops the bot, with no special case written anywhere.

---

## 4. Tradable edge (`px-edge`)

```
Tradable Edge = Fair Value − Expected Average Entry − Execution Costs − Safety Margin
```

Every term computed, none assumed:

- **Expected average entry** — walk the actual resting depth for the actual
  intended size. Partial fills and multi-level slippage are priced. Test case: a
  book showing 52¢ at the touch costs an average of **53.45¢** for 200 shares.
- **Execution costs** — the venue's exact fee formula, reproduced against the
  published table to half a cent per hundred shares.
- **Safety margin** — `k × σ_p`, converted exactly (a binary pays $1, so one
  unit of probability is one dollar of value).

### Size is an output, not an input

`optimal_take` walks level by level, recomputing the average entry, the fee at
that average, and the resulting *total* edge, then returns the argmax. The answer
is frequently not the touch: paying an extra tick for four times the size is
usually the better trade. In the test suite, a 9¢ edge on 25 shares loses to a
5.5¢ edge on 485 shares.

It stops as soon as net edge goes negative — average entry only worsens with
depth, so it cannot recover.

### Maker economics

```
net = (fair − price)          ← the mispricing we quote into
    + maker rebate            ← venue pays us
    + expected reward accrual ← liquidity programme, time-weighted
    − adverse selection       ← we fill when we are wrong
    − safety margin
```

The first three are why a maker can profitably quote a price a taker could not
profitably cross. A test asserts exactly this: the same 1.5¢ mispricing is a
**loss** as a taker and a **profit** as a maker.

### The liquidity reward, and the tension it creates

The venue scores resting orders with `S(v,s) = ((v−s)/v)²` — quadratic in
distance from mid, hard cliff at the max qualifying spread — sampled once per
minute at a random instant, normalised against every other maker.

Two consequences fall straight out:

1. **The objective is time-weighted presence.** Random sampling means what earns
   is the *fraction of wall-clock time* resting inside `v`. A quote cancelled and
   replaced fifty times a second earns exactly as much as one that rests, and
   costs fifty times the rate-limit budget. This is the strongest argument for
   the quote-economy governor.
2. **The gradient is steep.** Moving from 3 ticks out to 1 tick out quadruples
   the score.

**Unresolved tension.** Rewards require quoting within ~3 ticks of the venue's
mid. The strategy's entire premise is that the venue's mid is *wrong*. These pull
in opposite directions, and current quote uptime inside the reward band is only
6–30%. Either the reward is largely unreachable for this strategy, or the
strategy should run two quote layers — a reward-harvesting layer near the mid and
an edge layer away from it. This is not settled.

---

## 5. Position structures (`px-selector`)

### Relative value

```
Relative Score = (Current Gap − Typical Gap) / Historical Gap Volatility
```

Raw gaps are not comparable across markets: 2¢ on a 5-minute contract twenty
seconds from expiry is an enormous dislocation; the same 2¢ on a 4-hour contract
is noise. A market that habitually trades a cent rich has a *typical* gap of one
cent, and paying for that is paying for non-information.

The ranker sorts by **total edge dollars**, not z-score — a spectacular z on a
market holding nine shares is not where capital goes — and deduplicates by
correlation bucket, so once BTC 5m is taken, BTC 15m is not also taken.

### The selection ladder

A strict priority ladder, not a scoring function. Ladders are auditable: for any
decision exactly one rule fired, and the `Reason` enum returned with every
decision names it. "Rule 5 fired because the pair cost 96.2¢" is a debuggable
sentence at 3 am. "The utility function preferred it" is not.

```
 1. feed fault / model unusable ......... Flatten (if exposed) else Idle
 2. risk layer has frozen us ............ Idle
 3. inventory at hard limit ............. Flatten
 4. complete sets held, capital tight ... InventoryRelease
 5. sub-dollar pair available ........... SyntheticPair    ← model-free
 6. outcome decided, discount left ...... NearResolution
 7. dislocation beyond z_enter .......... HedgedDirectional
 8. fair value crossed, cooldown ok ..... DynamicRotation
 9. passive quote clears the hurdle ..... TwoSidedQuote    ← the default
10. otherwise .......................... Idle
```

Rule 9 is where the bot spends nearly all its time and earns nearly all its
money. Rules 5–8 are the exceptions that justify the machinery.

Rule 5 sits above the model-driven rules deliberately: a complete YES+NO set pays
exactly $1 regardless of outcome, so assembling one below $1 net of fees is the
only structure whose profit does not depend on the model being right. A test
asserts that 49¢ + 50¢ = 99¢ is correctly **rejected** — the two taker fees come
to 3.5¢.

### Whipsaw safeguards

| Guard | Mechanism |
|---|---|
| Dwell lock | 250 ms minimum in a structure; safety rules bypass it |
| Rotation cooldown | 2 s between direction flips |
| Rotation cap | 6 per market lifetime |
| Rotation band | fair value must clear 0.5 ± 0.04 |
| Entry/exit asymmetry | enter at z ≥ 2.5, leave at z < 1.0 |

Regression test: 2,000 ticks with a z-score oscillating every tick produces
fewer than 100 transitions. A second test drives `p` alternating 0.51/0.49 for
200 ticks and asserts **zero** rotations.

---

## 6. Inventory (`px-inventory`)

```
Working Price = Fair Value − Inventory Penalty
Penalty       = q · λ · σ² · τ
```

This is the Avellaneda–Stoikov reservation price. A maker already long is not
indifferent to buying more: the next unit adds variance to a book that already
carries variance.

**Skew the working price, don't widen the spread.** Widening makes us less
competitive on both sides equally and forfeits the reward on both. Skewing keeps
us aggressive on the risk-reducing side and passive on the risk-adding side — we
stay in the reward programme while actively recruiting the flow we want.

**On double-counting time.** `σ²τ` is total remaining variance, and for a TWAP
contract that does not decay linearly in `τ` (§0.2). Passing a per-second rate
*and* `τ` would be wrong twice. `penalty_from_model` takes the remaining variance
directly from the fair-value computation. A test confirms the penalty for
carrying a position through the last seconds of a TWAP window collapses to under
1% of its value at window open — a nearly-certain position is nearly riskless,
and the engine should stop paying to reduce it.

**Complete sets.** One YES + one NO merges back to $1, so the matched portion is
not exposure — it is capital waiting to be released. `Flattening` prefers merging
a matched pair (free) over selling a naked leg (taker fee).

### State machine

```
                    |net| grows  ─────────────►
  Flat ── Balanced ── Skewed ── Reducing ── Flattening
    ◄───────  ◄────────  ◄────────  ◄────────
              exit at 70% of entry threshold

  any state ──(risk gate / feed fault)──► Frozen ──(all clear)──► Flat
```

| State | Threshold | May add? | Size factor (add / reduce) |
|---|---:|---|---|
| Flat | 0 | yes | 1.0 / 1.0 |
| Balanced | 50 sh | yes | 0.8 / 1.0 |
| Skewed | 200 sh | yes | 0.4 / 1.2 |
| Reducing | 500 sh | **no** | 0.0 / 1.5 |
| Flattening | 1000 sh | **no** | 0.0 / 2.0 (crosses) |
| Frozen | — | **no** | 0.0 / 0.0 |

The 70% hysteresis band is not cosmetic. Without it, a position parked on a
boundary flips state on every fill and the quoting engine cancel-replaces
forever, burning the rate-limit budget the whole system depends on. Test: 50
one-share oscillations across the boundary produce **zero** transitions.

---

## 7. Risk (`px-risk`)

Everything here is a **veto**. No component in this crate can cause a trade; each
can only reduce or refuse one. A bug in the risk layer therefore fails toward
inaction.

### Sizing

For a binary at price `c` with true probability `p`, net odds are `b = (1−c)/c`
and the log-optimal stake fraction is

```
f* = (p·b − (1−p)) / b = (p − c) / (1 − c)
```

Note what `f*` is a fraction *of*: bankroll to stake, not share count. Shares are
`f*·W/c`. Confusing the two is how you end up eight times overbet on a 12¢
contract. The fee goes into the effective cost before sizing — a 52¢ contract
with a 1.75¢ fee is a 53.75¢ contract.

We size at **20% of full Kelly**. Full Kelly is optimal only if `p` is *known*;
ours is estimated, and the growth curve is brutally asymmetric. At half Kelly we
keep >70% of the growth; at double Kelly growth is already negative. An estimate
wrong by 3× at one-fifth Kelly still lands under full Kelly. At full Kelly it
lands in the red.

### Limits

| Limit | Default | Aggregation |
|---|---|---|
| Capital per market | $5,000 | per market |
| Unhedged shares | 2,000 | per market, absolute net |
| Bucket exposure | $20,000 | **per underlying** — BTC 5m + 15m share one |
| Gross exposure | $100,000 | scaled by `√(dof/N)` |
| Kelly fraction | 0.2 | global |

The gross limit contracting with correlation is the portfolio-level version of
the brief's correlation requirement: when the cross-asset monitor reports one
effective degree of freedom, every position is the same position, and the limit
contracts to 37.8% of base.

### Kill switch

Named causes, not a boolean, because the recovery procedure differs:

| Fault | Detector | Recovery |
|---|---|---|
| `StaleFeed(f)` | per-feed max age (1 s reference, 2 s market data) | automatic, after 50 clean checks |
| `SequenceGap` | book sequence discontinuity | resync required |
| `CrossedBook` | bid ≥ ask | resync required |
| `OracleDivergence` | our TWAP integral vs published feed | investigate |
| `ApiAnomaly` | unexpected venue response | human |
| `LossLimit` | daily loss breach | human |

Recovery requires **sustained** health, not one good tick. Flapping back into the
market the instant a gap closes is how a resync bug becomes a position.

### Rare-loss guard

Buying a 97¢ contract that is genuinely 99% to settle at $1 has positive
expectation (+2.0¢) and loses 32× its average win when it loses. Two things ruin
it, neither visible in a Sharpe ratio over a good week:

1. **One loss too large** — sizing capped so a single settlement against us costs
   a bounded fraction of capital, regardless of Kelly. Kelly assumes `p` is known;
   near resolution, `p` is exactly what a settlement dispute or a final-second
   wick puts in question.
2. **The true loss rate is worse than modelled** — a 99%-win strategy losing 3%
   of the time is a losing strategy, and P&L takes a long time to say so. A
   **sequential probability ratio test** asks the right question — "is this
   evidence more consistent with 1% or 3%?" — and in testing detects a 6% true
   loss rate in **under 400 observations**.

### Rate-limit governor

Mirrors the venue's token buckets exactly, including all-or-nothing batch
admission and negative cancel balances. On top:

- `min_requote_ticks` — a sub-tick move does not justify a cancel-replace.
- **Cancel reserve** — a fraction of the cancel bucket is held back so a
  kill-switch `cancel-all` can always be issued. Running out of cancel tokens
  while holding live quotes during a feed fault is the worst state this system
  can reach, and it is preventable by arithmetic.

Effect, measured in the bench: 220,000 ticks produced **5 requotes**. When fair
value is not moving, the engine does not spend budget.

---

## 8. Testing (`px-engine::replay`)

Backtests of market-making strategies are famously easy to fool yourself with.
Four failure modes account for most of it; each is addressed explicitly.

| Failure mode | Mitigation |
|---|---|
| Quotes fill without moving anything | Fills only when counterparty flow reaches the price, with queue position modelled |
| Quotes fill when you were right | Informed flow driven by the *true* path, so stale quotes get run over |
| Your marketable orders rest at your own price | Marketable quotes execute at the venue touch, not our limit |
| The venue knows what you know | `venue_lag_s` parameterises the staleness we trade against |

Plus: our own execution latency (decisions queue and execute one round trip
later, against whatever the book has become), the exact fee schedule, and a
mark-out feedback loop that matures on its due time and feeds the online
adverse-selection estimator through the same path it would in production.

Everything is deterministic — same seed, same result, on any machine, which is
why `px-core::math` implements its own `exp`, `ln`, and `norm_cdf` rather than
calling libm. A test asserts bit-for-bit reproducibility.

### The three sweeps that matter

- **Informed fraction** — what share of counterparty flow knows more than we do.
  This is a claim about the *venue's participant mix*, not our code, and it is
  the hard viability condition. No amount of engineering beats it.
- **Venue lag** — how stale the resting book is. Our edge source.
- **Execution latency** — what a faster path is worth in dollars (§2).

### What the harness caught

Six real bugs, none of which a unit test found:

1. **Crossing quotes priced at our own limit.** The engine emitted bids above the
   offer whenever fair value diverged by more than the half-spread — exactly the
   situation it exists for. Cost: 26¢ per share round-trip, and P&L completely
   insensitive to every strategy parameter. *A loss that does not respond to the
   knobs is never a strategy result.* Fixed with a post-only clamp in the engine
   and touch-price execution in the sim. Baseline went from −$1,476 to +$129.
2. **Volatility bias.** `max(fast, slow)` for pricing (§3).
3. **Over-corrected error bars.** Sampling noise read as regime change (§3).
4. **Duplicate order flood.** The engine re-decides each tick and had no concept
   of an in-flight order, so a flatten became one order per tick — 1,452 orders
   during a single simulated outage. Fixed with order-lifecycle tracking and a
   2 s timeout.
5. **Integer overflow.** A saturating `i32::MAX` margin sentinel wrapped when
   subtracted. Surfaced during a simulated volatility shock.
6. **NaN failing open.** `f64::max(NaN, 0.0)` returns `0.0`, so a NaN model
   uncertainty silently became *zero safety margin* — "trade freely" — instead of
   prohibitive. The most dangerous of the six.

---

## 9. Honest status

**220 tests, 0 failures, 0 warnings.** Critical path p50 832 ns / p99 2.25 µs.

**Not profitable in simulation.** Across 40 seeds on the default configuration:

| | P&L |
|---|---:|
| worst | −$3,483 |
| p10 | −$453 |
| median | −$211 |
| p90 | −$18 |
| best | +$129 |
| profitable | 4 / 40 |

Judge on the p10, not the median. The median is what you tell people; the p10 is
what you live through.

The remaining shortfall is a calibration question, not a bug hunt — the sweeps
now respond sensibly to their parameters, which they did not before. Concretely,
the gap is somewhere in:

- **Reward capture.** Uptime inside the qualifying spread is 6–30%. If the
  liquidity reward is a material part of maker economics — and at $550k/month
  across 5-minute crypto markets it should be — then most of it is currently
  being left on the table. §4 describes the unresolved tension.
- **Spread width.** The `spread_sigma_mult = 1.5` and `safety_k = 1.0` defaults
  were chosen by reasoning, not fitting. They are the obvious first candidates
  for a walk-forward parameter search — with the caveat that fitting parameters
  on a single simulator is the fastest known route to overfitting, and any search
  must be walk-forward with a held-out seed set.
- **Simulator realism.** The flow model is still crude: Poisson arrivals, a fixed
  informed fraction, no order-book queue dynamics beyond first-order queue
  position, and no other adaptive makers. Real books contain competitors who
  reprice when we do.

**What would settle it:** replay against recorded Polymarket book data rather
than a synthetic venue. Every parameter in `SimConfig` is a guess about the real
market until then, and the informed-fraction curve in particular is a claim that
can only be checked against real flow.

---

## 10. What was taken from each framework, and what was not

The framework survey was useful. Not all of it survived contact with this venue.

| Source | Adopted | Rejected, and why |
|---|---|---|
| **Nautilus** | Nanosecond timestamps on every event; time as an explicit parameter (never `Instant::now()` inside a decision); order-lifecycle state; typed `Action`/`Reason` values instead of side effects | **Cross-thread message bus on the hot path.** A bus hop costs 100 ns–1 µs plus cache misses; the entire critical path is 832 ns. Nautilus is fast *despite* its bus, via a single-threaded event loop. The bus belongs at the cold boundary — I/O, logging, telemetry — not between fair value and the risk gate. |
| **Hummingbot** | Inventory skew; kill switch; rate limiter; multi-venue connector shape | **`bid = mid − spread/2 − skew`.** Anchoring on the venue's mid is precisely what §1's one rule forbids. Adopting it would delete the edge. |
| **Freqtrade** | Dry-run mode (the replay harness *is* this); explicit stop-losses on near-resolution; separation of signal / entry / sizing | **Hyperopt as specified.** Searching a parameter space against one synthetic simulator produces a strategy tuned to the simulator. Any search must be walk-forward with held-out seeds, and preferably against recorded data. Also `calculate_edge` as `(proposed − current)/current` is a relative price change, not an edge — the fee-and-slippage-aware version already supersedes it. |
| **Lean** | Portfolio-level statistics (Sharpe, max drawdown, profit factor, payoff ratio, expectancy); correlation-aware buying power; universe selection via the relative-value ranker | Nothing material. |
| **Jesse** | Realistic fees (exact venue formula); slippage via depth walk; **latency simulation** — this one directly produced §2's latency-value table | Percentage-based slippage constants. A prediction-market book is shallow and lumpy; a flat 0.05% is meaningless where the real answer is "the touch holds 25 shares". |
| **CCXT** | Unified adapter shape and error taxonomy for the venue layer | Its synchronous request/response model. Market data must be streaming (`ws-subscriptions-clob`, PING every 10 s); polling would lose by more than every microsecond saved elsewhere. |

**The largest correction the survey suggested and this venue refuses:** the
proposed budget of "Order → Execution: 5 µs" via FIX and kernel bypass. There is
no FIX endpoint, no cross-connect, and no colocation. Signing an EIP-712 order
takes longer than the entire proposed budget. §2 replaces it with measured
numbers and a sweep showing what latency is actually worth.

---

## 11. Build

```bash
cargo test --release              # 220 tests
cargo run --release --bin px-bench    # latency percentiles
cargo run --release --bin px-replay   # scenarios + sweeps
```

No external dependencies. `std` only, `rust-version = 1.75`. The hot path
compiles with `panic = "abort"`: a panic on the critical path is a correctness
failure, and losing the process — letting the dead-man switch cancel every
resting order — is better than continuing with corrupt state.

---

## 12. Not built

Named so nobody assumes otherwise.

- **Venue adapters.** No live connectivity. The `Action` enum is the seam where a
  Polymarket CLOB client and a Kalshi client would attach; neither exists.
- **Order/trade websocket handling**, authentication, EIP-712 signing, the
  dead-man-switch heartbeat.
- **Cross-venue relative value.** The design assumes it; the code is
  single-venue.
- **Persistence and reconciliation.** Restart currently means flat.
- **Historical data replay.** The harness drives a synthetic venue. This is the
  most important missing piece (§9).

---

## Appendix: sources

- [Polymarket — Fees](https://docs.polymarket.com/trading/fees)
- [Polymarket — CLOB Trading Rate Limits](https://docs.polymarket.com/api-reference/trading-rate-limits)
- [Polymarket — Market Making](https://docs.polymarket.com/trading/market-making)
- [Polymarket — Liquidity Rewards](https://docs.polymarket.com/programs/liquidity-rewards)
- [Polymarket — Chainlink TWAP Prices](https://docs.polymarket.com/market-data/chainlink-twap)
- [Polymarket — Market Channel (WebSocket)](https://docs.polymarket.com/api-reference/wss/market)
- [Polymarket — Matching Engine Restarts](https://docs.polymarket.com/trading/matching-engine)
