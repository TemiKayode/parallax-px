# Using this as a work sample

Notes on positioning. Career markets vary and none of this is guaranteed — but
the reasoning behind it is worth more than the specific suggestions.

---

## Don't try to sell it as software

The market for trading systems is a lemons market, and everyone sophisticated
enough to be a buyer already knows it:

> If it makes money, why are you selling it instead of running it?

There is no good answer to that question, which is why the honest sellers in
this space sell *infrastructure* (data, execution plumbing, risk systems) and
the people selling *strategies* are usually selling to people who cannot
evaluate them. You do not want to be in the second group, and the first group is
a crowded market against well-funded incumbents.

The exception that occasionally works: license the **components**, not the
strategy. A correct venue fee model, a TWAP settlement pricer, a rate-limit
governor — these are boring, verifiable, and someone building their own system
might pay to skip. But it is a small market and a slow sale.

**Selling it as a credential is a far better trade.** Which is fortunate,
because that is where this codebase is genuinely strong.

---

## Why this is a good work sample

Most candidate trading projects show a backtest with a beautiful equity curve.
Experienced people discount those to roughly zero, because they know that
almost all of them are overfit, have a fee bug, or fill at prices that were
never available.

This project shows something rarer and much harder to fake: **a rigorous
negative result with a diagnosed cause.** That signals the thing firms actually
screen for, which is not "can you find alpha" — it is "will you fool yourself,
and will you notice when you have."

Five things here that read as senior:

**1. Reading the venue docs changed the strategy.**
The taker fee on crypto markets is `0.07 × p × (1−p)` — 1.75¢/share at the
money, while makers pay zero and collect a rebate. A 2.1¢ gap between the two
sides of the same trade. That single fact inverts the design from "cross the
spread on any disagreement" to "rest and almost never cross." Most people
assume fees are a rounding error and discover otherwise with real money.

**2. The TWAP derivation.**
Crypto 5m/15m/4h markets settle on a time-weighted average, not the spot print.
Inside the settlement window variance decays as `r³`, so standard deviation goes
as `r^1.5`, not `√r`. Ten seconds into a sixty-second window, remaining
uncertainty is 0.68% of what it was — a `√τ` model says 41%. This is a
closed-form result with a clean derivation and it is genuinely the most
interesting thing in the repo.

**3. The latency reframing.**
The brief asked for sub-10µs tick-to-trade. The critical path does run in ~1µs
(measured, p99 1.9µs). But the venue meters orders through per-signer token
buckets: at entry tier, ten markets quoted two-sided is **two requotes per
second per market**. Microseconds buy decision *freshness*, not throughput.
Recognising that the stated requirement was the wrong constraint — and saying so
with the arithmetic — is the kind of thing that distinguishes an engineer from
an implementer.

**4. The harness found ~20 bugs, several in the safety code.**
A sign flip in `exp` reachable through `norm_pdf` (returned −6.6e303 where a
tiny positive belonged, and would have inverted the inventory skew). A NaN that
turned the safety margin to *zero* instead of prohibitive. A risk gate that
wedged permanently after normal turnover because gross exposure measured
turnover, not position. An unvalidated feed price that could abort the process
via out-of-bounds index. Each was silent. None produced an error.

**5. `px-score` — building the thing that could falsify the work.**
Brier skill score against the venue mid, Murphy decomposition, calibration bins,
paired significance test. It answered the question on the first run, and the
answer was *no*. Resolution 0.092 (real information) but skill −0.064 (worse
than reading the price), driven by catastrophic overconfidence in one bin: the
model said 2% where reality was 22%.

Point 5 is the one to lead with. Anyone senior will recognise what it means that
you built the falsifier and then reported the result.

---

## Where to look

**Prop trading firms.** Jane Street, Jump, IMC, Optiver, DRW, SIG, Tower, Hudson
River, XTX, Akuna, Radix. They hire for exactly the reasoning above and they
interview hard on it.

**Crypto-native market makers.** Wintermute, GSR, Auros, Keyrock, B2C2, Amber,
Cumberland. Closer to this domain and often faster-moving hiring processes.

**The venues themselves.** Polymarket and Kalshi hire engineers, and you have
read their microstructure documentation more carefully than most applicants ever
will. The fee-curve and TWAP-settlement analysis is directly relevant to their
own product decisions.

**Rust infrastructure roles generally.** Zero dependencies, no `unsafe`, 263
tests, deterministic transcendentals so replay is bit-exact. That stands on its
own without anyone caring about prediction markets.

---

## Before showing it to anyone

Small things that cost credibility disproportionately:

- [ ] `cargo fmt --all` (two files still drift; CI will go red on this alone)
- [ ] Add a LICENSE
- [ ] Fix the clustered-standard-error flaw in `px-score` — forecasts within one
      session share an outcome, so `t = +10.2` is overstated. Finding and fixing
      a statistical flaw in *your own* evaluation code is itself a strong signal.
- [ ] Make sure the README leads with what it is, not what it aspires to

---

## How to talk about it

**Lead with the findings, not the code.** A short write-up titled something like
*"Five things about prediction-market microstructure that cost me a working
strategy"* will get read; a repo link will not.

**Never claim it is profitable.** It isn't, and anyone worth working for will
ask a follow-up question you cannot survive if you have overstated it. The true
answer is better anyway:

> It doesn't beat the venue mid. Skill score −0.064 against the market price,
> though the Murphy decomposition shows resolution of 0.09 — there's real
> information in it, it's just badly miscalibrated. The model says 2% in a bin
> where the outcome happens 22% of the time, and I think that's the cubic
> variance collapse amplifying a volatility underestimate. Recalibration is the
> next experiment.

That paragraph is a better interview than any equity curve, because it is
checkable, specific, and demonstrates you know what would falsify your own work.

**Expect to be tested on the maths.** The TWAP variance derivation, why Brier is
a proper scoring rule and directional accuracy is not, why fractional Kelly.
Know these cold — they are the parts of the project that are actually yours.
