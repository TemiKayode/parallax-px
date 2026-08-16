//! Position sizing.
//!
//! # Kelly for a binary contract
//!
//! Buying one share at price `c` with true probability `p` wins `1 - c` with
//! probability `p` and loses `c` with probability `1 - p`. Net odds are
//! `b = (1 - c)/c`, and the log-optimal stake fraction is
//!
//! ```text
//!   f* = (p*b - (1-p)) / b = (p - c) / (1 - c)
//! ```
//!
//! Note what `f*` is a fraction *of*: the bankroll to stake, not the number of
//! shares. Since each share costs `c`, share count is `f* * W / c`. Confusing
//! the two is the classic way to end up with an eight-times-Kelly position on a
//! 12-cent contract.
//!
//! # Why fractional
//!
//! Full Kelly is optimal only if `p` is *known*. Ours is estimated, and Kelly's
//! growth curve is brutally asymmetric around the optimum: betting 2x Kelly has
//! zero expected log growth, and anything beyond that is negative. Overstating
//! `p` by a little produces the same effect as deliberately overbetting.
//!
//! At 20% of full Kelly we capture roughly 36% of the theoretical growth rate
//! while cutting the variance of the growth rate by 25x, and — the part that
//! matters — an estimate that is wrong by a factor of two still leaves us
//! comfortably under full Kelly rather than in the ruinous zone.

use px_core::{Px, Qty, Usd};

/// Full-Kelly stake fraction for buying a binary at `entry` with fair
/// probability `p`. Returns 0 when the bet has no edge.
#[inline]
pub fn kelly_fraction(p: f64, entry: Px) -> f64 {
    let c = entry.as_f64();
    if !(c > 0.0 && c < 1.0 && p.is_finite()) {
        return 0.0;
    }
    let f = (p - c) / (1.0 - c);
    if f.is_finite() && f > 0.0 {
        f.min(1.0)
    } else {
        0.0
    }
}

/// Kelly fraction where the effective cost includes the taker fee. A contract
/// bought at 52c with a 1.75c fee is really a 53.75c contract, and sizing off
/// the quoted price overstates the edge every single time.
#[inline]
pub fn kelly_fraction_after_fee(p: f64, entry: Px, fee_per_share_micro: f64) -> f64 {
    let effective = Px((entry.0 as f64 + fee_per_share_micro) as i32).clamp_unit();
    kelly_fraction(p, effective)
}

/// Share count from a fractional-Kelly stake.
///
/// `fraction` is the Kelly multiplier, e.g. 0.2 for one-fifth Kelly.
pub fn kelly_shares(p: f64, entry: Px, bankroll: Usd, fraction: f64) -> Qty {
    let c = entry.as_f64();
    if c <= 0.0 {
        return Qty::ZERO;
    }
    let f = kelly_fraction(p, entry) * fraction.clamp(0.0, 1.0);
    let stake = bankroll.as_f64() * f;
    let shares = stake / c;
    if !shares.is_finite() || shares <= 0.0 {
        Qty::ZERO
    } else {
        Qty((shares * 1e6) as i64)
    }
}

/// Growth rate of the log-optimal bet, for diagnostics: how much this edge is
/// actually worth per unit of capital committed.
pub fn expected_log_growth(p: f64, entry: Px, fraction: f64) -> f64 {
    let c = entry.as_f64();
    if !(c > 0.0 && c < 1.0) {
        return 0.0;
    }
    let f = kelly_fraction(p, entry) * fraction.clamp(0.0, 1.0);
    if f <= 0.0 {
        return 0.0;
    }
    let b = (1.0 - c) / c;
    p * px_core::math::ln(1.0 + f * b) + (1.0 - p) * px_core::math::ln(1.0 - f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Both calls land on `kelly_fraction`'s `else { 0.0 }` branch (f == 0.0
    // exactly for p=0.50/c=0.50, f < 0.0 for p=0.40/c=0.50) — a literal,
    // not a rounding coincidence.
    #[allow(clippy::float_cmp)]
    fn no_edge_means_no_bet() {
        assert_eq!(kelly_fraction(0.50, Px(500_000)), 0.0);
        assert_eq!(kelly_fraction(0.40, Px(500_000)), 0.0);
        assert_eq!(
            kelly_shares(0.40, Px(500_000), Usd::dollars(10_000), 0.2),
            Qty::ZERO
        );
    }

    #[test]
    fn kelly_matches_the_closed_form() {
        // p = 0.60, c = 0.50 -> f* = (0.60-0.50)/(1-0.50) = 0.20
        assert!((kelly_fraction(0.60, Px(500_000)) - 0.20).abs() < 1e-12);
        // p = 0.99, c = 0.97 -> f* = 0.02/0.03 = 0.6667
        assert!((kelly_fraction(0.99, Px(970_000)) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn kelly_is_the_argmax_of_log_growth() {
        // Numerically confirm that the closed form really is the optimum.
        let p = 0.62;
        let c = Px(480_000);
        let f_star = kelly_fraction(p, c);
        let best = expected_log_growth(p, c, 1.0);
        for step in 1..40 {
            let mult = step as f64 / 20.0; // 0.05 .. 1.95 of full Kelly
            if (mult - 1.0).abs() < 1e-9 {
                continue;
            }
            let g = expected_log_growth(p, c, mult);
            assert!(g <= best + 1e-12, "mult {mult} beat full Kelly");
        }
        assert!(f_star > 0.0);
    }

    #[test]
    fn overbetting_destroys_growth_far_faster_than_underbetting() {
        // The asymmetry that justifies the 20% multiplier.
        //
        // The textbook "2x Kelly gives exactly zero growth" identity holds in
        // the continuous/Gaussian limit; for a discrete binary bet it is close
        // but not exact. What *is* exact, and what matters, is the shape: at
        // half Kelly we keep three quarters of the growth, while at double
        // Kelly the growth is already negative.
        let p = 0.60;
        let c = Px(500_000);

        let full = expected_log_growth(p, c, 1.0);
        let half = expected_log_growth(p, c, 0.5);
        let fifth = expected_log_growth(p, c, 0.2);

        assert!(half / full > 0.7, "half Kelly kept {}", half / full);
        assert!(fifth / full > 0.3, "one fifth Kelly kept {}", fifth / full);

        // Double Kelly: growth has already gone negative.
        let f2 = 2.0 * kelly_fraction(p, c);
        let b = (1.0 - c.as_f64()) / c.as_f64();
        let g2 = p * px_core::math::ln(1.0 + f2 * b) + (1.0 - p) * px_core::math::ln(1.0 - f2);
        assert!(g2 <= 0.0, "growth at 2x Kelly = {g2}");
        assert!(g2.abs() < full, "2x Kelly should be near the zero crossing");

        // Triple Kelly is catastrophic, and comfortably worse than 2x.
        let f3 = 3.0 * kelly_fraction(p, c);
        let g3 = p * px_core::math::ln(1.0 + f3 * b) + (1.0 - p) * px_core::math::ln(1.0 - f3);
        assert!(g3 < g2);

        // Which is the whole point: a p-estimate wrong by 3x at one-fifth Kelly
        // still lands under full Kelly. At full Kelly it lands in the red.
    }

    #[test]
    fn fee_adjusted_kelly_is_strictly_smaller() {
        // 52c contract, fair 56c, crypto fee 1.747c per share.
        let raw = kelly_fraction(0.56, Px(520_000));
        let net = kelly_fraction_after_fee(0.56, Px(520_000), 17_472.0);
        assert!(net < raw, "raw {raw} net {net}");
        assert!(net > 0.0);
    }

    #[test]
    // The fee-adjusted price pushes `f <= 0.0`, hitting the same literal
    // `else { 0.0 }` branch as `no_edge_means_no_bet`.
    #[allow(clippy::float_cmp)]
    fn a_fee_can_erase_the_bet_entirely() {
        // Fair 53c against a 52c offer looks like edge until the fee is added.
        let net = kelly_fraction_after_fee(0.53, Px(520_000), 17_472.0);
        assert_eq!(net, 0.0);
    }

    #[test]
    fn share_count_respects_the_stake_identity() {
        // shares * price must equal the intended stake.
        let bankroll = Usd::dollars(100_000);
        let p = 0.60;
        let c = Px(500_000);
        let frac = 0.2;
        let q = kelly_shares(p, c, bankroll, frac);
        let stake = q.as_f64() * c.as_f64();
        let expected = bankroll.as_f64() * kelly_fraction(p, c) * frac;
        assert!(
            (stake - expected).abs() < 1.0,
            "stake {stake} expected {expected}"
        );
    }

    #[test]
    fn one_fifth_kelly_sizes_a_concrete_case() {
        // $100k bankroll, fair 60c, buying at 50c, 20% Kelly.
        // f* = 0.20, stake = 0.2 * 0.2 * 100000 = $4,000, at 50c = 8,000 shares.
        let q = kelly_shares(0.60, Px(500_000), Usd::dollars(100_000), 0.2);
        assert!((q.as_f64() - 8_000.0).abs() < 1.0, "{}", q.as_f64());
    }

    #[test]
    // `Px::ZERO`/`Px::ONE` fail the `c > 0.0 && c < 1.0` guard, and `NAN`
    // fails `p.is_finite()` — all three hit `kelly_fraction`'s explicit
    // `return 0.0;`, not a computed value.
    #[allow(clippy::float_cmp)]
    fn degenerate_prices_are_handled() {
        assert_eq!(kelly_fraction(0.9, Px::ZERO), 0.0);
        assert_eq!(kelly_fraction(0.9, Px::ONE), 0.0);
        assert_eq!(kelly_fraction(f64::NAN, Px(500_000)), 0.0);
        assert_eq!(
            kelly_shares(0.9, Px::ZERO, Usd::dollars(1000), 0.2),
            Qty::ZERO
        );
    }

    #[test]
    fn certainty_is_capped_at_full_bankroll() {
        // p = 1.0 gives f* = 1.0; we must not return something larger.
        assert!(kelly_fraction(1.0, Px(500_000)) <= 1.0);
    }
}
