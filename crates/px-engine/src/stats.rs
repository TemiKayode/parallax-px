//! Portfolio performance tracking.
//!
//! Deliberately separate from the P&L accounting itself: these are the numbers
//! that decide whether the strategy is *worth running*, as opposed to whether it
//! made money last Tuesday.
//!
//! # On judging a market-making strategy
//!
//! A quoting strategy produces a very large number of very small wins and a
//! small number of large losses. That shape flatters every naive statistic:
//!
//! * **Win rate** looks superb and means almost nothing. A near-resolution book
//!   buying 97-cent contracts wins 99% of the time and can still be a losing
//!   strategy. It is reported here only so it can be compared against the
//!   *implied* win rate, which is the comparison that carries information.
//! * **Total P&L** over one run is a single draw from a wide distribution.
//! * **Sharpe** is the least bad summary, but it under-penalises exactly the
//!   left tail this strategy carries, which is why `max_drawdown` and
//!   `profit_factor` are reported alongside it rather than beneath it.
//!
//! The honest gate is: sweep seeds, report the distribution of all four, and
//! require the *worst decile* to be acceptable. One good number is an anecdote.

/// Rolling performance statistics over an equity curve.
#[derive(Clone, Debug)]
pub struct PerformanceTracker {
    /// Interval between equity samples, in seconds. Sets the annualisation.
    sample_dt_s: f64,
    last_equity: Option<f64>,
    returns: Vec<f64>,
    peak: f64,
    max_dd: f64,
    gross_profit: f64,
    gross_loss: f64,
    wins: u64,
    losses: u64,
}

impl PerformanceTracker {
    pub fn new(sample_dt_s: f64) -> Self {
        PerformanceTracker {
            sample_dt_s: sample_dt_s.max(1e-9),
            last_equity: None,
            returns: Vec::new(),
            peak: f64::NEG_INFINITY,
            max_dd: 0.0,
            gross_profit: 0.0,
            gross_loss: 0.0,
            wins: 0,
            losses: 0,
        }
    }

    /// Record an equity observation, in dollars.
    ///
    /// Equity is tracked in absolute dollars rather than as a percentage return
    /// because a prediction-market book can legitimately sit at zero net
    /// exposure, and percentage returns on a zero base are meaningless.
    pub fn sample(&mut self, equity: f64) {
        if !equity.is_finite() {
            return;
        }
        if let Some(prev) = self.last_equity {
            self.returns.push(equity - prev);
        }
        self.last_equity = Some(equity);

        if equity > self.peak {
            self.peak = equity;
        }
        let dd = self.peak - equity;
        if dd > self.max_dd {
            self.max_dd = dd;
        }
    }

    /// Record a closed trade's P&L, in dollars.
    pub fn record_trade(&mut self, pnl: f64) {
        if !pnl.is_finite() {
            return;
        }
        if pnl >= 0.0 {
            self.gross_profit += pnl;
            self.wins += 1;
        } else {
            self.gross_loss += -pnl;
            self.losses += 1;
        }
    }

    /// Annualised Sharpe ratio of the equity increments.
    ///
    /// Risk-free rate is taken as zero: over the horizons this strategy holds
    /// positions — minutes — the carry is immaterial and pretending otherwise
    /// adds a parameter without adding information.
    pub fn sharpe(&self) -> f64 {
        let n = self.returns.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.returns.iter().sum::<f64>() / n as f64;
        let var = self
            .returns
            .iter()
            .map(|r| (r - mean) * (r - mean))
            .sum::<f64>()
            / (n as f64 - 1.0);
        let sd = var.sqrt();
        if sd <= 0.0 {
            return 0.0;
        }
        let periods_per_year = (365.0 * 24.0 * 3600.0) / self.sample_dt_s;
        (mean / sd) * periods_per_year.sqrt()
    }

    /// Largest peak-to-trough decline in equity, in dollars.
    #[inline(always)]
    pub fn max_drawdown(&self) -> f64 {
        self.max_dd
    }

    /// Gross profit divided by gross loss. Above 1.0 is profitable; below 1.3
    /// is generally not worth the operational risk of running the system.
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss <= 0.0 {
            if self.gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            self.gross_profit / self.gross_loss
        }
    }

    pub fn win_rate(&self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 {
            0.0
        } else {
            self.wins as f64 / total as f64
        }
    }

    /// Mean win divided by mean loss. Read together with `win_rate`: the two
    /// multiply to the expectancy, and either alone is a way to mislead
    /// yourself.
    pub fn payoff_ratio(&self) -> f64 {
        if self.wins == 0 || self.losses == 0 {
            return 0.0;
        }
        let avg_win = self.gross_profit / self.wins as f64;
        let avg_loss = self.gross_loss / self.losses as f64;
        if avg_loss <= 0.0 {
            0.0
        } else {
            avg_win / avg_loss
        }
    }

    /// Expected P&L per trade, in dollars. The only statistic here that is
    /// directly actionable.
    pub fn expectancy(&self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 {
            0.0
        } else {
            (self.gross_profit - self.gross_loss) / total as f64
        }
    }

    #[inline(always)]
    pub fn trades(&self) -> u64 {
        self.wins + self.losses
    }

    #[inline(always)]
    pub fn samples(&self) -> usize {
        self.returns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_reports_zeros_not_nans() {
        let p = PerformanceTracker::new(1.0);
        assert_eq!(p.sharpe(), 0.0);
        assert_eq!(p.max_drawdown(), 0.0);
        assert_eq!(p.win_rate(), 0.0);
        assert_eq!(p.expectancy(), 0.0);
        assert_eq!(p.payoff_ratio(), 0.0);
        assert_eq!(p.trades(), 0);
    }

    #[test]
    fn drawdown_measures_peak_to_trough() {
        let mut p = PerformanceTracker::new(1.0);
        for e in [0.0, 100.0, 250.0, 120.0, 180.0, 60.0, 400.0] {
            p.sample(e);
        }
        // Peak 250 -> trough 60 is the largest decline.
        assert!((p.max_drawdown() - 190.0).abs() < 1e-9);
    }

    #[test]
    fn a_steady_gainer_has_infinite_sharpe_guarded_to_finite() {
        // Constant increments have zero variance; we return 0 rather than inf.
        let mut p = PerformanceTracker::new(1.0);
        for i in 0..100 {
            p.sample(i as f64 * 10.0);
        }
        assert_eq!(p.sharpe(), 0.0);
    }

    #[test]
    fn sharpe_is_positive_for_a_noisy_uptrend_and_negative_for_a_downtrend() {
        let mut up = PerformanceTracker::new(1.0);
        let mut down = PerformanceTracker::new(1.0);
        let mut seed = 7u64;
        let mut eu = 0.0;
        let mut ed = 0.0;
        for _ in 0..5000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let n = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            eu += 1.0 + n * 4.0;
            ed += -1.0 + n * 4.0;
            up.sample(eu);
            down.sample(ed);
        }
        assert!(up.sharpe() > 0.0, "up sharpe {}", up.sharpe());
        assert!(down.sharpe() < 0.0, "down sharpe {}", down.sharpe());
    }

    #[test]
    fn profit_factor_and_win_rate_can_disagree() {
        // The near-resolution shape: wins 99% of the time and loses money.
        // This is exactly why win rate alone is not a gate.
        let mut p = PerformanceTracker::new(1.0);
        for _ in 0..99 {
            p.record_trade(0.03);
        }
        p.record_trade(-4.00);

        assert!((p.win_rate() - 0.99).abs() < 1e-9);
        assert!(p.profit_factor() < 1.0, "pf {}", p.profit_factor());
        assert!(p.expectancy() < 0.0, "expectancy {}", p.expectancy());
    }

    #[test]
    fn profit_factor_above_one_means_profitable() {
        let mut p = PerformanceTracker::new(1.0);
        for _ in 0..99 {
            p.record_trade(0.05);
        }
        p.record_trade(-2.00);
        assert!(p.profit_factor() > 1.0);
        assert!(p.expectancy() > 0.0);
    }

    #[test]
    fn all_wins_gives_infinite_profit_factor() {
        let mut p = PerformanceTracker::new(1.0);
        p.record_trade(1.0);
        assert!(p.profit_factor().is_infinite());
    }

    #[test]
    fn payoff_ratio_reads_with_win_rate() {
        let mut p = PerformanceTracker::new(1.0);
        for _ in 0..90 {
            p.record_trade(1.0);
        }
        for _ in 0..10 {
            p.record_trade(-5.0);
        }
        assert!((p.win_rate() - 0.9).abs() < 1e-9);
        assert!((p.payoff_ratio() - 0.2).abs() < 1e-9);
        // 0.9 * 1.0 - 0.1 * 5.0 = +0.4 per trade.
        assert!((p.expectancy() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn nonsense_input_is_ignored() {
        let mut p = PerformanceTracker::new(1.0);
        p.sample(f64::NAN);
        p.sample(f64::INFINITY);
        p.record_trade(f64::NAN);
        assert_eq!(p.samples(), 0);
        assert_eq!(p.trades(), 0);
    }

    #[test]
    fn annualisation_scales_with_sample_interval() {
        // The same return series sampled more finely annualises to a larger
        // Sharpe, which is the standard (and standardly abused) property.
        let build = |dt: f64| {
            let mut p = PerformanceTracker::new(dt);
            let mut seed = 11u64;
            let mut e = 0.0;
            for _ in 0..3000 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let n = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                e += 0.5 + n * 2.0;
                p.sample(e);
            }
            p.sharpe()
        };
        let fine = build(1.0);
        let coarse = build(60.0);
        assert!(fine > coarse, "fine {fine} coarse {coarse}");
    }
}
