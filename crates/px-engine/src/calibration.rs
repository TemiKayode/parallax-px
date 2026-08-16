//! Checks `SimConfig`'s assumed `venue_lag_s` / `venue_noise` /
//! `venue_half_spread` / `venue_depth` against what a real recording
//! actually shows, instead of leaving them as guesses — the README's own
//! words for what every parameter in a simulator config is until it has
//! been checked against real flow.
//!
//! This is deliberately built as measurement, not a new simulation mode:
//! every function here takes recorded data in and returns a plain
//! number or distribution out. `px-replay`'s "IS SimConfig CALIBRATED
//! AGAINST REAL DATA?" section is what turns these into a report; this
//! module is just the arithmetic, kept separate and unit-tested the same
//! way `px-score`'s statistics are, since a calibration check that is
//! itself unverified is not worth trusting.

use crate::recording::{parse_quote_rows, RecordedQuote};

/// Observed half-spread at each snapshot, in probability units (`Px` /
/// `Prob`'s native scale) — the real-data analogue of
/// `SimConfig::venue_half_spread`, which is stored in micro-dollars
/// (divide by `1_000_000.0` to compare against this directly).
pub fn observed_half_spreads(quotes: &[RecordedQuote]) -> Vec<f64> {
    quotes.iter().map(|q| (q.ask - q.bid) / 2.0).collect()
}

/// Observed short-window "noise": each snapshot's mid minus the trailing
/// mean of the `window` snapshots before it — the real-data analogue of
/// `SimConfig::venue_noise`, which perturbs a *synthetic* venue's
/// rebuilt quote around its own (lagged) "true" view by a random amount
/// of this rough magnitude. The first `window` snapshots have no full
/// trailing window and are skipped, not zero-padded — a padded zero
/// would understate the real spread of this distribution for no reason
/// other than list length.
/// The guard above (`window == 0 || quotes.len() <= window`) means every
/// arithmetic and indexing operation below runs only once `quotes.len() >
/// window > 0` is established: `mids.len() - window` cannot underflow,
/// `i - window` is `>= 0` since `i >= window` throughout the loop, and
/// `mids[start..i]`/`mids[i]` index only positions `< mids.len()`.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn observed_noise(quotes: &[RecordedQuote], window: usize) -> Vec<f64> {
    if window == 0 || quotes.len() <= window {
        return Vec::new();
    }
    let mids: Vec<f64> = quotes.iter().map(|q| (q.bid + q.ask) / 2.0).collect();
    let mut out = Vec::with_capacity(mids.len() - window);
    for i in window..mids.len() {
        let start = i - window;
        let rolling_mean: f64 = mids[start..i].iter().sum::<f64>() / window as f64;
        out.push(mids[i] - rolling_mean);
    }
    out
}

/// Mean and (population) standard deviation of a non-empty slice.
/// `None` for an empty input — there is no meaningful mean of nothing,
/// and returning `0.0` would silently look like a real, tight
/// distribution instead of "no data".
pub fn mean_std(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    Some((mean, variance.sqrt()))
}

/// Pearson correlation coefficient between two equal-length series.
/// `None` if the lengths differ, there are fewer than 2 points, or
/// either series has zero variance — a flat series has an undefined
/// correlation with anything, not a zero one.
/// `a.len() == b.len()` is checked above, so indexing both with `i in
/// 0..a.len()` is in range for both by construction.
#[allow(clippy::indexing_slicing)]
pub fn correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return None;
    }
    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

/// Holds the most recent value at or before `t` — same semantics as
/// `recording::lookup`, generalised to a plain `(t_unix, value)` series
/// so it works for a reference-price capture too, not just
/// `RecordedQuote`s.
/// `series.is_empty() || ...` short-circuits before `series[0]` is ever
/// evaluated on an empty slice. `lo`/`hi` are a standard binary search
/// over `series`'s real length — `series.len()` would need to approach
/// `usize::MAX` for `lo + hi` to overflow, and `mid` stays within
/// `[lo, hi) ⊆ [0, series.len())` throughout.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
fn lookup_value(series: &[(f64, f64)], t: f64) -> Option<f64> {
    if series.is_empty() || t < series[0].0 {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = series.len();
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if series[mid].0 <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    series.get(lo).map(|&(_, v)| v)
}

/// The real-data check for `SimConfig::venue_lag_s`: correlation between
/// the venue's own price *changes* and the reference feed's price
/// changes at each of `lags_s`, evaluated at every consecutive pair of
/// real venue observations. If the venue's book genuinely lags a fast
/// reference feed by some amount, returns at that lag should correlate
/// most strongly with the venue's own contemporaneous returns.
///
/// `reference` and `venue` are both `(t_unix, price)` pairs in absolute
/// time — deliberately not `RecordedQuote::t_s`, which
/// `load_recording` normalises independently per market and would
/// misalign two separately-started captures exactly the way
/// `recording::complementarity_error`'s doc comment already explains for
/// the YES/NO case. Returns one `(lag_s, correlation)` pair per
/// requested lag; `None` where there were fewer than 2 usable pairs at
/// that lag (a lag reaching before the reference series' first sample,
/// most commonly).
/// `w[0]`/`w[1]` below: `Slice::windows(2)` guarantees every yielded slice
/// has exactly 2 elements, never fewer — both indices are always in
/// range by that contract.
#[allow(clippy::indexing_slicing)]
pub fn lag_correlation(
    reference: &[(f64, f64)],
    venue: &[(f64, f64)],
    lags_s: &[f64],
) -> Vec<(f64, Option<f64>)> {
    let mut venue_sorted = venue.to_vec();
    venue_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    lags_s
        .iter()
        .map(|&lag| {
            let mut venue_returns = Vec::new();
            let mut reference_returns = Vec::new();
            for w in venue_sorted.windows(2) {
                let (t_prev, v_prev) = w[0];
                let (t_cur, v_cur) = w[1];
                let (Some(r_prev), Some(r_cur)) = (
                    lookup_value(reference, t_prev - lag),
                    lookup_value(reference, t_cur - lag),
                ) else {
                    continue;
                };
                venue_returns.push(v_cur - v_prev);
                reference_returns.push(r_cur - r_prev);
            }
            (lag, correlation(&venue_returns, &reference_returns))
        })
        .collect()
}

/// Convenience: loads one market's real touch series as absolute-time
/// `(t_unix, mid)` pairs, for `lag_correlation`'s `venue` argument.
pub fn venue_mid_series(text: &str, market: &str) -> Vec<(f64, f64)> {
    parse_quote_rows(text, market)
        .into_iter()
        .map(|(t_unix, q)| (t_unix, (q.bid + q.ask) / 2.0))
        .collect()
}

#[cfg(test)]
// Test-only: an index or unwrap that would be wrong is exactly the test
// failing, which is the correct and intended outcome — not a production
// safety concern.
#[allow(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::recording::RecordedQuote;

    fn q(t_s: f64, bid: f64, ask: f64) -> RecordedQuote {
        RecordedQuote {
            t_s,
            bid,
            bid_size: 1.0,
            ask,
            ask_size: 1.0,
        }
    }

    #[test]
    fn observed_half_spreads_matches_hand_computed_values() {
        let quotes = vec![q(0.0, 0.49, 0.51), q(1.0, 0.40, 0.60)];
        let spreads = observed_half_spreads(&quotes);
        assert_eq!(spreads.len(), 2);
        assert!((spreads[0] - 0.01).abs() < 1e-9);
        assert!((spreads[1] - 0.10).abs() < 1e-9);
    }

    #[test]
    fn observed_noise_skips_the_unfilled_leading_window() {
        // Mids: 0.50, 0.50, 0.50, 0.60 — window 3: only the 4th point has
        // a full trailing window (mean of the first three, 0.50).
        let quotes = vec![
            q(0.0, 0.49, 0.51),
            q(1.0, 0.49, 0.51),
            q(2.0, 0.49, 0.51),
            q(3.0, 0.59, 0.61),
        ];
        let noise = observed_noise(&quotes, 3);
        assert_eq!(noise.len(), 1);
        assert!((noise[0] - 0.10).abs() < 1e-9);
    }

    #[test]
    fn observed_noise_of_too_short_a_series_is_empty() {
        let quotes = vec![q(0.0, 0.49, 0.51), q(1.0, 0.49, 0.51)];
        assert!(observed_noise(&quotes, 5).is_empty());
        assert!(observed_noise(&quotes, 0).is_empty());
    }

    #[test]
    fn mean_std_matches_a_hand_checked_case() {
        // [2, 4, 4, 4, 5, 5, 7, 9]: mean 5, population std 2.
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&v).unwrap();
        assert!((mean - 5.0).abs() < 1e-9);
        assert!((std - 2.0).abs() < 1e-9);
    }

    #[test]
    fn mean_std_of_empty_is_none() {
        assert!(mean_std(&[]).is_none());
    }

    #[test]
    fn correlation_of_perfectly_linear_series_is_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let c = correlation(&a, &b).unwrap();
        assert!((c - 1.0).abs() < 1e-9, "c={c}");
    }

    #[test]
    fn correlation_of_inversely_linear_series_is_negative_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let c = correlation(&a, &b).unwrap();
        assert!((c + 1.0).abs() < 1e-9, "c={c}");
    }

    #[test]
    fn correlation_of_a_flat_series_is_undefined_not_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let flat = vec![5.0, 5.0, 5.0];
        assert!(correlation(&a, &flat).is_none());
    }

    #[test]
    fn correlation_needs_matching_lengths_and_at_least_two_points() {
        assert!(correlation(&[1.0, 2.0], &[1.0]).is_none());
        assert!(correlation(&[1.0], &[1.0]).is_none());
    }

    #[test]
    fn lag_correlation_finds_a_known_injected_lag() {
        // Reference: a single step up at t=100, otherwise flat.
        let reference: Vec<(f64, f64)> = (0..200)
            .map(|i| (i as f64, if i < 100 { 0.0 } else { 1.0 }))
            .collect();
        // Venue: the exact same step, but delayed by 5s, sampled every
        // 3s to mimic real irregular-ish polling.
        let venue: Vec<(f64, f64)> = (0..67)
            .map(|i| {
                let t = i as f64 * 3.0;
                let v = if t < 105.0 { 0.0 } else { 1.0 };
                (t, v)
            })
            .collect();
        let lags = [0.0, 3.0, 5.0, 6.0, 9.0, 12.0];
        let results = lag_correlation(&reference, &venue, &lags);
        // The lag closest to the true 5s delay should score at least as
        // well as every other candidate lag tried.
        let best = results
            .iter()
            .filter_map(|&(lag, c)| c.map(|c| (lag, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (best_lag, best_corr) = best.expect("expected at least one usable lag");
        assert!(best_corr > 0.9, "best correlation too weak: {best_corr}");
        assert!(
            (best_lag - 5.0).abs() <= 3.0,
            "best lag {best_lag} not close to the true 5s delay"
        );
    }

    #[test]
    fn lag_correlation_reports_none_when_the_lag_reaches_before_data_starts() {
        let reference: Vec<(f64, f64)> = vec![(100.0, 1.0), (110.0, 2.0)];
        let venue: Vec<(f64, f64)> = vec![(100.0, 0.5), (110.0, 0.6)];
        // A 1000s lag shifts every venue timestamp to before the
        // reference series' first sample.
        let results = lag_correlation(&reference, &venue, &[1000.0]);
        assert_eq!(results, vec![(1000.0, None)]);
    }
}
