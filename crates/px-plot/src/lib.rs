//! `px-plot` — charts, with no dependencies.
//!
//! # Why SVG, and why hand-rolled
//!
//! A plotting crate would pull in fifty transitive dependencies to draw four
//! pictures. SVG is text: a chart is a `String`, produced by `format!`. That
//! keeps the zero-dependency property of the workspace intact, and the output
//! renders natively in a browser, in a GitHub README, and in any document — no
//! image pipeline, no headless renderer, no build step.
//!
//! There is also a terminal renderer here, because the fastest feedback loop is
//! the one that does not require opening a file.
//!
//! # Which charts, and why these
//!
//! Not decoration. Each one exists because it makes a specific failure visible
//! at a glance that a table of numbers hides:
//!
//! * **Reliability diagram** — the single most important chart in the project.
//!   A scalar Brier score says "worse than the market". This says *where*: the
//!   model claims 2% in a bin where the outcome happens 22% of the time. You
//!   cannot see that in a summary statistic, and it points straight at the fix.
//! * **Variance shape** — `r³` against `√τ`. The chart that explains why
//!   TWAP-settled contracts are mispriced by models that assume spot settlement.
//! * **Fee curve** — the downward parabola that inverts the strategy, with the
//!   maker rebate on the same axes. Shows at a glance why crossing at the money
//!   costs two ticks and why near-resolution is an order of magnitude cheaper.
//! * **Equity curve with drawdown** — because peak-to-trough is what you live
//!   through, and a final P&L number conceals it entirely.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

use px_score::Scorecard;

const W: f64 = 640.0;
const H: f64 = 440.0;
const PAD: f64 = 56.0;

const BG: &str = "#ffffff";
const INK: &str = "#1a1a1a";
const GRID: &str = "#e4e4e4";
const MUTED: &str = "#8a8a8a";
const ACCENT: &str = "#c2410c";
const GOOD: &str = "#15803d";
const BAD: &str = "#b91c1c";

/// A finished chart.
#[derive(Clone, Debug)]
pub struct Svg(pub String);

impl Svg {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for Svg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

fn head(title: &str, subtitle: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">
<rect width="{W}" height="{H}" fill="{BG}"/>
<text x="{PAD}" y="28" font-size="15" font-weight="600" fill="{INK}">{}</text>
<text x="{PAD}" y="46" font-size="11" fill="{MUTED}">{}</text>
"#,
        esc(title),
        esc(subtitle)
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Plot area in SVG coordinates.
struct Frame {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl Frame {
    fn default_frame() -> Frame {
        Frame {
            x0: PAD,
            y0: 64.0,
            x1: W - 24.0,
            y1: H - PAD,
        }
    }
    fn sx(&self, t: f64) -> f64 {
        self.x0 + t.clamp(0.0, 1.0) * (self.x1 - self.x0)
    }
    fn sy(&self, t: f64) -> f64 {
        self.y1 - t.clamp(0.0, 1.0) * (self.y1 - self.y0)
    }
}

fn axes(
    f: &Frame,
    xlab: &str,
    ylab: &str,
    xticks: &[(f64, String)],
    yticks: &[(f64, String)],
) -> String {
    let mut s = String::new();
    for (t, lab) in yticks {
        let y = f.sy(*t);
        s.push_str(&format!(
            r#"<line x1="{:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="{GRID}" stroke-width="1"/>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{MUTED}" text-anchor="end">{}</text>
"#,
            f.x0,
            f.x1,
            f.x0 - 6.0,
            y + 3.5,
            esc(lab)
        ));
    }
    for (t, lab) in xticks {
        let x = f.sx(*t);
        s.push_str(&format!(
            r#"<text x="{x:.1}" y="{:.1}" font-size="10" fill="{MUTED}" text-anchor="middle">{}</text>
"#,
            f.y1 + 16.0,
            esc(lab)
        ));
    }
    s.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{INK}" stroke-width="1.2"/>
<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{INK}" stroke-width="1.2"/>
<text x="{:.1}" y="{:.1}" font-size="11" fill="{INK}" text-anchor="middle">{}</text>
<text x="14" y="{:.1}" font-size="11" fill="{INK}" text-anchor="middle" transform="rotate(-90 14 {:.1})">{}</text>
"#,
        f.x0, f.y0, f.x0, f.y1,
        f.x0, f.y1, f.x1, f.y1,
        (f.x0 + f.x1) / 2.0, H - 14.0, esc(xlab),
        (f.y0 + f.y1) / 2.0, (f.y0 + f.y1) / 2.0, esc(ylab)
    ));
    s
}

// ---------------------------------------------------------------------------
// 1. Reliability diagram
// ---------------------------------------------------------------------------

/// Calibration curve: what we said against what happened.
///
/// The diagonal is perfect calibration. Points below it are overconfidence in
/// YES; points above are underconfidence. Marker area is proportional to the
/// number of forecasts in the bin, because a bin holding nine forecasts and a
/// bin holding three thousand should not look equally important — a detail
/// that is routinely omitted and routinely misleads.
pub fn reliability(s: &Scorecard) -> Svg {
    let f = Frame::default_frame();
    let mut out = head(
        "Reliability — is the model calibrated?",
        &format!(
            "n = {}   skill vs venue mid = {:+.4}   t(clustered) = {:+.2} over {} clusters   reliability = {:.5}   resolution = {:.5}",
            s.n, s.skill_score, s.t_stat_clustered, s.n_clusters, s.reliability, s.resolution
        ),
    );

    let ticks: Vec<(f64, String)> = (0..=5)
        .map(|i| (i as f64 / 5.0, format!("{:.1}", i as f64 / 5.0)))
        .collect();
    out.push_str(&axes(
        &f,
        "forecast probability",
        "observed frequency",
        &ticks,
        &ticks,
    ));

    // Perfect-calibration diagonal.
    out.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{MUTED}">perfect calibration</text>
"#,
        f.sx(0.0), f.sy(0.0), f.sx(1.0), f.sy(1.0),
        f.sx(0.55), f.sy(0.50)
    ));

    let max_n = s.bins.iter().map(|b| b.n).max().unwrap_or(1).max(1) as f64;

    // Connecting line through populated bins.
    let pts: Vec<String> = s
        .bins
        .iter()
        .filter(|b| b.n > 0)
        .map(|b| format!("{:.1},{:.1}", f.sx(b.mean_forecast), f.sy(b.observed)))
        .collect();
    if pts.len() > 1 {
        out.push_str(&format!(
            r#"<polyline points="{}" fill="none" stroke="{ACCENT}" stroke-width="1.8"/>
"#,
            pts.join(" ")
        ));
    }

    for b in s.bins.iter().filter(|b| b.n > 0) {
        let x = f.sx(b.mean_forecast);
        let y = f.sy(b.observed);
        let r = 3.0 + 9.0 * (b.n as f64 / max_n).sqrt();
        let colour = if b.error().abs() > 0.10 { BAD } else { ACCENT };
        out.push_str(&format!(
            r#"<circle cx="{x:.1}" cy="{y:.1}" r="{r:.1}" fill="{colour}" fill-opacity="0.75"/>
"#
        ));
        // Call out badly miscalibrated bins — the ones worth acting on.
        if b.error().abs() > 0.10 {
            out.push_str(&format!(
                r#"<text x="{:.1}" y="{y:.1}" font-size="9" fill="{BAD}">said {:.0}%, was {:.0}% (n={})</text>
"#,
                x + r + 5.0,
                b.mean_forecast * 100.0,
                b.observed * 100.0,
                b.n
            ));
        }
    }

    out.push_str("</svg>\n");
    Svg(out)
}

// ---------------------------------------------------------------------------
// 2. Variance shape — the TWAP result
// ---------------------------------------------------------------------------

/// Remaining standard deviation against time, TWAP settlement versus spot.
///
/// The chart that explains the edge thesis in one picture: inside the
/// settlement window variance falls as `r³`, so standard deviation falls as
/// `r^1.5`. A model assuming spot settlement and `√τ` decay is not slightly
/// wrong late in the window, it is wrong by an order of magnitude — and always
/// in the direction of pricing a decided market as a coin flip.
pub fn variance_shape(window_s: f64) -> Svg {
    // A zero-length window is degenerate: `r^3 / W^3` is 0/0. Fall back to a
    // nominal window rather than emitting NaN into the coordinates, which an
    // SVG renderer silently drops — producing a blank chart with no error.
    let window_s = if window_s > 0.0 { window_s } else { 60.0 };
    let f = Frame::default_frame();
    let mut out = head(
        "Why TWAP settlement is mispriced",
        &format!(
            "remaining standard deviation, {:.0}s settlement window, normalised to 1.0 at window open",
            window_s
        ),
    );

    let xt: Vec<(f64, String)> = (0..=4)
        .map(|i| {
            let frac = i as f64 / 4.0;
            (frac, format!("{:.0}s", frac * window_s))
        })
        .collect();
    let yt: Vec<(f64, String)> = (0..=4)
        .map(|i| (i as f64 / 4.0, format!("{:.2}", i as f64 / 4.0)))
        .collect();
    out.push_str(&axes(&f, "time remaining", "relative sd", &xt, &yt));

    let n = 200;
    let mut twap = Vec::new();
    let mut spot = Vec::new();
    for i in 0..=n {
        let frac = i as f64 / n as f64; // fraction of window remaining
        let r = frac * window_s;
        // TWAP inside the window: Var = sigma^2 r^3 / (3 W^2), normalised so
        // that r = W gives 1.0.
        let t = (r * r * r / (window_s * window_s * window_s)).sqrt();
        // Spot settlement: sd ~ sqrt(r), same normalisation.
        let s_ = frac.sqrt();
        twap.push(format!("{:.1},{:.1}", f.sx(frac), f.sy(t)));
        spot.push(format!("{:.1},{:.1}", f.sx(frac), f.sy(s_)));
    }

    out.push_str(&format!(
        r#"<polyline points="{}" fill="none" stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 3"/>
<polyline points="{}" fill="none" stroke="{ACCENT}" stroke-width="2.4"/>
"#,
        spot.join(" "),
        twap.join(" ")
    ));

    // Annotate the divergence at 1/6 of the window remaining (10s of 60s).
    let frac: f64 = 1.0 / 6.0;
    let t = (frac * frac * frac).sqrt();
    let s_ = frac.sqrt();
    out.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{BAD}" stroke-width="1.4"/>
<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{ACCENT}"/>
<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{MUTED}"/>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{BAD}">6x apart at {:.0}s left</text>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{MUTED}">spot / sqrt(t) model</text>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{ACCENT}">TWAP, correct</text>
"#,
        f.sx(frac),
        f.sy(t),
        f.sx(frac),
        f.sy(s_),
        f.sx(frac),
        f.sy(t),
        f.sx(frac),
        f.sy(s_),
        f.sx(frac) + 8.0,
        f.sy((t + s_) / 2.0),
        frac * window_s,
        f.sx(0.62),
        f.sy(0.86),
        f.sx(0.62),
        f.sy(0.70)
    ));

    out.push_str("</svg>\n");
    Svg(out)
}

// ---------------------------------------------------------------------------
// 3. Fee curve
// ---------------------------------------------------------------------------

/// Taker fee and maker rebate against price.
///
/// `fee = rate × p × (1−p)` is a downward parabola peaking at 50 cents. Two
/// things become obvious that a single number does not convey: the gap between
/// the two sides of the same trade is widest exactly at the money, and it
/// collapses toward the extremes — which is why the near-resolution structure
/// survives a fee schedule that makes at-the-money taking unviable.
pub fn fee_curve(taker_rate: f64, maker_rebate: f64, label: &str) -> Svg {
    let f = Frame::default_frame();
    let peak = taker_rate * 0.25;
    // Geopolitics markets are fee-free, so `peak` is legitimately zero and the
    // normalisation divides by it. Guard rather than special-case the caller:
    // a flat line at zero is the correct picture for a zero-fee market.
    let norm = if peak > 0.0 { peak } else { 1.0 };
    let mut out = head(
        &format!("Fee curve — {label}"),
        &format!(
            "taker = {:.2} x p x (1-p);  peak {:.2} cents per share at 50c;  maker rebate {:.0}% of it",
            taker_rate,
            peak * 100.0,
            maker_rebate * 100.0
        ),
    );

    let xt: Vec<(f64, String)> = (0..=5)
        .map(|i| (i as f64 / 5.0, format!("{:.0}c", i as f64 / 5.0 * 100.0)))
        .collect();
    let yt: Vec<(f64, String)> = (0..=4)
        .map(|i| {
            let v = i as f64 / 4.0 * norm * 100.0;
            (i as f64 / 4.0, format!("{v:.2}"))
        })
        .collect();
    out.push_str(&axes(&f, "trade price", "cents per share", &xt, &yt));

    let n = 200;
    let mut taker = Vec::new();
    let mut maker = Vec::new();
    for i in 0..=n {
        let p = i as f64 / n as f64;
        let fee = taker_rate * p * (1.0 - p);
        taker.push(format!("{:.1},{:.1}", f.sx(p), f.sy(fee / norm)));
        maker.push(format!(
            "{:.1},{:.1}",
            f.sx(p),
            f.sy(fee * maker_rebate / norm)
        ));
    }
    out.push_str(&format!(
        r#"<polyline points="{}" fill="none" stroke="{BAD}" stroke-width="2.4"/>
<polyline points="{}" fill="none" stroke="{GOOD}" stroke-width="2.4"/>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{BAD}">taker pays</text>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{GOOD}">maker receives</text>
"#,
        taker.join(" "),
        maker.join(" "),
        f.sx(0.5) + 8.0,
        f.sy(0.95),
        f.sx(0.5) + 8.0,
        f.sy(maker_rebate) - 6.0
    ));

    // The gap at the money — the number that inverted the strategy.
    let gap = peak * (1.0 + maker_rebate) * 100.0;
    out.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="3 3"/>
<text x="{:.1}" y="{:.1}" font-size="10" font-weight="600" fill="{INK}">{gap:.2}c gap between the two sides of the same trade</text>
"#,
        f.sx(0.5), f.sy(1.0), f.sx(0.5), f.sy(maker_rebate),
        f.sx(0.06), f.sy(0.42)
    ));

    out.push_str("</svg>\n");
    Svg(out)
}

// ---------------------------------------------------------------------------
// 4. Equity curve with drawdown
// ---------------------------------------------------------------------------

/// Equity over time with the underwater curve shaded beneath.
///
/// A final P&L number hides the path entirely. Peak-to-trough is what actually
/// determines whether a strategy is runnable, and it is the number that decides
/// whether the drawdown limit ever fires.
pub fn equity_curve(equity: &[f64], label: &str) -> Svg {
    let f = Frame::default_frame();
    if equity.len() < 2 {
        return Svg(format!("{}</svg>\n", head("Equity curve", "no data")));
    }

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd: f64 = 0.0;
    for &e in equity {
        lo = lo.min(e);
        hi = hi.max(e);
        peak = peak.max(e);
        max_dd = max_dd.max(peak - e);
    }
    if !(hi > lo) {
        hi = lo + 1.0;
    }
    let span = hi - lo;

    let final_pnl = equity.last().copied().unwrap_or(0.0);
    let mut out = head(
        &format!("Equity — {label}"),
        &format!(
            "final {final_pnl:+.2}   peak-to-trough {max_dd:.2}   samples {}",
            equity.len()
        ),
    );

    let yt: Vec<(f64, String)> = (0..=4)
        .map(|i| {
            let t = i as f64 / 4.0;
            (t, format!("{:.0}", lo + t * span))
        })
        .collect();
    let xt = vec![
        (0.0, "start".to_string()),
        (0.5, "".to_string()),
        (1.0, "end".to_string()),
    ];
    out.push_str(&axes(&f, "session time", "equity", &xt, &yt));

    // Zero line, if it is in range.
    if lo < 0.0 && hi > 0.0 {
        let z = (0.0 - lo) / span;
        out.push_str(&format!(
            r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3"/>
"#,
            f.x0, f.sy(z), f.x1, f.sy(z)
        ));
    }

    // Downsample to roughly one point per horizontal pixel. Fifteen thousand
    // samples on a 640px canvas produced a 519 KB file that renders no
    // differently from a 7 KB one — and min/max within each bucket is kept, so
    // a one-tick spike survives the reduction rather than being sampled away.
    let target = (W as usize).max(2);
    let src: Vec<f64> = if equity.len() <= target * 2 {
        equity.to_vec()
    } else {
        let mut v = Vec::with_capacity(target * 2);
        for i in 0..target {
            let a = i * equity.len() / target;
            let b = (((i + 1) * equity.len()) / target)
                .max(a + 1)
                .min(equity.len());
            let slice = equity.get(a..b).unwrap_or(&[]);
            if slice.is_empty() {
                continue;
            }
            let mut lo_b = f64::INFINITY;
            let mut hi_b = f64::NEG_INFINITY;
            for &x in slice {
                lo_b = lo_b.min(x);
                hi_b = hi_b.max(x);
            }
            // Preserve traversal order within the bucket so the line does not
            // zig-zag: emit whichever extreme the bucket reached first.
            let first = slice.first().copied().unwrap_or(lo_b);
            let last = slice.last().copied().unwrap_or(hi_b);
            if (first - lo_b).abs() < (first - hi_b).abs() {
                v.push(lo_b);
                v.push(hi_b);
            } else {
                v.push(hi_b);
                v.push(lo_b);
            }
            let _ = last;
        }
        v
    };
    let equity: &[f64] = &src;

    let n = equity.len();
    let mut line = Vec::with_capacity(n);
    let mut under = Vec::with_capacity(n);
    let mut run_peak = f64::NEG_INFINITY;
    for (i, &e) in equity.iter().enumerate() {
        run_peak = run_peak.max(e);
        let tx = i as f64 / (n - 1) as f64;
        line.push(format!("{:.1},{:.1}", f.sx(tx), f.sy((e - lo) / span)));
        under.push(format!(
            "{:.1},{:.1}",
            f.sx(tx),
            f.sy((run_peak - lo) / span)
        ));
    }
    // Shade the region between the running peak and the equity: the drawdown.
    let mut poly = under.clone();
    poly.extend(line.iter().rev().cloned());
    out.push_str(&format!(
        r#"<polygon points="{}" fill="{BAD}" fill-opacity="0.12"/>
<polyline points="{}" fill="none" stroke="{ACCENT}" stroke-width="1.8"/>
<text x="{:.1}" y="{:.1}" font-size="10" fill="{BAD}">shaded = underwater from peak</text>
"#,
        poly.join(" "),
        line.join(" "),
        f.sx(0.02),
        f.y0 + 12.0
    ));

    out.push_str("</svg>\n");
    Svg(out)
}

// ---------------------------------------------------------------------------
// Terminal rendering
// ---------------------------------------------------------------------------

/// A sparkline, for when opening a file is too slow a feedback loop.
pub fn sparkline(values: &[f64], width: usize) -> String {
    const RAMP: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || width == 0 {
        return String::new();
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return String::new();
    }
    let span = if hi > lo { hi - lo } else { 1.0 };
    let mut out = String::with_capacity(width);
    for i in 0..width {
        // Average the bucket rather than sampling it: a spike between two
        // sample points should not vanish because of where the grid landed.
        let a = i * values.len() / width;
        let b = (((i + 1) * values.len()) / width)
            .max(a + 1)
            .min(values.len());
        let slice = values.get(a..b).unwrap_or(&[]);
        if slice.is_empty() {
            out.push(' ');
            continue;
        }
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let t = ((mean - lo) / span).clamp(0.0, 1.0);
        let k = ((t * (RAMP.len() - 1) as f64).round() as usize).min(RAMP.len() - 1);
        out.push(*RAMP.get(k).unwrap_or(&' '));
    }
    out
}

/// Reliability diagram as text, for terminal output.
pub fn reliability_ascii(s: &Scorecard) -> String {
    let mut out = String::new();
    out.push_str("  observed frequency vs forecast (| = perfect calibration)\n");
    for (k, b) in s.bins.iter().enumerate() {
        if b.n == 0 {
            continue;
        }
        let lo = k as f64 / px_score::BINS as f64;
        let width = 40usize;
        let said = (b.mean_forecast * width as f64) as usize;
        let was = (b.observed * width as f64) as usize;
        let mut row = vec![' '; width + 1];
        if let Some(c) = row.get_mut(said.min(width)) {
            *c = '|';
        }
        if let Some(c) = row.get_mut(was.min(width)) {
            *c = if said == was { '#' } else { 'o' };
        }
        out.push_str(&format!(
            "  {:.1} {:>5}  {}  said {:>5.1}%  was {:>5.1}%  {:+.3}\n",
            lo,
            b.n,
            row.iter().collect::<String>(),
            b.mean_forecast * 100.0,
            b.observed * 100.0,
            b.error()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use px_score::{Forecast, Resolved, Scorer};

    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn sample_card() -> Scorecard {
        let mut s = Scorer::new();
        let mut rng = Lcg(5);
        for i in 0..3000u64 {
            let p = rng.u();
            s.record(Resolved {
                forecast: Forecast {
                    t_s: 0.0,
                    model_p: p,
                    venue_p: 0.5,
                    horizon_s: 60.0,
                    // Each draw is an independent sample for this test's
                    // purposes (chart rendering, not clustering) — its own
                    // cluster keeps the scorecard statistically well-formed
                    // too, not just visually.
                    cluster_id: i,
                },
                outcome: rng.u() < p,
            });
        }
        s.score()
    }

    fn well_formed(svg: &str) {
        assert!(svg.starts_with("<svg"), "missing svg open");
        assert!(svg.trim_end().ends_with("</svg>"), "missing svg close");
        // Balanced enough to render: every opened element is closed or self-closing.
        assert_eq!(svg.matches("<svg").count(), svg.matches("</svg>").count());
        assert_eq!(svg.matches("<text").count(), svg.matches("</text>").count());
        assert!(!svg.contains("NaN"), "NaN leaked into coordinates");
        assert!(!svg.contains("inf"), "infinity leaked into coordinates");
    }

    #[test]
    fn reliability_renders_valid_svg() {
        let svg = reliability(&sample_card());
        well_formed(svg.as_str());
        assert!(svg.as_str().contains("perfect calibration"));
        assert!(svg.as_str().contains("<circle"));
    }

    #[test]
    fn variance_shape_renders_and_shows_both_curves() {
        let svg = variance_shape(60.0);
        well_formed(svg.as_str());
        assert!(svg.as_str().contains("TWAP, correct"));
        assert!(svg.as_str().contains("sqrt(t) model"));
        assert_eq!(svg.as_str().matches("<polyline").count(), 2);
    }

    #[test]
    fn fee_curve_renders_and_reports_the_gap() {
        let svg = fee_curve(0.07, 0.20, "Polymarket crypto");
        well_formed(svg.as_str());
        // 0.07 * 0.25 * 1.20 = 0.021 -> 2.10 cents.
        assert!(svg.as_str().contains("2.10c gap"), "gap label wrong");
    }

    #[test]
    fn equity_curve_renders_and_shades_drawdown() {
        let eq: Vec<f64> = (0..200)
            .map(|i| {
                let x = i as f64;
                x * 0.5
                    - if (40..90).contains(&i) {
                        (x - 40.0) * 2.0
                    } else {
                        0.0
                    }
            })
            .collect();
        let svg = equity_curve(&eq, "test");
        well_formed(svg.as_str());
        assert!(svg.as_str().contains("<polygon"));
        assert!(svg.as_str().contains("underwater"));
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_emit_nan() {
        well_formed(equity_curve(&[], "empty").as_str().trim_end_matches('\n'));
        well_formed(equity_curve(&[5.0, 5.0, 5.0], "flat").as_str());
        well_formed(equity_curve(&[0.0, 0.0], "zero").as_str());
        well_formed(reliability(&Scorecard::default()).as_str());
        well_formed(variance_shape(0.0).as_str());
        well_formed(fee_curve(0.0, 0.0, "no fee").as_str());
    }

    #[test]
    fn sparkline_spans_the_full_ramp() {
        let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let s = sparkline(&v, 20);
        assert_eq!(s.chars().count(), 20);
        assert!(s.starts_with('▁'));
        assert!(s.ends_with('█'));
    }

    #[test]
    fn sparkline_averages_buckets_rather_than_sampling() {
        // A spike between grid points must still register.
        let mut v = vec![0.0; 100];
        if let Some(x) = v.get_mut(51) {
            *x = 100.0;
        }
        let s = sparkline(&v, 10);
        assert!(s.chars().any(|c| c != '▁'), "spike vanished: {s}");
    }

    #[test]
    fn sparkline_handles_degenerate_input() {
        assert_eq!(sparkline(&[], 10), "");
        assert_eq!(sparkline(&[1.0, 2.0], 0), "");
        assert_eq!(sparkline(&[f64::NAN, f64::NAN], 5), "");
        assert_eq!(sparkline(&[3.0, 3.0, 3.0], 5).chars().count(), 5);
    }

    #[test]
    fn ascii_reliability_renders() {
        let text = reliability_ascii(&sample_card());
        assert!(text.contains("said"));
        assert!(text.lines().count() > 3);
    }

    #[test]
    fn svg_escapes_hostile_labels() {
        let svg = fee_curve(0.07, 0.2, "<script>alert(1)</script>");
        assert!(!svg.as_str().contains("<script>"));
        assert!(svg.as_str().contains("&lt;script&gt;"));
    }
}
