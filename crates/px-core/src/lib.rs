//! `px-core` — shared vocabulary for the Parallax engine.
//!
//! Contains only things that every other crate needs: fixed-point numeric
//! types, deterministic math, the dense order book, time, and the market
//! metadata that describes *what* we are pricing.
//!
//! This crate has no dependencies, allocates nothing after construction, and
//! compiles clean under `#![deny(warnings)]` in CI.

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod book;
pub mod clock;
pub mod math;
pub mod num;

pub use book::{DenseBook, Side, Walk};
pub use clock::{Clock, Nanos, RealClock, ReplayClock};
pub use num::{notional, Prob, Px, Qty, Usd};

/// Opaque venue identifier for one binary outcome token (Polymarket calls this
/// an `asset_id`). Interned to a `u32` at startup so the hot path compares
/// integers instead of 78-digit decimal strings.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct TokenId(pub u32);

/// Identifier for a market (a YES/NO pair). Polymarket's `condition_id`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct MarketId(pub u32);

/// Reference asset a market is written on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Underlying {
    Btc,
    Eth,
    Sol,
    Xrp,
    Doge,
    Bnb,
    Hype,
}

impl Underlying {
    pub const ALL: [Underlying; 7] = [
        Underlying::Btc,
        Underlying::Eth,
        Underlying::Sol,
        Underlying::Xrp,
        Underlying::Doge,
        Underlying::Bnb,
        Underlying::Hype,
    ];

    #[inline(always)]
    pub fn index(self) -> usize {
        match self {
            Underlying::Btc => 0,
            Underlying::Eth => 1,
            Underlying::Sol => 2,
            Underlying::Xrp => 3,
            Underlying::Doge => 4,
            Underlying::Bnb => 5,
            Underlying::Hype => 6,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Underlying::Btc => "BTC",
            Underlying::Eth => "ETH",
            Underlying::Sol => "SOL",
            Underlying::Xrp => "XRP",
            Underlying::Doge => "DOGE",
            Underlying::Bnb => "BNB",
            Underlying::Hype => "HYPE",
        }
    }
}

/// Venue fee category. The taker fee rate is a property of the category, and it
/// dominates the economics: on a crypto market at 50c the taker pays 1.75c per
/// share, which is larger than most of the mispricings worth chasing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Crypto,
    Sports,
    Finance,
    Politics,
    Economics,
    Culture,
    Weather,
    Tech,
    Mentions,
    Geopolitics,
    Other,
}

impl Category {
    /// `feeRate` in the venue's `fee = C * feeRate * p * (1 - p)` formula.
    #[inline(always)]
    pub fn taker_fee_rate(self) -> f64 {
        match self {
            Category::Crypto => 0.07,
            Category::Sports => 0.05,
            Category::Economics => 0.05,
            Category::Culture => 0.05,
            Category::Weather => 0.05,
            Category::Other => 0.05,
            Category::Finance => 0.04,
            Category::Politics => 0.04,
            Category::Tech => 0.04,
            Category::Mentions => 0.04,
            Category::Geopolitics => 0.0,
        }
    }

    /// Fraction of the counterparty's fee rebated to the resting maker.
    #[inline(always)]
    pub fn maker_rebate(self) -> f64 {
        match self {
            Category::Crypto => 0.20,
            Category::Sports => 0.15,
            Category::Geopolitics => 0.0,
            _ => 0.25,
        }
    }
}

/// How a market determines its settlement value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Settlement {
    /// Settles on a time-weighted average price over the final `window_s`
    /// seconds. This is how Polymarket's crypto 5m / 15m / 4h markets resolve,
    /// and modelling it correctly is the largest single source of edge in the
    /// system — see `px_alpha::twap`.
    Twap { window_s: f64 },
    /// Settles on the spot print at expiry.
    Spot,
}

/// Everything static about a market. Loaded once at subscription time, then
/// read-only on the hot path.
#[derive(Clone, Copy, Debug)]
pub struct MarketSpec {
    pub market: MarketId,
    pub yes: TokenId,
    pub no: TokenId,
    pub underlying: Underlying,
    pub category: Category,
    pub settlement: Settlement,
    /// Strike in the underlying's own units (e.g. USD per BTC).
    pub strike: f64,
    /// Expiry on the local monotonic timebase.
    pub expiry: Nanos,
    /// Minimum price increment, micro-dollars.
    pub tick: i32,
    /// Venue minimum order size.
    pub min_size: Qty,
    /// Maximum distance from mid, in ticks, at which a resting order still
    /// scores liquidity rewards. Orders outside this earn nothing, which makes
    /// it a hard boundary for the quoting engine rather than a soft preference.
    pub reward_max_spread_ticks: i32,
    /// Minimum resting size that qualifies for rewards.
    pub reward_min_size: Qty,
}

impl MarketSpec {
    /// Seconds remaining until expiry. Clamped at zero.
    #[inline(always)]
    pub fn tau_secs(&self, now: Nanos) -> f64 {
        self.expiry.since(now).as_secs_f64()
    }

    #[inline(always)]
    pub fn is_expired(&self, now: Nanos) -> bool {
        now.0 >= self.expiry.0
    }

    /// Length of the settlement averaging window in seconds (0 for spot).
    #[inline(always)]
    pub fn twap_window(&self) -> f64 {
        match self.settlement {
            Settlement::Twap { window_s } => window_s,
            Settlement::Spot => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crypto_taker_fee_matches_published_table() {
        // Venue table: 100 shares of a crypto market at 50c costs $1.75.
        let r = Category::Crypto.taker_fee_rate();
        let fee_per_share = r * 0.5 * 0.5;
        assert!((fee_per_share * 100.0 - 1.75).abs() < 1e-9);

        // ...and at 90c it costs $0.63.
        let fee_90 = r * 0.9 * 0.1 * 100.0;
        assert!((fee_90 - 0.63).abs() < 1e-9);

        // Politics at 50c: $1.00 per 100 shares.
        let fee_pol = Category::Politics.taker_fee_rate() * 0.25 * 100.0;
        assert!((fee_pol - 1.00).abs() < 1e-9);
    }

    #[test]
    fn fee_is_symmetric_about_fifty_cents() {
        let r = Category::Crypto.taker_fee_rate();
        for p in [0.05, 0.2, 0.31, 0.45] {
            let lo = r * p * (1.0 - p);
            let hi = r * (1.0 - p) * p;
            assert!((lo - hi).abs() < 1e-15);
        }
    }

    #[test]
    fn geopolitics_is_free() {
        assert_eq!(
            Category::Geopolitics.taker_fee_rate().to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    fn tau_clamps_at_expiry() {
        let spec = MarketSpec {
            market: MarketId(1),
            yes: TokenId(1),
            no: TokenId(2),
            underlying: Underlying::Btc,
            category: Category::Crypto,
            settlement: Settlement::Twap { window_s: 60.0 },
            strike: 65_000.0,
            expiry: Nanos::from_millis(300_000),
            tick: 10_000,
            min_size: Qty::shares(5),
            reward_max_spread_ticks: 3,
            reward_min_size: Qty::shares(50),
        };
        assert!((spec.tau_secs(Nanos::from_millis(240_000)) - 60.0).abs() < 1e-9);
        assert_eq!(
            spec.tau_secs(Nanos::from_millis(400_000)).to_bits(),
            0.0f64.to_bits()
        );
        assert!(spec.is_expired(Nanos::from_millis(400_000)));
    }
}
