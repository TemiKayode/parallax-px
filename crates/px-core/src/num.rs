//! Fixed-point primitives for the hot path.
//!
//! Everything that touches an order book or an order message is an integer.
//! Floating point is confined to the alpha layer (`px-alpha`), where the
//! quantities are genuinely continuous and where we control the transcendental
//! implementations ourselves so that replay is bit-exact (see `crate::math`).
//!
//! Scale conventions:
//!   * `Px`   — micro-dollars per share. A 52c quote is `Px(520_000)`.
//!   * `Qty`  — micro-shares. 100 shares is `Qty(100_000_000)`.
//!   * `Usd`  — micro-dollars of notional.
//!   * `Prob` — parts per million. 54.3% is `Prob(543_000)`.
//!
//! Prediction-market prices live in `[0, 1]`, so `Px` never exceeds `ONE`.
//! That bound is what lets us use a dense array-indexed book (see `crate::book`).

pub const PPM: i64 = 1_000_000;

/// Price of one outcome share, in micro-dollars. Domain `[0, 1_000_000]`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct Px(pub i32);

/// Quantity of outcome shares, in micro-shares.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct Qty(pub i64);

/// Notional value, in micro-dollars. Signed: negative means we paid out.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct Usd(pub i64);

/// Probability in parts per million. Domain `[0, 1_000_000]`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct Prob(pub i32);

impl Px {
    pub const ZERO: Px = Px(0);
    pub const ONE: Px = Px(1_000_000);

    /// Clamp into the tradable domain. Polymarket rejects 0 and 1 outright, so
    /// we clamp to one tick inside on each end at the quoting layer instead.
    #[inline(always)]
    pub fn clamp_unit(self) -> Px {
        Px(self.0.clamp(0, 1_000_000))
    }

    /// Round *down* to the market's tick grid (used for bids).
    #[inline(always)]
    pub fn floor_to_tick(self, tick: i32) -> Px {
        debug_assert!(tick > 0);
        Px(self.0 - self.0.rem_euclid(tick))
    }

    /// Round *up* to the market's tick grid (used for asks).
    #[inline(always)]
    pub fn ceil_to_tick(self, tick: i32) -> Px {
        debug_assert!(tick > 0);
        let r = self.0.rem_euclid(tick);
        if r == 0 {
            self
        } else {
            Px(self.0 + (tick - r))
        }
    }

    /// The complementary outcome's price. Buying NO at 48c is economically
    /// identical to selling YES at 52c; the selector relies on this identity
    /// to choose the cheaper leg.
    #[inline(always)]
    pub fn complement(self) -> Px {
        Px(1_000_000 - self.0)
    }

    #[inline(always)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64 * 1e-6
    }

    #[inline(always)]
    pub fn from_f64(v: f64) -> Px {
        // `as` on f64 -> i32 saturates in Rust (since 1.45), so this cannot UB.
        Px(((v * 1e6) + 0.5).floor() as i32).clamp_unit()
    }
}

impl Qty {
    pub const ZERO: Qty = Qty(0);

    #[inline(always)]
    pub fn shares(n: i64) -> Qty {
        Qty(n * PPM)
    }

    #[inline(always)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64 * 1e-6
    }

    #[inline(always)]
    pub fn min(self, other: Qty) -> Qty {
        if self.0 < other.0 {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl Usd {
    pub const ZERO: Usd = Usd(0);

    #[inline(always)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64 * 1e-6
    }

    #[inline(always)]
    pub fn dollars(n: i64) -> Usd {
        Usd(n * PPM)
    }
}

impl Prob {
    pub const ZERO: Prob = Prob(0);
    pub const HALF: Prob = Prob(500_000);
    pub const ONE: Prob = Prob(1_000_000);

    #[inline(always)]
    pub fn from_f64(v: f64) -> Prob {
        Prob(((v * 1e6) + 0.5).floor() as i32).clamp_unit()
    }

    #[inline(always)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64 * 1e-6
    }

    #[inline(always)]
    pub fn clamp_unit(self) -> Prob {
        Prob(self.0.clamp(0, 1_000_000))
    }

    /// A probability is also a price: the risk-neutral value of a $1 binary.
    #[inline(always)]
    pub fn as_px(self) -> Px {
        Px(self.0)
    }

    #[inline(always)]
    pub fn complement(self) -> Prob {
        Prob(1_000_000 - self.0)
    }
}

/// `px * qty` -> notional. Both are 1e-6 scaled, so the raw product is 1e-12;
/// we do the multiply in i128 to avoid overflow on large sizes and then rescale.
#[inline(always)]
pub fn notional(px: Px, qty: Qty) -> Usd {
    let raw = (px.0 as i128) * (qty.0 as i128);
    Usd((raw / (PPM as i128)) as i64)
}

/// Signed difference between two prices, in micro-dollars. This is the unit the
/// edge calculator works in: "9 cents of edge" is `90_000`.
#[inline(always)]
pub fn spread(a: Px, b: Px) -> i32 {
    a.0 - b.0
}

impl core::ops::Add for Usd {
    type Output = Usd;
    #[inline(always)]
    fn add(self, rhs: Usd) -> Usd {
        Usd(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Usd {
    type Output = Usd;
    #[inline(always)]
    fn sub(self, rhs: Usd) -> Usd {
        Usd(self.0 - rhs.0)
    }
}

impl core::ops::Add for Qty {
    type Output = Qty;
    #[inline(always)]
    fn add(self, rhs: Qty) -> Qty {
        Qty(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Qty {
    type Output = Qty;
    #[inline(always)]
    fn sub(self, rhs: Qty) -> Qty {
        Qty(self.0 - rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_rounding_is_directional() {
        // Bids round down, asks round up: we never accidentally cross by rounding.
        assert_eq!(Px(523_400).floor_to_tick(10_000), Px(520_000));
        assert_eq!(Px(523_400).ceil_to_tick(10_000), Px(530_000));
        assert_eq!(Px(520_000).ceil_to_tick(10_000), Px(520_000));
    }

    #[test]
    fn complement_is_involutive() {
        assert_eq!(Px(480_000).complement(), Px(520_000));
        assert_eq!(Px(480_000).complement().complement(), Px(480_000));
    }

    #[test]
    fn notional_is_exact_for_round_numbers() {
        // 100 shares at 52c = $52.00
        assert_eq!(notional(Px(520_000), Qty::shares(100)), Usd::dollars(52));
    }

    #[test]
    fn notional_does_not_overflow_at_scale() {
        // 10 million shares at 99c: raw i64 product would be 9.9e18, near i64::MAX.
        let n = notional(Px(990_000), Qty::shares(10_000_000));
        assert_eq!(n, Usd::dollars(9_900_000));
    }
}
