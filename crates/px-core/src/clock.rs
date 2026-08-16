//! Time.
//!
//! The engine never reads the clock itself. Every decision function takes `now`
//! as an explicit parameter. This is not stylistic fussiness: it is the single
//! change that makes the replay harness able to reproduce a production decision
//! exactly, including the time-decay term of the fair-value model.

/// Nanoseconds on the local monotonic timebase (TSC-derived in production).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct Nanos(pub u64);

impl Nanos {
    pub const ZERO: Nanos = Nanos(0);

    #[inline(always)]
    pub fn from_millis(ms: u64) -> Nanos {
        // Saturating, not wrapping: an absurd `ms` (a config typo, a
        // corrupted duration) must clamp to "very far in the future"
        // rather than wrap into "very soon" — the same reasoning `since`
        // below already applies to subtraction.
        Nanos(ms.saturating_mul(1_000_000))
    }

    #[inline(always)]
    pub fn from_secs_f64(s: f64) -> Nanos {
        Nanos((s * 1e9) as u64)
    }

    #[inline(always)]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 * 1e-9
    }

    #[inline(always)]
    pub fn as_millis(self) -> u64 {
        self.0 / 1_000_000
    }

    /// Saturating difference — a clock that appears to run backwards (NTP step,
    /// core migration on a system without invariant TSC) yields zero rather
    /// than a wrapped enormous value that would poison the decay term.
    #[inline(always)]
    pub fn since(self, earlier: Nanos) -> Nanos {
        Nanos(self.0.saturating_sub(earlier.0))
    }
}

impl core::ops::Add for Nanos {
    type Output = Nanos;
    #[inline(always)]
    fn add(self, rhs: Nanos) -> Nanos {
        // Saturating for the same reason `since` is: a wrapped timestamp
        // silently becomes a *small* one, which is the one failure mode a
        // monotonic clock must never produce.
        Nanos(self.0.saturating_add(rhs.0))
    }
}

/// Source of monotonic time. `RealClock` in production, `ReplayClock` under the
/// harness so that a recorded session advances time by recorded amounts.
pub trait Clock {
    fn now(&self) -> Nanos;
}

#[derive(Debug)]
pub struct RealClock {
    origin: std::time::Instant,
}

impl Default for RealClock {
    fn default() -> Self {
        RealClock {
            origin: std::time::Instant::now(),
        }
    }
}

impl Clock for RealClock {
    #[inline(always)]
    fn now(&self) -> Nanos {
        Nanos(self.origin.elapsed().as_nanos() as u64)
    }
}

/// Deterministic clock driven by the replay harness.
#[derive(Default, Debug)]
pub struct ReplayClock {
    t: std::cell::Cell<u64>,
}

impl ReplayClock {
    pub fn set(&self, t: Nanos) {
        self.t.set(t.0);
    }
    pub fn advance(&self, d: Nanos) {
        self.t.set(self.t.get().saturating_add(d.0));
    }
}

impl Clock for ReplayClock {
    #[inline(always)]
    fn now(&self) -> Nanos {
        Nanos(self.t.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_saturates_on_backwards_clock() {
        assert_eq!(Nanos(100).since(Nanos(500)), Nanos(0));
        assert_eq!(Nanos(500).since(Nanos(100)), Nanos(400));
    }

    #[test]
    fn replay_clock_is_deterministic() {
        let c = ReplayClock::default();
        c.set(Nanos::from_millis(1000));
        assert_eq!(c.now(), Nanos::from_millis(1000));
        c.advance(Nanos::from_millis(250));
        assert_eq!(c.now(), Nanos::from_millis(1250));
        assert_eq!(c.now(), Nanos::from_millis(1250)); // reading does not advance
    }
}
