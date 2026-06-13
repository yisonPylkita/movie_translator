//! Thread-safe rate limiter with HTTP header awareness.
//!
//! Designed for APIs that return `X-RateLimit-*` headers (e.g., OpenSubtitles).
//! The clock is injected via the `Clock` trait so tests are deterministic.

use std::sync::Mutex;

/// Clock abstraction for testability — callers provide `now` and `sleep`.
pub trait Clock: Send + Sync {
    /// Monotonic "now" in seconds.
    fn now(&self) -> f64;
    /// Block for `secs` seconds.
    fn sleep(&self, secs: f64);
}

/// Real clock backed by `std::time::Instant`.
pub struct RealClock {
    start: std::time::Instant,
}

impl Default for RealClock {
    fn default() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
}

impl Clock for RealClock {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn sleep(&self, secs: f64) {
        if secs > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
    }
}

struct RateLimiterState {
    last_request: f64,
    blocked_until: f64,
}

/// Rate limiter that enforces minimum intervals and respects API rate limit
/// headers. Uses `Clock` injection for deterministic tests.
pub struct RateLimiter<C: Clock = RealClock> {
    min_interval: f64,
    state: Mutex<RateLimiterState>,
    clock: C,
}

impl RateLimiter<RealClock> {
    /// Create a real-clock rate limiter with the given minimum interval in seconds.
    pub fn new(min_interval: f64) -> Self {
        Self::with_clock(min_interval, RealClock::default())
    }
}

impl<C: Clock> RateLimiter<C> {
    /// Create a rate limiter with a custom clock (for testing).
    pub fn with_clock(min_interval: f64, clock: C) -> Self {
        Self {
            min_interval,
            state: Mutex::new(RateLimiterState {
                last_request: 0.0,
                blocked_until: 0.0,
            }),
            clock,
        }
    }

    /// Block until it is safe to make the next request.
    pub fn wait(&self) {
        let mut state = self.state.lock().expect("rate limiter poisoned");

        let now = self.clock.now();

        // Respect 429 / header-based block
        if now < state.blocked_until {
            let delay = state.blocked_until - now;
            drop(state); // release lock before sleeping
            self.clock.sleep(delay);
            state = self.state.lock().expect("rate limiter poisoned");
        }

        // Respect minimum interval
        let elapsed = self.clock.now() - state.last_request;
        if elapsed < self.min_interval {
            let sleep_for = self.min_interval - elapsed;
            drop(state);
            self.clock.sleep(sleep_for);
            state = self.state.lock().expect("rate limiter poisoned");
        }

        state.last_request = self.clock.now();
    }

    /// Parse `X-RateLimit-*` headers to adjust pacing.
    pub fn update_from_headers(&self, headers: &std::collections::HashMap<String, String>) {
        let remaining = match headers.get("X-RateLimit-Remaining") {
            Some(v) => v,
            None => return,
        };
        let reset = match headers.get("X-RateLimit-Reset") {
            Some(v) => v,
            None => return,
        };

        let remaining_int: i64 = match remaining.parse() {
            Ok(v) => v,
            Err(_) => return,
        };
        let reset_secs: f64 = match reset.parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        if remaining_int <= 1 && reset_secs > 0.0 {
            let mut state = self.state.lock().expect("rate limiter poisoned");
            state.blocked_until = self.clock.now() + reset_secs;
        }
    }

    /// Record a 429 response. Back off for `retry_after` seconds (default: 5s).
    pub fn record_429(&self, retry_after: Option<f64>) {
        let delay = retry_after.unwrap_or(5.0);
        let mut state = self.state.lock().expect("rate limiter poisoned");
        state.blocked_until = self.clock.now() + delay;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A fake monotonic clock: `now` is advanced by explicit `advance` calls;
    /// `sleep` advances `now` by the requested duration (deterministic).
    struct FakeClock {
        // Stored as f64 bits in an AtomicU64 for interior mutability.
        now_bits: AtomicU64,
        total_slept: std::sync::Mutex<f64>,
    }

    impl FakeClock {
        fn new(start: f64) -> Self {
            Self {
                now_bits: AtomicU64::new(start.to_bits()),
                total_slept: std::sync::Mutex::new(0.0),
            }
        }

        fn advance(&self, delta: f64) {
            let old = f64::from_bits(self.now_bits.load(Ordering::SeqCst));
            self.now_bits
                .store((old + delta).to_bits(), Ordering::SeqCst);
        }

        fn total_slept(&self) -> f64 {
            *self.total_slept.lock().unwrap()
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> f64 {
            f64::from_bits(self.now_bits.load(Ordering::SeqCst))
        }

        fn sleep(&self, secs: f64) {
            if secs > 0.0 {
                *self.total_slept.lock().unwrap() += secs;
                self.advance(secs);
            }
        }
    }

    fn fake_limiter(min_interval: f64, start: f64) -> RateLimiter<FakeClock> {
        RateLimiter::with_clock(min_interval, FakeClock::new(start))
    }

    #[test]
    fn first_call_no_sleep() {
        let limiter = fake_limiter(0.5, 100.0);
        limiter.wait();
        assert_eq!(limiter.clock.total_slept(), 0.0);
    }

    #[test]
    fn second_call_delayed_by_min_interval() {
        let limiter = fake_limiter(0.3, 100.0);
        limiter.wait(); // first call — no sleep
        // No real time passes (fake clock only moves via sleep)
        limiter.wait(); // second call — should sleep 0.3s
        assert!((limiter.clock.total_slept() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn update_from_headers_remaining_zero_blocks() {
        let limiter = fake_limiter(0.0, 100.0);
        let mut headers = HashMap::new();
        headers.insert("X-RateLimit-Remaining".to_string(), "0".to_string());
        headers.insert("X-RateLimit-Reset".to_string(), "1".to_string());
        limiter.update_from_headers(&headers);
        limiter.wait();
        // Should have slept ~1 second (the reset value)
        assert!(limiter.clock.total_slept() >= 0.9);
    }

    #[test]
    fn record_429_blocks_for_retry_after() {
        let limiter = fake_limiter(0.0, 100.0);
        limiter.record_429(Some(0.3));
        limiter.wait();
        assert!(limiter.clock.total_slept() >= 0.29);
    }

    #[test]
    fn update_from_headers_high_remaining_no_block() {
        let limiter = fake_limiter(0.0, 100.0);
        let mut headers = HashMap::new();
        headers.insert("X-RateLimit-Remaining".to_string(), "40".to_string());
        headers.insert("X-RateLimit-Reset".to_string(), "60".to_string());
        limiter.update_from_headers(&headers);
        limiter.wait();
        assert_eq!(limiter.clock.total_slept(), 0.0);
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    #[test]
    fn record_429_default_delay_is_5s() {
        let limiter = fake_limiter(0.0, 100.0);
        limiter.record_429(None);
        limiter.wait();
        assert!((limiter.clock.total_slept() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn update_from_headers_exactly_one_remaining_blocks() {
        // remaining == 1 → should block (remaining <= 1 condition)
        let limiter = fake_limiter(0.0, 100.0);
        let mut headers = HashMap::new();
        headers.insert("X-RateLimit-Remaining".to_string(), "1".to_string());
        headers.insert("X-RateLimit-Reset".to_string(), "2".to_string());
        limiter.update_from_headers(&headers);
        limiter.wait();
        assert!(limiter.clock.total_slept() >= 1.9);
    }

    #[test]
    fn update_from_headers_missing_header_is_noop() {
        let limiter = fake_limiter(0.0, 100.0);
        let headers = HashMap::new(); // empty
        limiter.update_from_headers(&headers);
        limiter.wait();
        assert_eq!(limiter.clock.total_slept(), 0.0);
    }

    #[test]
    fn update_from_headers_bad_value_is_noop() {
        let limiter = fake_limiter(0.0, 100.0);
        let mut headers = HashMap::new();
        headers.insert(
            "X-RateLimit-Remaining".to_string(),
            "not-a-number".to_string(),
        );
        headers.insert("X-RateLimit-Reset".to_string(), "60".to_string());
        limiter.update_from_headers(&headers);
        limiter.wait();
        assert_eq!(limiter.clock.total_slept(), 0.0);
    }
}
