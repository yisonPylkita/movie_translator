//! Retry helper for transient network failures.
//!
//! Inject a sleep function for deterministic testing.

use std::io::{Error, ErrorKind};
use tracing::debug;

/// Error kinds that are worth retrying (transient network issues): network
/// errors and the `io::Error` kinds corresponding to connection failures.
fn is_retryable(e: &FetchError) -> bool {
    match e {
        FetchError::Network(_) => true,
        FetchError::Io(io_err) => matches!(
            io_err.kind(),
            ErrorKind::ConnectionRefused
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::TimedOut
                | ErrorKind::NotConnected
                | ErrorKind::BrokenPipe
                | ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

/// Errors that can be retried.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("network error: {0}")]
    Network(String),
    #[error("I/O error: {0}")]
    Io(#[from] Error),
    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },
    #[error("parse error: {0}")]
    Parse(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("auth error: {0}")]
    Auth(String),
    #[error("quota exceeded")]
    QuotaExceeded,
}

/// Call `fn()`, retrying on transient network errors.
///
/// Returns the result on success, re-raises on final failure.
/// `sleep_fn` is called between retries (inject a no-op in tests).
pub fn with_retry<T, F, S>(
    mut f: F,
    retries: usize,
    delay_secs: f64,
    label: &str,
    sleep_fn: S,
) -> Result<T, FetchError>
where
    F: FnMut() -> Result<T, FetchError>,
    S: Fn(f64),
{
    let mut last_err = None;
    let total = 1 + retries;
    for attempt in 0..total {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) if is_retryable(&e) => {
                last_err = Some(e);
                if attempt < retries {
                    debug!(
                        "{label} attempt {} failed, retrying in {}s",
                        attempt + 1,
                        delay_secs
                    );
                    sleep_fn(delay_secs);
                } else {
                    debug!("{label} all {total} attempts failed");
                }
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    // Error is imported at module level

    fn no_sleep(_: f64) {}

    #[test]
    fn returns_on_first_success() {
        let result = with_retry(|| Ok::<i32, FetchError>(42), 2, 0.0, "test", no_sleep);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn retries_on_network_error() {
        let calls = Cell::new(0usize);
        let result = with_retry(
            || {
                let n = calls.get();
                calls.set(n + 1);
                if n == 0 {
                    Err(FetchError::Network("timeout".to_string()))
                } else {
                    Ok("ok")
                }
            },
            1,
            0.0,
            "test",
            no_sleep,
        );
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn retries_on_timed_out_io_error() {
        let calls = Cell::new(0usize);
        let result = with_retry(
            || {
                let n = calls.get();
                calls.set(n + 1);
                if n == 0 {
                    Err(FetchError::Io(Error::new(ErrorKind::TimedOut, "timed out")))
                } else {
                    Ok("ok")
                }
            },
            1,
            0.0,
            "test",
            no_sleep,
        );
        assert_eq!(result.unwrap(), "ok");
    }

    #[test]
    fn raises_after_all_retries_exhausted() {
        let result = with_retry(
            || Err::<(), _>(FetchError::Network("refused".to_string())),
            1,
            0.0,
            "test",
            no_sleep,
        );
        assert!(matches!(result, Err(FetchError::Network(_))));
    }

    #[test]
    fn non_retryable_error_not_retried() {
        let calls = Cell::new(0usize);
        let result = with_retry(
            || {
                calls.set(calls.get() + 1);
                Err::<(), _>(FetchError::Parse("bad input".to_string()))
            },
            2,
            0.0,
            "test",
            no_sleep,
        );
        assert!(matches!(result, Err(FetchError::Parse(_))));
        assert_eq!(calls.get(), 1); // only called once
    }

    // ── Additional: sleep_fn is called between retries ────────────────────────

    #[test]
    fn sleep_fn_called_between_retries() {
        let sleep_count = Cell::new(0usize);
        let calls = Cell::new(0usize);
        let _ = with_retry(
            || {
                calls.set(calls.get() + 1);
                if calls.get() < 3 {
                    Err(FetchError::Network("x".to_string()))
                } else {
                    Ok(())
                }
            },
            2,
            0.0,
            "test",
            |_| sleep_count.set(sleep_count.get() + 1),
        );
        // 3 attempts → 2 sleeps (between 1→2 and 2→3)
        assert_eq!(sleep_count.get(), 2);
    }

    // ── Additional: http error is not retried ─────────────────────────────────

    #[test]
    fn http_error_not_retried() {
        let calls = Cell::new(0usize);
        let _ = with_retry(
            || {
                calls.set(calls.get() + 1);
                Err::<(), _>(FetchError::Http {
                    status: 404,
                    body: "not found".to_string(),
                })
            },
            3,
            0.0,
            "test",
            no_sleep,
        );
        assert_eq!(calls.get(), 1);
    }
}
