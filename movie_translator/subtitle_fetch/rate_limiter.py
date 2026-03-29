"""Thread-safe rate limiter with HTTP header awareness.

Designed for APIs that return X-RateLimit-* headers (e.g., OpenSubtitles).
"""

import threading
import time


class RateLimiter:
    """Rate limiter that enforces minimum intervals and respects API rate limit headers."""

    def __init__(self, min_interval: float = 0.25):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request: float = 0.0
        self._blocked_until: float = 0.0

    def wait(self) -> None:
        """Block until it is safe to make the next request."""
        with self._lock:
            now = time.monotonic()

            # Respect 429 / header-based block
            if now < self._blocked_until:
                delay = self._blocked_until - now
                time.sleep(delay)

            # Respect minimum interval
            elapsed = time.monotonic() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request = time.monotonic()

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Parse X-RateLimit-* headers to adjust pacing."""
        remaining = headers.get('X-RateLimit-Remaining')
        reset = headers.get('X-RateLimit-Reset')

        if remaining is not None and reset is not None:
            try:
                remaining_int = int(remaining)
                reset_secs = float(reset)
            except ValueError, TypeError:
                return

            if remaining_int <= 1 and reset_secs > 0:
                with self._lock:
                    self._blocked_until = time.monotonic() + reset_secs

    def record_429(self, retry_after: float | None = None) -> None:
        """Record a 429 response. Back off for retry_after seconds (default: 5s)."""
        delay = retry_after if retry_after is not None else 5.0
        with self._lock:
            self._blocked_until = time.monotonic() + delay
