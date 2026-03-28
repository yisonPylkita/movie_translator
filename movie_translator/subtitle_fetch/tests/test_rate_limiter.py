import time

from movie_translator.subtitle_fetch.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_call_allowed_immediately(self):
        limiter = RateLimiter(min_interval=0.5)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_second_call_delayed(self):
        limiter = RateLimiter(min_interval=0.3)
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25  # allow small tolerance

    def test_update_from_headers_respects_remaining(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.update_from_headers(
            {
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Reset': '1',
            }
        )
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.8  # should sleep ~1s

    def test_backoff_on_429(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.record_429(retry_after=0.3)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25

    def test_no_delay_when_remaining_is_high(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.update_from_headers(
            {
                'X-RateLimit-Remaining': '40',
                'X-RateLimit-Reset': '60',
            }
        )
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
