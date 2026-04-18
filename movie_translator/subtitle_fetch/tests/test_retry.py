import urllib.error

import pytest

from movie_translator.subtitle_fetch.retry import with_retry


class TestWithRetry:
    def test_returns_on_first_success(self):
        result = with_retry(lambda: 42, retries=2, delay=0, label='test')
        assert result == 42

    def test_retries_on_url_error(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError('timeout')
            return 'ok'

        result = with_retry(flaky, retries=1, delay=0, label='test')
        assert result == 'ok'
        assert len(calls) == 2

    def test_retries_on_timeout(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError('timed out')
            return 'ok'

        result = with_retry(flaky, retries=1, delay=0, label='test')
        assert result == 'ok'

    def test_raises_after_exhausted_retries(self):
        def always_fail():
            raise ConnectionError('refused')

        with pytest.raises(ConnectionError):
            with_retry(always_fail, retries=1, delay=0, label='test')

    def test_non_retryable_error_not_retried(self):
        calls = []

        def bad():
            calls.append(1)
            raise ValueError('bad input')

        with pytest.raises(ValueError):
            with_retry(bad, retries=2, delay=0, label='test')
        assert len(calls) == 1
