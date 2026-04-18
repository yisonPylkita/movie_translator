"""Simple retry helper for transient network failures."""

import time
import urllib.error

from ..logging import logger

# Errors worth retrying (transient network issues)
_RETRYABLE = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
)


def with_retry(fn, *, retries: int = 1, delay: float = 2.0, label: str = ''):
    """Call fn(), retrying on transient network errors.

    Returns fn() result on success, re-raises on final failure.
    """
    last_exc = None
    for attempt in range(1 + retries):
        try:
            return fn()
        except _RETRYABLE as e:
            last_exc = e
            if attempt < retries:
                logger.debug(f'{label} attempt {attempt + 1} failed ({e}), retrying in {delay}s')
                time.sleep(delay)
            else:
                logger.debug(f'{label} all {1 + retries} attempts failed')
    assert last_exc is not None
    raise last_exc
