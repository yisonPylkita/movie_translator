from functools import lru_cache

from ..logging import logger


@lru_cache(maxsize=1)
def get_pysubs2():
    try:
        import pysubs2

        return pysubs2
    except ImportError:
        logger.error('pysubs2 package not found. Install with: uv add pysubs2')
        return None
