from ..logging import logger

_pysubs2 = None


def get_pysubs2():
    global _pysubs2

    if _pysubs2 is not None:
        return _pysubs2

    try:
        import pysubs2

        _pysubs2 = pysubs2
        return _pysubs2
    except ImportError:
        logger.error('pysubs2 package not found. Install with: uv add pysubs2')
        return None
