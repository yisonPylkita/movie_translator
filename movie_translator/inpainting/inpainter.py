"""Backend-agnostic inpainter interface.

Provides a stable import path (Inpainter) that is independent of the concrete
backend implementation. This allows callers to use a single import regardless
of which backend is active, and makes it easy to swap backends (e.g. LAMA,
ProPainter, STTN) in the future without updating every call site.
"""

from .backends import LamaBackend

# Public alias — callers should import Inpainter from here rather than
# reaching into a specific backend module.
Inpainter = LamaBackend
