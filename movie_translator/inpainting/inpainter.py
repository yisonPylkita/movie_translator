"""Backend-agnostic inpainter interface.

Provides a stable import path (Inpainter) that is independent of the concrete
backend implementation. Uses OpenCV Telea by default (fast, no ML deps).
"""

from .backends import OpenCVTeleaBackend

# Public alias — callers should import Inpainter from here rather than
# reaching into a specific backend module.
Inpainter = OpenCVTeleaBackend
