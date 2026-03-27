"""Backward-compatible wrapper. Use backends module directly for new code."""

from .backends import LamaBackend

# Keep Inpainter as alias for backward compatibility
Inpainter = LamaBackend
