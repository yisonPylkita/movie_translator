"""Hardsub-OCR integration: download a hardsubbed stream and OCR its baked-in subs.

Graduated from the `scripts/hardsub_poc/` proof of concept. Exposes exactly two
entry points consumed by the Rust PyO3 bridge:

  * :func:`download_episode` — fetch the lowest OCR-legible copy of a player
    embed URL via yt-dlp.
  * :func:`ocr_and_clean` — OCR burned-in subtitles (Apple Vision) and
    post-process the per-frame results into a cleaned ``.srt``.
"""

from __future__ import annotations

from .download import HardsubError, download_episode
from .resolver import ocr_and_clean

__all__ = [
    'HardsubError',
    'download_episode',
    'ocr_and_clean',
]
