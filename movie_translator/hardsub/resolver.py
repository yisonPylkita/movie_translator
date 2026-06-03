"""OCR a hardsubbed video and clean the result into a usable .srt.

Bridges the shared OCR stage (:func:`movie_translator.ocr.extract_burned_in_subtitles`,
Apple Vision, bottom 25%, 3fps) to the hardsub post-processor: the raw
per-frame OCR results are fuzzy-merged (jitter) and garbage-filtered into clean
dialogue lines, then written as a cleaned ``.srt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from movie_translator.ocr import extract_burned_in_subtitles

from .download import HardsubError
from .postprocess import merge_ocr_results, to_srt

logger = logging.getLogger(__name__)


def ocr_and_clean(video_path: str, out_dir: str, language: str = 'pl') -> str:
    """OCR burned-in subtitles from `video_path` and clean them into a .srt.

    Runs :func:`extract_burned_in_subtitles` (Apple Vision, bottom 25%, 3fps),
    then post-processes the raw per-frame OCR results — fuzzy-merging jitter and
    dropping garbage — into a cleaned ``.srt`` written under `out_dir`.

    Returns the cleaned ``.srt`` path, or ``''`` if nothing usable was produced.
    Raises :class:`HardsubError` on hard failure.
    """
    video = Path(video_path)
    out = Path(out_dir)

    try:
        result = extract_burned_in_subtitles(video, out, language=language)
    except Exception as exc:  # noqa: BLE001 - normalize any OCR failure
        raise HardsubError(f'Burned-in OCR failed for {video}: {exc}') from exc

    if result is None or not result.ocr_results:
        logger.info('No burned-in OCR results for %s', video)
        return ''

    frame_texts = [(r.timestamp_ms, r.text) for r in result.ocr_results]
    clean = merge_ocr_results(frame_texts)
    if not clean:
        logger.info('No usable dialogue after cleanup for %s', video)
        return ''

    out.mkdir(parents=True, exist_ok=True)
    srt_path = out / f'{video.stem}.pl.cleaned.srt'
    srt_path.write_text(to_srt(clean), encoding='utf-8')
    logger.info('Wrote cleaned hardsub srt: %s (%d lines)', srt_path, len(clean))
    return str(srt_path)
