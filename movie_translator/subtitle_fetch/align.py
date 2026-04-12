"""Piecewise subtitle realignment for fetched subtitles.

Compares a fetched subtitle file against a reference track (typically
extracted from the video) and estimates timing offsets. Handles the
common case where the candidate was timed to a video source with the
opening sequence (OP) removed, producing a different offset for pre-OP
and post-OP content.

Algorithm:
  1. Detect structural gaps (OP/ED) in the reference by finding large
     dialogue-free intervals.
  2. If no gap found, estimate a single global offset via cross-correlation.
  3. If an OP gap is found, estimate separate offsets for the pre-OP and
     post-OP segments. The post-OP search range accounts for the full
     OP duration being removed.
  4. Apply per-segment shifts to the subtitle file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..logging import logger
from ..subtitles._pysubs2 import get_pysubs2
from .validator import build_activity_vector, extract_timestamps


def estimate_offset(
    ref_timestamps: list[tuple[int, int]],
    cand_timestamps: list[tuple[int, int]],
    bin_size_ms: int = 100,
    max_shift_ms: int = 15000,
    min_quality: float = 0.4,
) -> int | None:
    """Estimate static timing offset via cross-correlation.

    Builds binary activity vectors from both tracks and finds the shift
    that maximises their overlap.

    A positive result means the candidate is early (shift it later).
    A negative result means the candidate is late (shift it earlier).

    Args:
        min_quality: Minimum normalized correlation to accept (0.0–1.0).
            Use lower values (e.g. 0.2) for segment-level estimation where
            fewer events are available.

    Returns:
        Estimated offset in milliseconds, or None if inputs are empty
        or the correlation quality is too low.
    """
    if not ref_timestamps or not cand_timestamps:
        return None

    duration = max(
        max(e for _, e in ref_timestamps),
        max(e for _, e in cand_timestamps),
    )

    ref_vec = build_activity_vector(ref_timestamps, duration, bin_size_ms)
    cand_vec = build_activity_vector(cand_timestamps, duration, bin_size_ms)

    if len(ref_vec) == 0 or len(cand_vec) == 0:
        return None

    max_len = max(len(ref_vec), len(cand_vec))
    ref = np.zeros(max_len, dtype=np.float64)
    cand = np.zeros(max_len, dtype=np.float64)
    ref[: len(ref_vec)] = ref_vec
    cand[: len(cand_vec)] = cand_vec

    max_shift_bins = max_shift_ms // bin_size_ms
    effective_max = min(max_shift_bins, max_len - 1)

    corr = np.correlate(ref, cand, mode='full')
    # Zero-lag is at index len(ref) - 1; extract the slice for our shift range.
    zero_lag = len(ref) - 1
    lo = zero_lag - effective_max
    hi = zero_lag + effective_max + 1  # exclusive
    corr_slice = corr[lo:hi]
    best_idx = int(np.argmax(corr_slice))
    best_shift = best_idx - effective_max
    best_score = float(corr_slice[best_idx])

    # Quality check: reject if the peak overlap is too low.
    ref_energy = float(np.dot(ref, ref))
    cand_energy = float(np.dot(cand, cand))
    if ref_energy == 0 or cand_energy == 0:
        return None
    norm = math.sqrt(ref_energy * cand_energy)
    if best_score / norm < min_quality:
        return None

    return best_shift * bin_size_ms


# ---------------------------------------------------------------------------
# OP gap detection
# ---------------------------------------------------------------------------

# Minimum dialogue-free interval to be considered an OP/ED gap.
_MIN_GAP_MS = 60000

# Only look for the OP gap in this time window of the reference track.
_OP_SEARCH_START_MS = 30000  # 0:30 — OP rarely starts before 30s
_OP_SEARCH_END_MS = 360000  # 6:00 — OP should be done by 6 min


def detect_op_gap(
    timestamps: list[tuple[int, int]],
    min_gap_ms: int = _MIN_GAP_MS,
    search_start_ms: int = _OP_SEARCH_START_MS,
    search_end_ms: int = _OP_SEARCH_END_MS,
) -> tuple[int, int] | None:
    """Find the opening-sequence gap in a subtitle track.

    Looks for the largest dialogue-free interval within the expected OP
    time window. The gap boundaries are the end of the last pre-OP event
    and the start of the first post-OP event.

    Returns:
        (gap_start_ms, gap_end_ms) or None if no qualifying gap found.
    """
    if not timestamps:
        return None

    # Sort by start time, use end times for gap measurement
    events = sorted(timestamps, key=lambda t: t[0])

    best_gap = None
    best_gap_size = 0

    for i in range(len(events) - 1):
        _, end_i = events[i]
        start_next, _ = events[i + 1]
        gap_size = start_next - end_i

        if gap_size < min_gap_ms:
            continue
        # The gap must start within the search window
        if not (search_start_ms <= end_i <= search_end_ms):
            continue
        if gap_size > best_gap_size:
            best_gap_size = gap_size
            best_gap = (end_i, start_next)

    return best_gap


# ---------------------------------------------------------------------------
# Piecewise offset estimation
# ---------------------------------------------------------------------------


def _apply_piecewise_offsets(
    subtitle_path: Path,
    boundary_ms: int,
    pre_offset_ms: int,
    post_offset_ms: int,
) -> None:
    """Shift events in a subtitle file with different offsets per segment.

    Events with start time < boundary_ms are shifted by pre_offset_ms.
    Events with start time >= boundary_ms are shifted by post_offset_ms.

    Modifies the file in place.
    """
    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return

    subs = pysubs2.load(str(subtitle_path))
    for event in subs:
        if event.start < boundary_ms:
            event.start += pre_offset_ms
            event.end += pre_offset_ms
        else:
            event.start += post_offset_ms
            event.end += post_offset_ms
    subs.save(str(subtitle_path))


def apply_offset(subtitle_path: Path, offset_ms: int) -> None:
    """Shift all events in a subtitle file by the given offset.

    Modifies the file in place.
    """
    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return

    subs = pysubs2.load(str(subtitle_path))
    subs.shift(ms=offset_ms)
    subs.save(str(subtitle_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MIN_OFFSET_MS = 150


def align_to_reference(
    subtitle_path: Path,
    reference_path: Path,
    min_offset_ms: int = _MIN_OFFSET_MS,
) -> int:
    """Align a subtitle file to a reference track, handling OP-removed sources.

    Detects if the reference has an opening-sequence gap. If so, estimates
    separate offsets for the pre-OP and post-OP segments and applies a
    piecewise shift. Otherwise, falls back to a single global offset.

    Args:
        subtitle_path: Path to the subtitle file to realign.
        reference_path: Path to the reference subtitle file.
        min_offset_ms: Minimum absolute offset to apply (below this is noise).

    Returns:
        The applied offset in milliseconds. For piecewise alignment,
        returns the post-OP offset (the dominant one). Returns 0 if no
        correction was needed.
    """
    ref_timestamps, _ = extract_timestamps(reference_path)
    cand_timestamps, _ = extract_timestamps(subtitle_path)

    if not ref_timestamps or not cand_timestamps:
        return 0

    op_gap = detect_op_gap(ref_timestamps)

    if op_gap is not None:
        return _align_piecewise(
            subtitle_path, ref_timestamps, cand_timestamps, op_gap, min_offset_ms
        )

    return _align_global(subtitle_path, ref_timestamps, cand_timestamps, min_offset_ms)


def _align_global(
    subtitle_path: Path,
    ref_timestamps: list[tuple[int, int]],
    cand_timestamps: list[tuple[int, int]],
    min_offset_ms: int,
) -> int:
    """Single global offset alignment (no OP gap detected)."""
    offset = estimate_offset(ref_timestamps, cand_timestamps, max_shift_ms=15000)

    if offset is None:
        logger.debug('Global offset estimation failed')
        return 0

    if abs(offset) < min_offset_ms:
        logger.debug('Global offset %+dms below threshold, skipping', offset)
        return 0

    apply_offset(subtitle_path, offset)
    logger.info('Realigned subtitle by %+dms (global)', offset)
    return offset


def _align_piecewise(
    subtitle_path: Path,
    ref_timestamps: list[tuple[int, int]],
    cand_timestamps: list[tuple[int, int]],
    op_gap: tuple[int, int],
    min_offset_ms: int,
) -> int:
    """Piecewise alignment for OP-removed subtitle sources."""
    gap_start, gap_end = op_gap
    op_duration = gap_end - gap_start

    logger.info(
        'Detected OP gap in reference: %d-%dms (%.1fs)',
        gap_start,
        gap_end,
        op_duration / 1000,
    )

    # Split reference into pre-OP and post-OP segments
    pre_op_ref = [(s, e) for s, e in ref_timestamps if e <= gap_start]
    post_op_ref = [(s, e) for s, e in ref_timestamps if s >= gap_end]

    # Estimate pre-OP offset (small search range — just a few seconds)
    pre_offset = estimate_offset(pre_op_ref, cand_timestamps, max_shift_ms=15000, min_quality=0.2)

    # Estimate post-OP offset (large search range — OP could be removed)
    post_offset = estimate_offset(
        post_op_ref, cand_timestamps, max_shift_ms=op_duration + 30000, min_quality=0.2
    )

    if pre_offset is None and post_offset is None:
        logger.debug('Piecewise offset estimation failed for both segments')
        return 0

    # If one segment failed, use the other's offset for both
    if pre_offset is None:
        pre_offset = post_offset
    if post_offset is None:
        post_offset = pre_offset

    # Both cannot be None here (early return above handles that case)
    assert pre_offset is not None
    assert post_offset is not None

    # Check if the offsets are actually different (piecewise needed)
    if abs(pre_offset - post_offset) < min_offset_ms:
        # Offsets are the same — just do a global shift
        offset = post_offset  # post-OP has more lines, more reliable
        if abs(offset) < min_offset_ms:
            return 0
        apply_offset(subtitle_path, offset)
        logger.info('Realigned subtitle by %+dms (uniform)', offset)
        return offset

    # Determine the boundary in the candidate timeline.
    # Pre-OP candidate events end at approximately: gap_start - pre_offset
    # Post-OP candidate events start at approximately: gap_end - post_offset
    # Use the midpoint as the boundary.
    pre_op_cand_end = gap_start - pre_offset
    post_op_cand_start = gap_end - post_offset
    boundary = (pre_op_cand_end + post_op_cand_start) // 2

    logger.info(
        'Piecewise alignment: pre-OP %+dms, post-OP %+dms, boundary at %dms',
        pre_offset,
        post_offset,
        boundary,
    )

    # Only apply if at least one offset is significant
    pre_significant = abs(pre_offset) >= min_offset_ms
    post_significant = abs(post_offset) >= min_offset_ms

    if not pre_significant and not post_significant:
        return 0

    effective_pre = pre_offset if pre_significant else 0
    effective_post = post_offset if post_significant else 0

    _apply_piecewise_offsets(subtitle_path, boundary, effective_pre, effective_post)
    logger.info(
        'Applied piecewise realignment: pre-OP %+dms, post-OP %+dms',
        effective_pre,
        effective_post,
    )
    return effective_post  # Return the dominant offset
