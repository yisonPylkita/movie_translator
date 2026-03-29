"""Static offset realignment for fetched subtitles.

Compares a fetched subtitle file against a reference track (typically
extracted from the video) and estimates a constant timing offset. If the
offset is significant, shifts all events in the fetched file to align
with the reference.

Uses cross-correlation of binary activity vectors to find the shift that
maximises overlap between the two subtitle tracks. This is robust to
different line splitting, different line counts, and offsets comparable
to the inter-line spacing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..logging import logger
from ..subtitles._pysubs2 import get_pysubs2
from .validator import extract_timestamps


def _build_binary_vector(
    timestamps: list[tuple[int, int]],
    duration_ms: int,
    bin_size_ms: int,
) -> np.ndarray:
    """Convert timestamps to a binary activity vector."""
    n_bins = math.ceil(duration_ms / bin_size_ms) if duration_ms > 0 else 0
    if n_bins == 0:
        return np.zeros(0, dtype=np.float64)

    vec = np.zeros(n_bins, dtype=np.float64)
    for start, end in timestamps:
        first_bin = max(0, start // bin_size_ms)
        last_bin = min(n_bins - 1, (end - 1) // bin_size_ms) if end > start else first_bin
        vec[first_bin : last_bin + 1] = 1.0
    return vec


def estimate_offset(
    ref_timestamps: list[tuple[int, int]],
    cand_timestamps: list[tuple[int, int]],
    bin_size_ms: int = 100,
    max_shift_ms: int = 15000,
) -> int | None:
    """Estimate static timing offset via cross-correlation.

    Builds binary activity vectors from both tracks and finds the shift
    that maximises their overlap. This works regardless of how lines are
    split and handles offsets of any size up to max_shift_ms.

    A positive result means the candidate is early (shift it later).
    A negative result means the candidate is late (shift it earlier).

    Args:
        ref_timestamps: Reference (start_ms, end_ms) pairs.
        cand_timestamps: Candidate (start_ms, end_ms) pairs.
        bin_size_ms: Resolution of the activity vectors.
        max_shift_ms: Maximum offset to search in each direction.

    Returns:
        Estimated offset in milliseconds, or None if inputs are empty.
    """
    if not ref_timestamps or not cand_timestamps:
        return None

    duration = max(
        max(e for _, e in ref_timestamps),
        max(e for _, e in cand_timestamps),
    )

    ref_vec = _build_binary_vector(ref_timestamps, duration, bin_size_ms)
    cand_vec = _build_binary_vector(cand_timestamps, duration, bin_size_ms)

    if len(ref_vec) == 0 or len(cand_vec) == 0:
        return None

    max_len = max(len(ref_vec), len(cand_vec))
    ref = np.zeros(max_len, dtype=np.float64)
    cand = np.zeros(max_len, dtype=np.float64)
    ref[: len(ref_vec)] = ref_vec
    cand[: len(cand_vec)] = cand_vec

    max_shift_bins = max_shift_ms // bin_size_ms
    effective_max = min(max_shift_bins, max_len - 1)

    best_score = -1.0
    best_shift = 0

    for shift in range(-effective_max, effective_max + 1):
        if shift >= 0:
            score = float(np.dot(ref[shift:], cand[: max_len - shift]))
        else:
            score = float(np.dot(ref[: max_len + shift], cand[-shift:]))
        if score > best_score:
            best_score = score
            best_shift = shift

    # Quality check: reject if the peak overlap is too low relative to
    # how much activity each track has. This catches spurious matches
    # when the true offset exceeds the search window.
    ref_energy = float(np.dot(ref, ref))
    cand_energy = float(np.dot(cand, cand))
    if ref_energy == 0 or cand_energy == 0:
        return None
    norm = math.sqrt(ref_energy * cand_energy)
    if best_score / norm < 0.4:
        return None

    return best_shift * bin_size_ms


def apply_offset(subtitle_path: Path, offset_ms: int) -> None:
    """Shift all events in a subtitle file by the given offset.

    Modifies the file in place.

    Args:
        subtitle_path: Path to the subtitle file.
        offset_ms: Offset in milliseconds (positive = shift later,
            negative = shift earlier).
    """
    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return

    subs = pysubs2.load(str(subtitle_path))
    subs.shift(ms=offset_ms)
    subs.save(str(subtitle_path))


# Offsets below this threshold are imperceptible and not worth correcting.
_MIN_OFFSET_MS = 150

# Offsets above this threshold likely indicate a misidentified subtitle
# rather than a simple release-cut difference.
_MAX_OFFSET_MS = 15000


def align_to_reference(
    subtitle_path: Path,
    reference_path: Path,
    min_offset_ms: int = _MIN_OFFSET_MS,
    max_offset_ms: int = _MAX_OFFSET_MS,
) -> int:
    """Estimate and apply a static timing offset to align a subtitle file.

    Compares the subtitle against a reference track, estimates the offset,
    and shifts the subtitle file in place if the offset is significant.

    Args:
        subtitle_path: Path to the subtitle file to realign.
        reference_path: Path to the reference subtitle file.
        min_offset_ms: Minimum absolute offset to apply (below this is noise).
        max_offset_ms: Maximum absolute offset to trust (above this is suspect).

    Returns:
        The applied offset in milliseconds (0 if no correction was needed).
    """
    ref_timestamps, _ = extract_timestamps(reference_path)
    cand_timestamps, _ = extract_timestamps(subtitle_path)

    offset = estimate_offset(ref_timestamps, cand_timestamps, max_shift_ms=max_offset_ms)

    if offset is None:
        logger.debug('Offset estimation failed: empty inputs')
        return 0

    abs_offset = abs(offset)

    if abs_offset < min_offset_ms:
        logger.debug('Offset %+dms below threshold (%dms), skipping', offset, min_offset_ms)
        return 0

    if abs_offset > max_offset_ms:
        logger.warning(
            'Offset %+dms exceeds safety limit (%dms), skipping realignment',
            offset,
            max_offset_ms,
        )
        return 0

    apply_offset(subtitle_path, offset)
    logger.info('Realigned subtitle by %+dms', offset)
    return offset
