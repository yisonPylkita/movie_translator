"""Subtitle validation via timing pattern fingerprinting.

Compares downloaded subtitle candidates against a reference track by
converting timing information into binary activity vectors and measuring
their cross-correlation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..logging import logger
from ..subtitles._pysubs2 import get_pysubs2
from ..types import NON_DIALOGUE_STYLES
from .types import SubtitleMatch


def build_activity_vector(
    timestamps: list[tuple[int, int]],
    duration_ms: int,
    bin_size_ms: int = 2000,
) -> np.ndarray:
    """Convert subtitle timestamps to a binary activity vector.

    Divides the timeline into fixed-width bins and marks each bin as 1
    if any dialogue event overlaps it, 0 otherwise.

    Args:
        timestamps: List of (start_ms, end_ms) pairs.
        duration_ms: Total duration of the subtitle track in milliseconds.
        bin_size_ms: Width of each bin in milliseconds.

    Returns:
        1-D numpy array of 0s and 1s.
    """
    n_bins = math.ceil(duration_ms / bin_size_ms) if duration_ms > 0 else 0
    if n_bins == 0:
        return np.zeros(0, dtype=np.float64)

    vec = np.zeros(n_bins, dtype=np.float64)
    for start, end in timestamps:
        first_bin = max(0, start // bin_size_ms)
        # Last bin that the event touches (inclusive).
        # An event ending exactly on a bin boundary doesn't spill into the next bin.
        last_bin = min(n_bins - 1, (end - 1) // bin_size_ms) if end > start else first_bin
        vec[first_bin : last_bin + 1] = 1.0

    return vec


def compute_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_shift_bins: int = 15,
) -> float:
    """Compute normalized cross-correlation between two activity vectors.

    Tries shifts from -max_shift_bins to +max_shift_bins and returns the
    peak correlation, normalized by the geometric mean of energies.

    Args:
        reference: Binary activity vector for the reference track.
        candidate: Binary activity vector for the candidate track.
        max_shift_bins: Maximum number of bins to shift in each direction.

    Returns:
        Peak correlation score between 0.0 and 1.0.
    """
    ref_energy = float(np.dot(reference, reference))
    cand_energy = float(np.dot(candidate, candidate))

    if ref_energy == 0.0 or cand_energy == 0.0:
        return 0.0

    norm = math.sqrt(ref_energy * cand_energy)

    # Pad both vectors to the same length for easier shifting.
    max_len = max(len(reference), len(candidate))
    ref = np.zeros(max_len, dtype=np.float64)
    cand = np.zeros(max_len, dtype=np.float64)
    ref[: len(reference)] = reference
    cand[: len(candidate)] = candidate

    best = 0.0
    # Clamp shift range so we don't exceed vector length.
    effective_max = min(max_shift_bins, max_len - 1)
    for shift in range(-effective_max, effective_max + 1):
        if shift >= 0:
            overlap = np.dot(ref[shift:], cand[: max_len - shift])
        else:
            overlap = np.dot(ref[: max_len + shift], cand[-shift:])
        score = float(overlap) / norm
        if score > best:
            best = score

    return min(best, 1.0)


def extract_timestamps(subtitle_path: Path) -> tuple[list[tuple[int, int]], int]:
    """Extract dialogue timestamps from a subtitle file.

    Parses the subtitle file using pysubs2, filters out non-dialogue events
    (signs, songs, etc.), and returns timing pairs.

    Args:
        subtitle_path: Path to subtitle file (SRT, ASS, or any pysubs2-supported format).

    Returns:
        Tuple of (timestamps, duration_ms) where timestamps is a list of
        (start_ms, end_ms) pairs and duration_ms is the end time of the
        last event.
    """
    pysubs2 = get_pysubs2()
    if pysubs2 is None:
        return [], 0

    try:
        subs = pysubs2.load(str(subtitle_path))
    except Exception:
        return [], 0

    timestamps: list[tuple[int, int]] = []
    for event in subs:
        if not event.text or not event.text.strip():
            continue

        # Filter non-dialogue styles
        style = getattr(event, 'style', 'Default').lower()
        if any(keyword in style for keyword in NON_DIALOGUE_STYLES):
            continue

        # Skip empty plaintext (after stripping ASS tags)
        if hasattr(event, 'plaintext') and not event.plaintext.strip():
            continue

        timestamps.append((event.start, event.end))

    duration_ms = max(end for _, end in timestamps) if timestamps else 0
    return timestamps, duration_ms


class SubtitleValidator:
    """Validates subtitle candidates against a reference track.

    Uses timing pattern fingerprinting: converts subtitle timestamps to
    binary activity vectors and measures cross-correlation to score how
    well candidates match the reference timing pattern.
    """

    def __init__(self, reference_path: Path, bin_size_ms: int = 2000) -> None:
        self._bin_size_ms = bin_size_ms
        self._ref_timestamps, self._ref_duration = extract_timestamps(reference_path)
        self._ref_vector = build_activity_vector(
            self._ref_timestamps, self._ref_duration, bin_size_ms
        )

    def score_candidate(self, candidate_path: Path) -> float:
        """Score a single candidate subtitle file against the reference.

        Both vectors are built using max(ref_duration, cand_duration) so
        they have the same length for comparison.

        Args:
            candidate_path: Path to the candidate subtitle file.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        cand_timestamps, cand_duration = extract_timestamps(candidate_path)
        duration = max(self._ref_duration, cand_duration)

        if duration == 0:
            return 0.0

        ref_vec = build_activity_vector(self._ref_timestamps, duration, self._bin_size_ms)
        cand_vec = build_activity_vector(cand_timestamps, duration, self._bin_size_ms)

        return compute_similarity(ref_vec, cand_vec)

    def validate_candidates(
        self,
        candidates: list[tuple[SubtitleMatch, Path]],
        min_threshold: float = 0.5,
    ) -> list[tuple[SubtitleMatch, Path, float]]:
        """Score all candidates, filter by threshold, sort by score descending.

        Args:
            candidates: List of (SubtitleMatch, file_path) pairs.
            min_threshold: Minimum similarity score to include in results.

        Returns:
            List of (SubtitleMatch, file_path, score) triples, sorted by
            score descending.
        """
        results: list[tuple[SubtitleMatch, Path, float]] = []
        for match, path in candidates:
            try:
                score = self.score_candidate(path)
            except Exception:
                logger.warning('Failed to score candidate %s: %s', match.subtitle_id, path)
                continue

            if score >= min_threshold:
                results.append((match, path, score))
                logger.debug('Candidate %s scored %.3f (pass)', match.subtitle_id, score)
            else:
                logger.debug(
                    'Candidate %s scored %.3f (below %.2f threshold)',
                    match.subtitle_id,
                    score,
                    min_threshold,
                )

        results.sort(key=lambda x: x[2], reverse=True)
        return results
