"""Subtitle validation via line-level timing matching.

Compares downloaded subtitle candidates against a reference track by
matching individual dialogue line start times. For each candidate line,
finds the nearest reference line and checks if it falls within a tolerance
window. The fraction of matched lines is the similarity score.

This approach discriminates between episodes of the same show because
exact line timing differs between episodes even when overall dialogue
density is similar.
"""

from __future__ import annotations

import bisect
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


def compute_line_match_score(
    ref_starts: list[int],
    cand_starts: list[int],
    tolerance_ms: int = 1500,
) -> float:
    """Score how well candidate line timings match reference line timings.

    For each candidate line start time, finds the nearest reference line
    start time using binary search. A candidate line is "matched" if the
    nearest reference line is within tolerance_ms. The score is the fraction
    of candidate lines that matched.

    Args:
        ref_starts: Sorted list of reference line start times (ms).
        cand_starts: Sorted list of candidate line start times (ms).
        tolerance_ms: Maximum distance (ms) for a line to count as matched.

    Returns:
        Fraction of candidate lines matched (0.0 to 1.0).
    """
    if not ref_starts or not cand_starts:
        return 0.0

    matched = 0
    for cand_t in cand_starts:
        # Binary search for nearest reference line
        idx = bisect.bisect_left(ref_starts, cand_t)
        best_dist = float('inf')

        # Check the insertion point and its neighbor
        for i in (idx - 1, idx):
            if 0 <= i < len(ref_starts):
                dist = abs(ref_starts[i] - cand_t)
                if dist < best_dist:
                    best_dist = dist

        if best_dist <= tolerance_ms:
            matched += 1

    return matched / len(cand_starts)


def build_density_vector(
    timestamps: list[tuple[int, int]],
    duration_ms: int,
    window_ms: int = 10000,
) -> np.ndarray:
    """Build a dialogue density vector — count of events starting in each window.

    Unlike binary activity vectors, density vectors preserve how MUCH dialogue
    occurs in each window. This discriminates between episodes because different
    scenes have different amounts of dialogue even when overall density is similar.

    Args:
        timestamps: List of (start_ms, end_ms) pairs.
        duration_ms: Total timeline duration in milliseconds.
        window_ms: Width of each time window in milliseconds.

    Returns:
        Float numpy array with event counts per window.
    """
    n_bins = math.ceil(duration_ms / window_ms) if duration_ms > 0 else 0
    if n_bins == 0:
        return np.zeros(0, dtype=np.float64)

    vec = np.zeros(n_bins, dtype=np.float64)
    for start, _ in timestamps:
        bin_idx = min(start // window_ms, n_bins - 1)
        vec[bin_idx] += 1.0

    return vec


def compute_density_correlation(
    ref_density: np.ndarray,
    cand_density: np.ndarray,
    max_shift: int = 1,
) -> float:
    """Compute Pearson correlation between density vectors with shifting.

    Tries shifts in both directions and returns peak correlation.

    Args:
        ref_density: Density vector for reference.
        cand_density: Density vector for candidate.
        max_shift: Max bins to shift in each direction.

    Returns:
        Peak Pearson correlation from -1.0 to 1.0, or 0.0 if degenerate.
    """
    if len(ref_density) == 0 or len(cand_density) == 0:
        return 0.0

    # Pad to same length
    max_len = max(len(ref_density), len(cand_density))
    ref = np.zeros(max_len, dtype=np.float64)
    cand = np.zeros(max_len, dtype=np.float64)
    ref[: len(ref_density)] = ref_density
    cand[: len(cand_density)] = cand_density

    ref_std = np.std(ref)
    cand_std = np.std(cand)

    if ref_std == 0 or cand_std == 0:
        return 0.0

    best = 0.0
    effective_max = min(max_shift, max_len - 1)
    for shift in range(-effective_max, effective_max + 1):
        if shift >= 0:
            r = ref[shift:]
            c = cand[: max_len - shift]
        else:
            r = ref[: max_len + shift]
            c = cand[-shift:]

        if len(r) < 3:
            continue

        r_mean = np.mean(r)
        c_mean = np.mean(c)
        r_std = np.std(r)
        c_std = np.std(c)

        if r_std == 0 or c_std == 0:
            continue

        corr = float(np.mean((r - r_mean) * (c - c_mean)) / (r_std * c_std))
        if corr > best:
            best = corr

    return best


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

    Uses line-level timing matching: for each candidate dialogue line,
    finds the nearest reference line by start time and checks if it falls
    within a tolerance window. The fraction of matched lines is the score.
    """

    def __init__(self, reference_path: Path, window_ms: int = 10000) -> None:
        self._window_ms = window_ms
        self._ref_timestamps, self._ref_duration = extract_timestamps(reference_path)

    def score_candidate(self, candidate_path: Path) -> float:
        """Score a single candidate subtitle file against the reference.

        Uses line-level timing match: for each candidate line, finds the
        nearest reference line by start time. A candidate line is "matched"
        if the nearest reference line is within tolerance. The score is the
        fraction of candidate lines matched.

        This is robust to differences in OP/ED style filtering between
        reference and candidate files.

        Args:
            candidate_path: Path to the candidate subtitle file.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        cand_timestamps, cand_duration = extract_timestamps(candidate_path)

        if not cand_timestamps or not self._ref_timestamps:
            return 0.0

        ref_starts = sorted(s for s, _ in self._ref_timestamps)
        cand_starts = sorted(s for s, _ in cand_timestamps)

        return compute_line_match_score(ref_starts, cand_starts, tolerance_ms=2000)

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
