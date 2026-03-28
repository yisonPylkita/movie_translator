"""Subtitle validation via timing pattern fingerprinting.

Compares downloaded subtitle candidates against a reference track by
converting timing information into binary activity vectors and measuring
their cross-correlation.
"""

from __future__ import annotations

import math

import numpy as np


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
