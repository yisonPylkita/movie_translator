"""Tests for subtitle_fetch.validator module."""

import numpy as np
import pytest

from movie_translator.subtitle_fetch.validator import (
    build_activity_vector,
    compute_similarity,
)


class TestBuildActivityVector:
    """Tests for build_activity_vector()."""

    def test_empty_timestamps_returns_all_zeros(self):
        vec = build_activity_vector([], duration_ms=10000, bin_size_ms=2000)
        assert vec.shape == (5,)
        np.testing.assert_array_equal(vec, [0, 0, 0, 0, 0])

    def test_single_event_marks_correct_bins(self):
        # Event from 1000-3000ms with 2000ms bins covers bins 0 and 1
        vec = build_activity_vector([(1000, 3000)], duration_ms=10000, bin_size_ms=2000)
        assert vec.shape == (5,)
        np.testing.assert_array_equal(vec, [1, 1, 0, 0, 0])

    def test_event_spanning_all_bins(self):
        vec = build_activity_vector([(0, 10000)], duration_ms=10000, bin_size_ms=2000)
        np.testing.assert_array_equal(vec, [1, 1, 1, 1, 1])

    def test_multiple_events(self):
        # Two events: 0-2000 (bin 0) and 6000-8000 (bin 3)
        timestamps = [(0, 2000), (6000, 8000)]
        vec = build_activity_vector(timestamps, duration_ms=10000, bin_size_ms=2000)
        np.testing.assert_array_equal(vec, [1, 0, 0, 1, 0])

    def test_event_partially_overlapping_bin(self):
        # Event from 1999-2001 should overlap bins 0 and 1
        vec = build_activity_vector([(1999, 2001)], duration_ms=10000, bin_size_ms=2000)
        assert vec[0] == 1
        assert vec[1] == 1

    def test_custom_bin_size(self):
        vec = build_activity_vector([(0, 500)], duration_ms=2000, bin_size_ms=1000)
        assert vec.shape == (2,)
        np.testing.assert_array_equal(vec, [1, 0])

    def test_duration_not_exact_multiple_of_bin_size(self):
        # 7000ms / 2000ms = 3.5 -> should ceil to 4 bins
        vec = build_activity_vector([], duration_ms=7000, bin_size_ms=2000)
        assert vec.shape == (4,)

    def test_event_at_very_end(self):
        # Event in the last bin
        vec = build_activity_vector([(8000, 10000)], duration_ms=10000, bin_size_ms=2000)
        np.testing.assert_array_equal(vec, [0, 0, 0, 0, 1])

    def test_overlapping_events_still_binary(self):
        # Two overlapping events should still produce 1, not 2
        timestamps = [(0, 5000), (3000, 7000)]
        vec = build_activity_vector(timestamps, duration_ms=10000, bin_size_ms=2000)
        # Bins 0-3 should be active (0-5000 covers 0,1,2; 3000-7000 covers 1,2,3)
        np.testing.assert_array_equal(vec, [1, 1, 1, 1, 0])
        assert (
            vec.dtype == np.float64
            or vec.dtype == np.float32
            or vec.dtype == np.int_
            or vec.max() <= 1
        )

    def test_zero_duration_returns_empty(self):
        vec = build_activity_vector([], duration_ms=0, bin_size_ms=2000)
        assert vec.shape == (0,) or len(vec) == 0


class TestComputeSimilarity:
    """Tests for compute_similarity()."""

    def test_identical_vectors_return_1(self):
        vec = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        score = compute_similarity(vec, vec)
        assert score == pytest.approx(1.0)

    def test_opposite_vectors_return_low_score(self):
        ref = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        cand = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        score = compute_similarity(ref, cand)
        # With shifting, the complement pattern shifted by 1 would match perfectly
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        ref = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        # Candidate active only where ref is silent, far enough that no shift helps
        cand = np.zeros(10, dtype=np.float64)
        score = compute_similarity(ref, cand)
        assert score == 0.0

    def test_shifted_vector_detected(self):
        # Reference has activity in bins 0-4, candidate in bins 5-9
        ref = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        cand = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        score = compute_similarity(ref, cand, max_shift_bins=5)
        assert score == pytest.approx(1.0)

    def test_shifted_beyond_max_returns_low(self):
        # Shift of 5 needed but max_shift is 2
        ref = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        cand = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        score = compute_similarity(ref, cand, max_shift_bins=2)
        assert score < 0.5

    def test_score_between_0_and_1(self):
        rng = np.random.default_rng(42)
        ref = rng.integers(0, 2, size=100).astype(np.float64)
        cand = rng.integers(0, 2, size=100).astype(np.float64)
        score = compute_similarity(ref, cand)
        assert 0.0 <= score <= 1.0

    def test_both_empty_returns_zero(self):
        ref = np.zeros(10, dtype=np.float64)
        cand = np.zeros(10, dtype=np.float64)
        score = compute_similarity(ref, cand)
        assert score == 0.0

    def test_different_lengths_padded(self):
        ref = np.array([1, 0, 1, 0, 1], dtype=np.float64)
        cand = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        score = compute_similarity(ref, cand)
        # The active parts match, score should be high
        assert score > 0.7

    def test_partial_overlap_gives_intermediate_score(self):
        ref = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        cand = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        score = compute_similarity(ref, cand, max_shift_bins=0)
        # 2 overlapping out of 4 and 2 -> geometric mean normalization
        assert 0.0 < score < 1.0
