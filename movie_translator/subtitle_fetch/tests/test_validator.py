"""Tests for subtitle_fetch.validator module."""

from pathlib import Path

import numpy as np
import pytest

from movie_translator.subtitle_fetch.types import SubtitleMatch
from movie_translator.subtitle_fetch.validator import (
    SubtitleValidator,
    build_activity_vector,
    compute_similarity,
    extract_timestamps,
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


# -- Helpers for creating subtitle fixture files --

SRT_CONTENT = """\
1
00:00:01,000 --> 00:00:03,000
Hello, world!

2
00:00:05,000 --> 00:00:07,000
Second line.

3
00:00:10,000 --> 00:00:12,000
Third line.
"""

ASS_CONTENT = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: OP,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Song-Lyrics,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello from ASS
Dialogue: 0,0:00:05.00,0:00:07.00,Sign,,0,0,0,,{\\an8}Sign text
Dialogue: 0,0:00:08.00,0:00:10.00,Default,,0,0,0,,Second dialogue
Dialogue: 0,0:00:12.00,0:00:14.00,OP,,0,0,0,,Opening song
Dialogue: 0,0:00:15.00,0:00:17.00,Song-Lyrics,,0,0,0,,La la la
"""


class TestExtractTimestamps:
    """Tests for extract_timestamps()."""

    def test_srt_extracts_all_events(self, tmp_path: Path):
        srt = tmp_path / 'test.srt'
        srt.write_text(SRT_CONTENT)
        timestamps, duration = extract_timestamps(srt)
        assert len(timestamps) == 3
        assert timestamps[0] == (1000, 3000)
        assert timestamps[1] == (5000, 7000)
        assert timestamps[2] == (10000, 12000)
        assert duration == 12000

    def test_ass_filters_non_dialogue_styles(self, tmp_path: Path):
        ass_file = tmp_path / 'test.ass'
        ass_file.write_text(ASS_CONTENT)
        timestamps, duration = extract_timestamps(ass_file)
        # Only Default style events should be included (not Sign, OP, Song-Lyrics)
        assert len(timestamps) == 2
        assert timestamps[0] == (1000, 3000)
        assert timestamps[1] == (8000, 10000)
        assert duration == 10000

    def test_empty_subtitle_file(self, tmp_path: Path):
        srt = tmp_path / 'empty.srt'
        srt.write_text('')
        timestamps, duration = extract_timestamps(srt)
        assert timestamps == []
        assert duration == 0

    def test_srt_with_empty_text_lines_skipped(self, tmp_path: Path):
        content = """\
1
00:00:01,000 --> 00:00:03,000
Hello

2
00:00:05,000 --> 00:00:07,000

"""
        srt = tmp_path / 'sparse.srt'
        srt.write_text(content)
        timestamps, duration = extract_timestamps(srt)
        # Second event has empty text, should be skipped
        assert len(timestamps) == 1
        assert timestamps[0] == (1000, 3000)
        assert duration == 3000

    def test_duration_is_max_end_time(self, tmp_path: Path):
        content = """\
1
00:00:01,000 --> 00:00:03,000
First

2
00:00:20,000 --> 00:00:25,000
Last
"""
        srt = tmp_path / 'duration.srt'
        srt.write_text(content)
        _, duration = extract_timestamps(srt)
        assert duration == 25000

    def test_returns_sorted_timestamps(self, tmp_path: Path):
        # pysubs2 should return events in order
        srt = tmp_path / 'ordered.srt'
        srt.write_text(SRT_CONTENT)
        timestamps, _ = extract_timestamps(srt)
        starts = [t[0] for t in timestamps]
        assert starts == sorted(starts)


# -- Helpers for SubtitleValidator tests --
# Density correlation needs enough dialogue lines spread over time to be meaningful.
# These fixtures simulate a ~5-minute episode with clustered dialogue.


def _make_srt(lines: list[tuple[int, int, str]]) -> str:
    """Build an SRT string from (start_ms, end_ms, text) tuples."""
    parts = []
    for i, (start, end, text) in enumerate(lines, 1):
        sh, sm, ss, sms = (
            start // 3600000,
            (start % 3600000) // 60000,
            (start % 60000) // 1000,
            start % 1000,
        )
        eh, em, es, ems = (
            end // 3600000,
            (end % 3600000) // 60000,
            (end % 60000) // 1000,
            end % 1000,
        )
        parts.append(
            f'{i}\n{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}\n{text}\n'
        )
    return '\n'.join(parts)


# Reference: dialogue clustered at 0-30s, 60-90s, 150-180s, 240-270s
_REF_LINES = [
    # Cluster 1: 0-30s (dense)
    (1000, 3000, 'A'),
    (4000, 6000, 'B'),
    (7000, 9000, 'C'),
    (10000, 12000, 'D'),
    (14000, 16000, 'E'),
    (18000, 20000, 'F'),
    (22000, 24000, 'G'),
    (26000, 28000, 'H'),
    # Cluster 2: 60-90s (medium)
    (60000, 62000, 'I'),
    (65000, 67000, 'J'),
    (70000, 72000, 'K'),
    (75000, 77000, 'L'),
    (80000, 82000, 'M'),
    # Gap: 90-150s (silence)
    # Cluster 3: 150-180s (dense)
    (150000, 152000, 'N'),
    (153000, 155000, 'O'),
    (157000, 159000, 'P'),
    (161000, 163000, 'Q'),
    (165000, 167000, 'R'),
    (170000, 172000, 'S'),
    (175000, 177000, 'T'),
    # Gap: 180-240s
    # Cluster 4: 240-270s (sparse)
    (240000, 242000, 'U'),
    (250000, 252000, 'V'),
    (260000, 262000, 'W'),
]

REFERENCE_SRT = _make_srt(_REF_LINES)

# Matching: same cluster pattern, slightly shifted timing (like a translation)
_MATCH_LINES = [
    (1200, 3200, 'A2'),
    (4200, 6200, 'B2'),
    (7500, 9500, 'C2'),
    (10500, 12500, 'D2'),
    (14500, 16500, 'E2'),
    (18500, 20500, 'F2'),
    (22500, 24500, 'G2'),
    (26500, 28500, 'H2'),
    (60500, 62500, 'I2'),
    (65500, 67500, 'J2'),
    (70500, 72500, 'K2'),
    (75500, 77500, 'L2'),
    (80500, 82500, 'M2'),
    (150500, 152500, 'N2'),
    (153500, 155500, 'O2'),
    (157500, 159500, 'P2'),
    (161500, 163500, 'Q2'),
    (165500, 167500, 'R2'),
    (170500, 172500, 'S2'),
    (175500, 177500, 'T2'),
    (240500, 242500, 'U2'),
    (250500, 252500, 'V2'),
    (260500, 262500, 'W2'),
]

MATCHING_SRT = _make_srt(_MATCH_LINES)

# Mismatched: uniform sparse dialogue (like a different episode with action scenes)
# Reference has dense clusters with large gaps; this has evenly spaced lines.
_MISMATCH_LINES = [
    # Evenly distributed: one line every ~15s across 300s (very different density shape)
    (5000, 7000, 'X'),
    (20000, 22000, 'Y'),
    (35000, 37000, 'Z'),
    (50000, 52000, 'AA'),
    (65000, 67000, 'BB'),
    (80000, 82000, 'CC'),
    (95000, 97000, 'DD'),
    (110000, 112000, 'EE'),
    (125000, 127000, 'FF'),
    (140000, 142000, 'GG'),
    (155000, 157000, 'HH'),
    (170000, 172000, 'II'),
    (185000, 187000, 'JJ'),
    (200000, 202000, 'KK'),
    (215000, 217000, 'LL'),
    (230000, 232000, 'MM'),
    (245000, 247000, 'NN'),
    (260000, 262000, 'OO'),
    (275000, 277000, 'PP'),
    (290000, 292000, 'QQ'),
]

MISMATCHED_SRT = _make_srt(_MISMATCH_LINES)


def _make_match(subtitle_id: str, score: float = 0.8) -> SubtitleMatch:
    return SubtitleMatch(
        language='eng',
        source='test',
        subtitle_id=subtitle_id,
        release_name='test-release',
        format='srt',
        score=score,
        hash_match=False,
    )


class TestSubtitleValidator:
    """Tests for SubtitleValidator class."""

    def test_init_loads_reference(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)
        validator = SubtitleValidator(ref)
        assert validator._ref_timestamps
        assert validator._ref_duration > 0

    def test_score_candidate_matching_is_high(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'candidate.srt'
        cand.write_text(MATCHING_SRT)

        validator = SubtitleValidator(ref)
        score = validator.score_candidate(cand)
        assert score >= 0.7

    def test_score_candidate_mismatched_is_low(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'candidate.srt'
        cand.write_text(MISMATCHED_SRT)

        validator = SubtitleValidator(ref)
        score = validator.score_candidate(cand)
        assert score < 0.5

    def test_validate_candidates_filters_by_threshold(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)

        good = tmp_path / 'good.srt'
        good.write_text(MATCHING_SRT)
        bad = tmp_path / 'bad.srt'
        bad.write_text(MISMATCHED_SRT)

        validator = SubtitleValidator(ref)
        candidates = [
            (_make_match('good'), good),
            (_make_match('bad'), bad),
        ]

        results = validator.validate_candidates(candidates, min_threshold=0.5)
        assert len(results) == 1
        match, path, score = results[0]
        assert match.subtitle_id == 'good'
        assert score >= 0.5

    def test_validate_candidates_sorted_by_score_descending(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)

        cand1 = tmp_path / 'cand1.srt'
        cand1.write_text(MATCHING_SRT)
        cand2 = tmp_path / 'cand2.srt'
        cand2.write_text(MISMATCHED_SRT)

        validator = SubtitleValidator(ref)
        candidates = [
            (_make_match('bad'), cand2),
            (_make_match('good'), cand1),
        ]

        results = validator.validate_candidates(candidates, min_threshold=0.0)
        assert len(results) == 2
        assert results[0][2] >= results[1][2]
        assert results[0][0].subtitle_id == 'good'

    def test_validate_candidates_empty_list(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)

        validator = SubtitleValidator(ref)
        results = validator.validate_candidates([])
        assert results == []

    def test_custom_window_size(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)

        validator = SubtitleValidator(ref, window_ms=5000)
        assert validator._window_ms == 5000

    def test_score_empty_candidate(self, tmp_path: Path):
        ref = tmp_path / 'reference.srt'
        ref.write_text(REFERENCE_SRT)

        cand = tmp_path / 'candidate.srt'
        cand.write_text('')

        validator = SubtitleValidator(ref)
        score = validator.score_candidate(cand)
        assert score == 0.0
