"""Tests for subtitle_fetch.align module."""

from pathlib import Path

import pytest

from movie_translator.subtitle_fetch.align import (
    align_to_reference,
    apply_offset,
    detect_op_gap,
    estimate_offset,
)
from movie_translator.subtitle_fetch.validator import extract_timestamps


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


def _to_timestamps(starts: list[int], duration: int = 2000) -> list[tuple[int, int]]:
    """Convert start times to (start, end) timestamp pairs."""
    return [(s, s + duration) for s in starts]


# ---------------------------------------------------------------------------
# Fixtures: simulated episode with OP gap
# ---------------------------------------------------------------------------

# Reference track with OP gap: dialogue at 0-100s, OP at 110-200s, dialogue at 210-500s
_REF_PRE_OP = [
    (5000, 7000, 'A'),
    (10000, 12000, 'B'),
    (15000, 17000, 'C'),
    (20000, 22000, 'D'),
    (30000, 32000, 'E'),
    (40000, 42000, 'F'),
    (50000, 52000, 'G'),
    (60000, 62000, 'H'),
    (70000, 72000, 'I'),
    (80000, 82000, 'J'),
    (90000, 92000, 'K'),
    (100000, 102000, 'L'),
]

_REF_POST_OP = [
    (210000, 212000, 'M'),
    (215000, 217000, 'N'),
    (220000, 222000, 'O'),
    (230000, 232000, 'P'),
    (240000, 242000, 'Q'),
    (250000, 252000, 'R'),
    (260000, 262000, 'S'),
    (270000, 272000, 'T'),
    (280000, 282000, 'U'),
    (290000, 292000, 'V'),
    (300000, 302000, 'W'),
    (310000, 312000, 'X'),
    (350000, 352000, 'Y'),
    (400000, 402000, 'Z'),
    (450000, 452000, 'AA'),
    (500000, 502000, 'BB'),
]

_REF_LINES = _REF_PRE_OP + _REF_POST_OP
REF_SRT = _make_srt(_REF_LINES)


def _shift_lines(lines, offset_ms):
    return [(s + offset_ms, e + offset_ms, t) for s, e, t in lines]


def _make_op_removed_candidate(pre_offset_ms, post_offset_ms):
    """Build a candidate SRT where the OP gap has been removed.

    pre_offset_ms: how much earlier the candidate pre-OP events are vs reference
    post_offset_ms: how much earlier the candidate post-OP events are vs reference
    """
    pre_lines = _shift_lines(_REF_PRE_OP, pre_offset_ms)
    post_lines = _shift_lines(_REF_POST_OP, post_offset_ms)
    return _make_srt(pre_lines + post_lines)


# Simple reference without OP gap (for global alignment tests)
_SIMPLE_REF = [
    (1000, 3000, 'A'),
    (4000, 6000, 'B'),
    (7000, 9000, 'C'),
    (10000, 12000, 'D'),
    (14000, 16000, 'E'),
    (18000, 20000, 'F'),
    (60000, 62000, 'G'),
    (65000, 67000, 'H'),
    (70000, 72000, 'I'),
    (150000, 152000, 'J'),
    (153000, 155000, 'K'),
    (157000, 159000, 'L'),
    (240000, 242000, 'M'),
    (250000, 252000, 'N'),
    (260000, 262000, 'O'),
]
SIMPLE_REF_SRT = _make_srt(_SIMPLE_REF)


# ---------------------------------------------------------------------------
# Tests: detect_op_gap
# ---------------------------------------------------------------------------


class TestDetectOpGap:
    def test_finds_op_gap(self):
        ts = [(s, e) for s, e, _ in _REF_LINES]
        gap = detect_op_gap(ts)
        assert gap is not None
        gap_start, gap_end = gap
        # Gap should be between last pre-OP event end and first post-OP event start
        assert 100000 <= gap_start <= 110000
        assert 205000 <= gap_end <= 215000

    def test_no_gap_when_none_exists(self):
        ts = _to_timestamps([1000, 5000, 10000, 15000, 20000])
        gap = detect_op_gap(ts)
        assert gap is None

    def test_ignores_gaps_outside_search_window(self):
        # Gap at 500s — outside the 30s-360s search window
        ts = [(1000, 3000), (5000, 7000), (500000, 502000), (600000, 602000)]
        gap = detect_op_gap(ts, search_start_ms=30000, search_end_ms=360000)
        assert gap is None

    def test_finds_largest_gap_in_window(self):
        # Two gaps: 80s gap at 60s mark, 200s gap at 120s mark
        ts = [
            (10000, 12000),
            (50000, 52000),
            # 80s gap (52s to 132s)
            (132000, 134000),
            (140000, 142000),
            # 200s gap (142s to 342s)
            (342000, 344000),
            (400000, 402000),
        ]
        gap = detect_op_gap(ts)
        assert gap is not None
        gap_start, gap_end = gap
        # Should find the 200s gap, not the 80s one
        assert gap_end - gap_start > 150000

    def test_empty_timestamps(self):
        assert detect_op_gap([]) is None


# ---------------------------------------------------------------------------
# Tests: estimate_offset (cross-correlation)
# ---------------------------------------------------------------------------


class TestEstimateOffset:
    def test_identical_timings_returns_zero(self):
        ts = _to_timestamps([1000, 4000, 7000, 10000, 14000])
        offset = estimate_offset(ts, ts)
        assert offset == 0

    def test_positive_offset_candidate_early(self):
        ref = _to_timestamps([1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000])
        cand = _to_timestamps(
            [s - 1500 for s in [1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000]]
        )
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(1500, abs=100)

    def test_negative_offset_candidate_late(self):
        ref = _to_timestamps([1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000])
        cand = _to_timestamps(
            [s + 2000 for s in [1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000]]
        )
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(-2000, abs=100)

    def test_large_offset_with_dense_lines(self):
        ref = _to_timestamps([1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000])
        cand = _to_timestamps(
            [s + 5000 for s in [1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000]]
        )
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(-5000, abs=200)

    def test_empty_reference_returns_none(self):
        assert estimate_offset([], _to_timestamps([1000, 2000])) is None

    def test_empty_candidate_returns_none(self):
        assert estimate_offset(_to_timestamps([1000, 2000]), []) is None


# ---------------------------------------------------------------------------
# Tests: apply_offset
# ---------------------------------------------------------------------------


class TestApplyOffset:
    def test_shifts_all_events_forward(self, tmp_path: Path):
        srt = tmp_path / 'test.srt'
        srt.write_text(_make_srt([(1000, 3000, 'A'), (5000, 7000, 'B')]))
        apply_offset(srt, 2000)
        timestamps, _ = extract_timestamps(srt)
        assert timestamps[0] == (3000, 5000)
        assert timestamps[1] == (7000, 9000)

    def test_shifts_all_events_backward(self, tmp_path: Path):
        srt = tmp_path / 'test.srt'
        srt.write_text(_make_srt([(5000, 7000, 'A'), (10000, 12000, 'B')]))
        apply_offset(srt, -2000)
        timestamps, _ = extract_timestamps(srt)
        assert timestamps[0] == (3000, 5000)
        assert timestamps[1] == (8000, 10000)

    def test_zero_offset_no_change(self, tmp_path: Path):
        srt = tmp_path / 'test.srt'
        srt.write_text(_make_srt([(1000, 3000, 'A'), (5000, 7000, 'B')]))
        apply_offset(srt, 0)
        timestamps, _ = extract_timestamps(srt)
        assert timestamps[0] == (1000, 3000)
        assert timestamps[1] == (5000, 7000)


# ---------------------------------------------------------------------------
# Tests: align_to_reference (global — no OP gap)
# ---------------------------------------------------------------------------


class TestAlignToReferenceGlobal:
    def test_no_correction_when_aligned(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(SIMPLE_REF_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(SIMPLE_REF_SRT)
        assert align_to_reference(cand, ref) == 0

    def test_corrects_late_subtitles(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(SIMPLE_REF_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_SIMPLE_REF, 2000)))
        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(-2000, abs=100)

    def test_returns_zero_for_empty_candidate(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(SIMPLE_REF_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text('')
        assert align_to_reference(cand, ref) == 0


# ---------------------------------------------------------------------------
# Tests: align_to_reference (piecewise — OP gap detected)
# ---------------------------------------------------------------------------


class TestAlignToReferencePiecewise:
    def test_corrects_op_removed_candidate(self, tmp_path: Path):
        """Core test: candidate was timed to a video with OP removed."""
        ref = tmp_path / 'ref.srt'
        ref.write_text(REF_SRT)

        # Candidate: pre-OP is 2s early, post-OP is ~110s early (OP duration removed)
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_op_removed_candidate(pre_offset_ms=-2000, post_offset_ms=-110000))

        offset = align_to_reference(cand, ref)
        # Should return the post-OP offset (dominant)
        assert offset == pytest.approx(110000, abs=1000)

        # Verify both segments are corrected
        timestamps, _ = extract_timestamps(cand)
        ref_timestamps, _ = extract_timestamps(ref)
        ref_starts = sorted(s for s, _ in ref_timestamps)
        cand_starts = sorted(s for s, _ in timestamps)

        # Pre-OP lines should be within ~200ms of reference
        for cs in cand_starts:
            if cs < 105000:
                # Find nearest reference line
                dists = [abs(cs - rs) for rs in ref_starts]
                assert min(dists) < 500, f'Pre-OP line at {cs}ms not aligned'

        # Post-OP lines should be within ~1500ms of reference
        for cs in cand_starts:
            if cs > 200000:
                dists = [abs(cs - rs) for rs in ref_starts]
                assert min(dists) < 2000, f'Post-OP line at {cs}ms not aligned'

    def test_uniform_offset_with_op_gap(self, tmp_path: Path):
        """When both segments have the same offset, apply a single shift."""
        ref = tmp_path / 'ref.srt'
        ref.write_text(REF_SRT)

        # Same offset for both segments — should use uniform shift
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_op_removed_candidate(-2000, -2000))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(2000, abs=200)

    def test_only_pre_op_offset(self, tmp_path: Path):
        """When post-OP is already aligned but pre-OP is off."""
        ref = tmp_path / 'ref.srt'
        ref.write_text(REF_SRT)

        # Pre-OP is 3s early, post-OP is correctly timed
        pre = _shift_lines(_REF_PRE_OP, -3000)
        post = [(s, e, t) for s, e, t in _REF_POST_OP]
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(pre + post))

        offset = align_to_reference(cand, ref)
        # Should still apply piecewise correction
        assert offset is not None

    def test_skips_tiny_offsets(self, tmp_path: Path):
        """Offsets below threshold are ignored."""
        ref = tmp_path / 'ref.srt'
        ref.write_text(REF_SRT)

        # Both segments only 50ms off
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_op_removed_candidate(-50, -50))
        assert align_to_reference(cand, ref, min_offset_ms=150) == 0
