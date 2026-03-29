"""Tests for subtitle_fetch.align module."""

from pathlib import Path

import pytest

from movie_translator.subtitle_fetch.align import (
    align_to_reference,
    apply_offset,
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


# Reference: representative dialogue spread across a ~5-minute timeline
_REF_LINES = [
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

REFERENCE_SRT = _make_srt(_REF_LINES)


def _shift_lines(lines, offset_ms):
    """Create a copy of lines shifted by offset_ms."""
    return [(s + offset_ms, e + offset_ms, t) for s, e, t in lines]


class TestEstimateOffset:
    """Tests for estimate_offset() using cross-correlation."""

    def test_identical_timings_returns_zero(self):
        ts = _to_timestamps([1000, 4000, 7000, 10000, 14000])
        offset = estimate_offset(ts, ts)
        assert offset == 0

    def test_positive_offset_candidate_early(self):
        ref = _to_timestamps([1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000])
        # Candidate is 1500ms early (needs shifting forward by +1500)
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
        # This is the case that broke nearest-neighbor: 5s offset with 3s line spacing
        ref = _to_timestamps([1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000])
        cand = _to_timestamps(
            [s + 5000 for s in [1000, 4000, 7000, 10000, 14000, 60000, 65000, 70000]]
        )
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(-5000, abs=200)

    def test_three_second_offset_like_konosuba(self):
        # Simulate the real-world case: ~3s offset, lines every 2-3s
        ref = _to_timestamps(
            [
                7000,
                9000,
                12000,
                19000,
                21000,
                33000,
                36000,
                42000,
                46000,
                54000,
                60000,
                62000,
                67000,
                70000,
                86000,
            ]
        )
        cand = _to_timestamps(
            [
                4000,
                6000,
                9000,
                16000,
                18000,
                30000,
                33000,
                39000,
                43000,
                51000,
                58000,
                59000,
                65000,
                67000,
                83000,
            ]
        )
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(3000, abs=200)

    def test_different_line_counts(self):
        # Candidate has more lines (different splitting) but same 2s offset.
        # Each ref line is 2s long; candidate splits each into two 1s lines.
        ref = [
            (1000, 3000),
            (10000, 12000),
            (20000, 22000),
            (60000, 62000),
            (80000, 82000),
            (150000, 152000),
        ]
        cand = [
            (3000, 4000),
            (4000, 5000),
            (12000, 13000),
            (13000, 14000),
            (22000, 23000),
            (23000, 24000),
            (62000, 63000),
            (63000, 64000),
            (82000, 83000),
            (83000, 84000),
            (152000, 153000),
            (153000, 154000),
        ]
        offset = estimate_offset(ref, cand)
        assert offset == pytest.approx(-2000, abs=200)

    def test_empty_reference_returns_none(self):
        assert estimate_offset([], _to_timestamps([1000, 2000])) is None

    def test_empty_candidate_returns_none(self):
        assert estimate_offset(_to_timestamps([1000, 2000]), []) is None

    def test_custom_bin_size(self):
        ref = _to_timestamps([10000, 20000, 30000, 60000, 80000])
        cand = _to_timestamps([s + 3000 for s in [10000, 20000, 30000, 60000, 80000]])
        offset = estimate_offset(ref, cand, bin_size_ms=200)
        assert offset == pytest.approx(-3000, abs=200)


class TestApplyOffset:
    """Tests for apply_offset()."""

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
        content = _make_srt([(1000, 3000, 'A'), (5000, 7000, 'B')])
        srt.write_text(content)

        apply_offset(srt, 0)

        timestamps, _ = extract_timestamps(srt)
        assert timestamps[0] == (1000, 3000)
        assert timestamps[1] == (5000, 7000)

    def test_preserves_text_content(self, tmp_path: Path):
        srt = tmp_path / 'test.srt'
        srt.write_text(_make_srt([(1000, 3000, 'Hello world'), (5000, 7000, 'Second line')]))

        apply_offset(srt, 1000)

        pysubs2 = __import__('pysubs2')
        subs = pysubs2.load(str(srt))
        texts = [e.plaintext.strip() for e in subs]
        assert 'Hello world' in texts
        assert 'Second line' in texts


class TestAlignToReference:
    """Tests for align_to_reference() end-to-end."""

    def test_no_correction_when_already_aligned(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(REFERENCE_SRT)

        offset = align_to_reference(cand, ref)
        assert offset == 0

    def test_corrects_late_subtitles(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        # Candidate is 2 seconds late
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, 2000)))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(-2000, abs=100)

        # Verify the file was actually corrected
        timestamps, _ = extract_timestamps(cand)
        ref_timestamps, _ = extract_timestamps(ref)
        for (cs, _), (rs, _) in zip(timestamps, ref_timestamps, strict=True):
            assert abs(cs - rs) < 150

    def test_corrects_early_subtitles(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, -1500)))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(1500, abs=100)

    def test_skips_offset_below_threshold(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        # Candidate is only 50ms off — well below 150ms threshold
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, 50)))

        offset = align_to_reference(cand, ref, min_offset_ms=150)
        assert offset == 0

    def test_skips_offset_above_safety_limit(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        # Candidate is 20 seconds off — above 15s safety limit.
        # The quality check also rejects spurious matches within the window.
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, 20000)))

        offset = align_to_reference(cand, ref, max_offset_ms=15000)
        assert offset == 0

    def test_corrects_three_second_offset(self, tmp_path: Path):
        """Regression test for the Konosuba S1E01 case."""
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, -3000)))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(3000, abs=100)

    def test_works_with_noisy_candidate(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        shifted = _shift_lines(_REF_LINES, 3000)
        shifted.append((100000, 102000, 'noise1'))
        shifted.append((200000, 202000, 'noise2'))
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(shifted))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(-3000, abs=200)

    def test_returns_zero_for_empty_candidate(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text('')

        offset = align_to_reference(cand, ref)
        assert offset == 0

    def test_returns_zero_for_empty_reference(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text('')
        cand = tmp_path / 'cand.srt'
        cand.write_text(REFERENCE_SRT)

        offset = align_to_reference(cand, ref)
        assert offset == 0

    def test_corrects_fractional_second_offset(self, tmp_path: Path):
        ref = tmp_path / 'ref.srt'
        ref.write_text(REFERENCE_SRT)
        cand = tmp_path / 'cand.srt'
        cand.write_text(_make_srt(_shift_lines(_REF_LINES, 700)))

        offset = align_to_reference(cand, ref)
        assert offset == pytest.approx(-700, abs=100)
