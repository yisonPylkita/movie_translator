"""Tests for OCR post-processing: fuzzy merge + garbage filtering."""

from __future__ import annotations

from movie_translator.hardsub.postprocess import (
    CleanLine,
    is_dialogue,
    merge_ocr_results,
    to_srt,
)


def test_is_dialogue_accepts_real_polish():
    assert is_dialogue('To legendarna broń prawda?')
    assert is_dialogue('Dziękuję ci. To był wypadek.')


def test_is_dialogue_rejects_garbage():
    assert not is_dialogue('‡.|.tz')  # symbols, < 3 letters
    assert not is_dialogue('--|')
    assert not is_dialogue('0 *FE')  # mostly symbols/digits
    assert not is_dialogue('')


def test_merge_folds_near_duplicate_jitter_into_one_line():
    # The same subtitle OCR'd slightly differently across consecutive frames.
    frames = [
        (10000, 'To był wypadek.'),
        (10333, 'To byl wypadek.'),
        (10666, 'To byt wypadek.'),
        (11000, 'Coś zupełnie innego tutaj.'),
    ]
    lines = merge_ocr_results(frames, min_duration_ms=200)
    assert len(lines) == 2
    # Canonical pick is a real variant; the block spans the whole jitter run.
    assert lines[0].text in {'To był wypadek.', 'To byl wypadek.', 'To byt wypadek.'}
    assert lines[0].start_ms == 10000
    assert lines[0].end_ms == 11000


def test_merge_drops_transient_single_frame_garbage():
    # OP karaoke: every frame different gibberish, each lasts one frame.
    frames = [
        (0, '# Baá'),
        (333, '‡.|.tz'),
        (666, 'fbt fL'),
        (1000, 'Prawdziwe zdanie dialogu które trwa.'),
        (3000, ''),
    ]
    lines = merge_ocr_results(frames, min_duration_ms=500)
    # Only the persistent real line survives (1000..3000 = 2s).
    assert len(lines) == 1
    assert lines[0].text.startswith('Prawdziwe zdanie')


def test_merge_blank_frame_closes_block():
    frames = [
        (0, 'Zdanie pierwsze tutaj jest.'),
        (2000, ''),
        (4000, 'Zdanie drugie tutaj jest.'),
        (6000, ''),
    ]
    lines = merge_ocr_results(frames, min_duration_ms=200)
    assert len(lines) == 2
    assert lines[0].end_ms == 2000


def test_to_srt_format():
    lines = [CleanLine(1000, 2500, 'Cześć świecie.')]
    srt = to_srt(lines)
    assert '1\n00:00:01,000 --> 00:00:02,500\nCześć świecie.' in srt
