"""Post-filter: the exact Whisper failure shapes the bake-off measured."""

from __future__ import annotations

from movie_translator.transcription.postfilter import clean_segments
from movie_translator.types import DialogueLine

AUDIO_MS = 1_422_000  # the 24-min test episode


def test_ed_hallucination_loop_collapsed_and_clamped():
    # Real shape from the PoC: looped "thanks for watching" past the audio end.
    segs = [
        DialogueLine(1_370_000, 1_372_000, '始まったばかりだった'),
        DialogueLine(1_372_000, 1_402_000, 'ご視聴ありがとうございました'),
        DialogueLine(1_402_000, 1_432_000, 'ご視聴ありがとうございました'),
    ]
    out = clean_segments(segs, AUDIO_MS)
    assert [s.text for s in out] == ['始まったばかりだった', 'ご視聴ありがとうございました']
    assert out[-1].end_ms <= AUDIO_MS


def test_segment_starting_past_audio_end_dropped():
    segs = [DialogueLine(1_425_000, 1_430_000, 'ghost')]
    assert clean_segments(segs, AUDIO_MS) == []


def test_empty_text_dropped_and_normal_kept():
    segs = [
        DialogueLine(0, 1000, '  '),
        DialogueLine(1000, 2000, 'fine'),
        DialogueLine(2000, 3000, 'fine again'),
    ]
    out = clean_segments(segs, AUDIO_MS)
    assert [s.text for s in out] == ['fine', 'fine again']
