"""Splitter: coarse multi-sentence utterances -> subtitle-sized lines."""

from __future__ import annotations

from movie_translator.transcription.splitter import split_segment, split_segments
from movie_translator.types import DialogueLine


def test_single_sentence_unchanged():
    seg = DialogueLine(0, 3000, 'Hello there.')
    assert split_segment(seg) == [seg]


def test_two_sentences_split_proportionally():
    # Equal-length sentences over 10 s -> contiguous halves.
    seg = DialogueLine(0, 10000, 'Hello there mate. How are you, eh?')
    out = split_segment(seg)
    assert [s.text for s in out] == ['Hello there mate.', 'How are you, eh?']
    assert out[0].start_ms == 0
    assert out[-1].end_ms == 10000
    assert out[0].end_ms == out[1].start_ms  # contiguous
    assert 4000 <= out[0].end_ms <= 6000  # ~proportional to length


def test_japanese_sentences_split_on_kuten():
    seg = DialogueLine(1000, 9000, '姉ちゃんの子供か。大きくなったな。')
    out = split_segment(seg)
    assert [s.text for s in out] == ['姉ちゃんの子供か。', '大きくなったな。']
    assert out[0].start_ms == 1000
    assert out[-1].end_ms == 9000


def test_split_segments_preserves_order_and_flattens():
    segs = [
        DialogueLine(0, 4000, 'One. Two.'),
        DialogueLine(5000, 7000, 'Three.'),
    ]
    out = split_segments(segs)
    assert [s.text for s in out] == ['One.', 'Two.', 'Three.']
    assert out[2] == segs[1]
