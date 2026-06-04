"""Split coarse ASR utterances into subtitle-sized lines.

Apple SpeechAnalyzer returns long multi-sentence utterances (the PoC measured
16 segments where the reference had ~58 dialogue lines). Subtitles want one
sentence-ish line at a time, so we split on sentence punctuation and allocate
the utterance's time span proportionally to each piece's character length.
"""

from __future__ import annotations

import re

from ..types import DialogueLine

# A sentence piece: text up to (and including) sentence punctuation plus any
# trailing quote, or a final unpunctuated remainder.
_SENTENCE = re.compile(r'[^.!?。！？]*[.!?。！？]+["」』]?\s*|[^.!?。！？]+$')
# A piece must contain at least one word character (any script) to stand on
# its own; bare punctuation runs ('...') glue onto the following sentence.
_HAS_WORD = re.compile(r'\w')


def _pieces(text: str) -> list[str]:
    raw = [p.strip() for p in _SENTENCE.findall(text) if p.strip()]
    out: list[str] = []
    carry = ''
    for p in raw:
        if not _HAS_WORD.search(p):
            carry += p  # e.g. a leading '...' — prefix it to the next sentence
            continue
        out.append(carry + p)
        carry = ''
    if carry:
        if out:
            out[-1] += carry
        else:
            out.append(carry)
    return out


def split_segment(seg: DialogueLine) -> list[DialogueLine]:
    """Split one segment into sentence pieces with proportional timing.

    Timing is allocated proportionally to piece length and clamped so every
    piece satisfies `start <= end <= seg.end_ms` even under rounding
    accumulation on degenerate (many-sentences, tiny-span) segments.
    """
    pieces = _pieces(seg.text)
    if len(pieces) <= 1:
        return [seg]
    total = sum(len(p) for p in pieces)
    span = seg.end_ms - seg.start_ms
    out: list[DialogueLine] = []
    cursor = seg.start_ms
    for i, piece in enumerate(pieces):
        if i == len(pieces) - 1:
            end = seg.end_ms
        else:
            end = min(seg.end_ms, max(cursor, cursor + round(span * len(piece) / total)))
        out.append(DialogueLine(cursor, end, piece))
        cursor = end
    return out


def split_segments(segs: list[DialogueLine]) -> list[DialogueLine]:
    """Split every segment; order preserved."""
    out: list[DialogueLine] = []
    for seg in segs:
        out.extend(split_segment(seg))
    return out
