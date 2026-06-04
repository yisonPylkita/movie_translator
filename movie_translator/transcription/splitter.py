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


def split_segment(seg: DialogueLine) -> list[DialogueLine]:
    """Split one segment into sentence pieces with proportional timing."""
    pieces = [p.strip() for p in _SENTENCE.findall(seg.text) if p.strip()]
    if len(pieces) <= 1:
        return [seg]
    total = sum(len(p) for p in pieces)
    span = seg.end_ms - seg.start_ms
    out: list[DialogueLine] = []
    cursor = seg.start_ms
    for i, piece in enumerate(pieces):
        end = seg.end_ms if i == len(pieces) - 1 else cursor + round(span * len(piece) / total)
        out.append(DialogueLine(cursor, end, piece))
        cursor = end
    return out


def split_segments(segs: list[DialogueLine]) -> list[DialogueLine]:
    """Split every segment; order preserved."""
    out: list[DialogueLine] = []
    for seg in segs:
        out.extend(split_segment(seg))
    return out
