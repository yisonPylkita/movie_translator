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


def split_segment(
    seg: DialogueLine,
    boundaries: list[int] | None = None,
) -> list[DialogueLine]:
    """Split one segment into sentence pieces with timing from boundaries.

    When ``boundaries`` (VAD-detected pause timestamps in ms within this
    segment's time span) are provided, each piece's end time is snapped to
    the nearest boundary.  This gives natural pause-aligned cuts instead of
    proportional-length guesses.  Falls back to proportional timing when no
    boundaries are given or none fall inside the segment's range.
    """
    pieces = _pieces(seg.text)
    if len(pieces) <= 1:
        return [seg]

    # Narrow to boundaries that actually fall inside this segment.
    inner = [b for b in (boundaries or []) if seg.start_ms < b < seg.end_ms]
    # Cap the number of splits to the number of sentence pieces.
    if len(inner) >= len(pieces):
        inner = inner[: len(pieces) - 1]

    if not inner:
        # Fall back to purely proportional timing.
        return _proportional_split(seg, pieces)

    return _boundary_split(seg, pieces, inner)


def _proportional_split(seg: DialogueLine, pieces: list[str]) -> list[DialogueLine]:
    """Split proportionally by character length (original heuristic)."""
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


def _boundary_split(
    seg: DialogueLine,
    pieces: list[str],
    boundaries: list[int],
) -> list[DialogueLine]:
    """Split pieces across VAD pause boundaries.

    Groups sentence pieces into chunks and snaps each chunk's end time to
    the corresponding VAD pause boundary. When more pieces than boundaries,
    the first ``N`` pieces share the first boundary; the last ``M`` pieces
    share the last boundary; middle pieces get their own boundary.

    This is a practical heuristic: VAD catches natural pause points, but
    there may be more sentence boundaries than silence gaps (when the
    speaker doesn't pause between sentences).
    """
    num_boundaries = len(boundaries)
    num_pieces = len(pieces)

    # Distribute pieces across the available boundaries.
    # ─── Strategy ──────────────────────────────────────────────────────
    #   pieces:  [A] [B] [C] [D] [E]      boundaries: [t1] [t2]
    #   Groups:  {A, B, C}  |  {D}  |  {E}
    #              ^-- t1       ^-- t2     ^-- seg.end_ms
    # ────────────────────────────────────────────────────────────────────
    # Group assignments: first pieces share first boundary, middle gets own,
    # last pieces share last boundary.
    if num_pieces <= num_boundaries:
        # Each piece gets its own boundary (or share for extras).
        return _one_per_piece(seg, pieces, boundaries)

    # More pieces than boundaries: distribute.
    # pieces per boundary, distributing the remainder.
    base = num_pieces // (num_boundaries + 1)
    rem = num_pieces % (num_boundaries + 1)

    groups: list[list[str]] = []
    idx = 0
    for b in range(num_boundaries + 1):
        count = base + (1 if b < rem else 0)
        groups.append(pieces[idx : idx + count])
        idx += count

    out: list[DialogueLine] = []
    cursor = seg.start_ms
    for i, group in enumerate(groups):
        text = ' '.join(group).strip()
        if not text:
            continue
        if i < len(boundaries):
            end = boundaries[i]
        else:
            end = seg.end_ms
        out.append(DialogueLine(cursor, end, text))
        cursor = end
    return out


def _one_per_piece(
    seg: DialogueLine,
    pieces: list[str],
    boundaries: list[int],
) -> list[DialogueLine]:
    """Each piece gets one boundary, with extras sharing the last."""
    out: list[DialogueLine] = []
    cursor = seg.start_ms
    for i, piece in enumerate(pieces):
        if i < len(boundaries):
            end = boundaries[i]
        else:
            end = seg.end_ms
        out.append(DialogueLine(cursor, end, piece))
        cursor = end
    return out


def split_segments(
    segs: list[DialogueLine],
    boundaries: list[int] | None = None,
) -> list[DialogueLine]:
    """Split every segment; order preserved.

    When ``boundaries`` (VAD-detected pause timestamps) are provided, they
    are used across all segments — each segment picks only the boundaries
    that fall within its time span.
    """
    out: list[DialogueLine] = []
    for seg in segs:
        out.extend(split_segment(seg, boundaries))
    return out
