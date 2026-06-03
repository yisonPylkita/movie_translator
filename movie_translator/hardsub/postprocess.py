"""Clean raw burned-in OCR output into a usable .srt.

The shared :func:`movie_translator.ocr.extract_burned_in_subtitles` only merges
*exactly-equal* consecutive frames, so OCR jitter on a static subtitle ("był" /
"byl" / "byt") becomes several near-duplicate lines, and transient garbage from
the OP/ED karaoke or on-screen signs leaks through. This module post-processes
the per-frame OCR text (text + timestamp) into clean lines:

  * fuzzy-merge consecutive frames whose text is *similar* into one timed block,
    keeping the best (most complete) variant;
  * drop non-dialogue noise by content (alpha ratio / min letters) and by
    persistence (a real line spans a minimum duration; single-frame OCR flicker
    does not) — this is anime-agnostic, so it handles OP/ED/sign junk wherever
    it appears without hardcoding per-series timings.

Graduated verbatim from the `scripts/hardsub_poc/postprocess.py` PoC (logic
unchanged) so the main pipeline's OCR stage stays untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

# Tuning defaults. Conservative: keep a line only if it looks like text AND
# persisted long enough to be a real subtitle rather than OCR flicker.
SIMILARITY = 0.80  # >= this ratio (normalized) => same subtitle, merge
TAIL_MS = 800  # how long the last line lingers past its last frame
MIN_DURATION_MS = 500  # drop lines shorter than this (transient garbage)
MIN_LETTERS = 3  # drop lines with fewer real letters
MIN_ALPHA_RATIO = 0.5  # drop lines that are mostly symbols/punctuation


@dataclass(frozen=True)
class CleanLine:
    start_ms: int
    end_ms: int
    text: str


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace — for similarity comparison only."""
    return ' '.join(text.lower().split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def is_dialogue(text: str) -> bool:
    """Heuristic: does this OCR text look like real dialogue (not garbage)?

    Drops the karaoke/logo gibberish (e.g. ``‡.|.tz``, ``# Baá``) by requiring
    a minimum number of real letters and a letter-to-symbol ratio — content,
    not position, so it works for any anime's OP/ED.
    """
    flat = text.replace('\n', ' ').strip()
    if len(flat) < 2:
        return False
    letters = sum(c.isalpha() for c in flat)
    non_space = sum(not c.isspace() for c in flat)
    if letters < MIN_LETTERS:
        return False
    return not (non_space and letters / non_space < MIN_ALPHA_RATIO)


def _best_variant(variants: list[str]) -> str:
    """Pick the canonical text for a merged group: most frequent, then longest."""
    counts = Counter(variants)
    top = counts.most_common(1)[0][1]
    tied = [v for v, c in counts.items() if c == top]
    return max(tied, key=len)


def merge_ocr_results(
    frame_texts: list[tuple[int, str]],
    *,
    similarity: float = SIMILARITY,
    tail_ms: int = TAIL_MS,
    min_duration_ms: int = MIN_DURATION_MS,
) -> list[CleanLine]:
    """Merge per-frame ``(timestamp_ms, text)`` into clean, deduped lines.

    Consecutive frames whose text is >= ``similarity`` to the group's first
    text are folded into one block (start of first .. start of the frame that
    breaks the group). Blank frames close the current block. Groups shorter
    than ``min_duration_ms`` or failing :func:`is_dialogue` are dropped.
    """
    frames = sorted(frame_texts, key=lambda f: f[0])
    lines: list[CleanLine] = []

    anchor: str | None = None
    variants: list[str] = []
    start_ms = 0

    def close(end_ms: int) -> None:
        if anchor is None:
            return
        text = _best_variant(variants)
        if end_ms - start_ms >= min_duration_ms and is_dialogue(text):
            lines.append(CleanLine(start_ms, end_ms, text))

    for ts, raw in frames:
        text = (raw or '').strip()
        if anchor is not None and text and _similar(text, anchor) >= similarity:
            variants.append(text)  # same subtitle, still showing
            continue
        # Boundary: close the running group, ending at this frame's start.
        close(ts)
        if text:
            anchor, variants, start_ms = text, [text], ts
        else:
            anchor, variants = None, []

    if anchor is not None:
        close((frames[-1][0] if frames else start_ms) + tail_ms)
    return lines


def _fmt_ts(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def to_srt(lines: list[CleanLine]) -> str:
    """Render clean lines as SRT text."""
    blocks = [
        f'{i}\n{_fmt_ts(ln.start_ms)} --> {_fmt_ts(ln.end_ms)}\n{ln.text}\n'
        for i, ln in enumerate(lines, 1)
    ]
    return '\n'.join(blocks)
