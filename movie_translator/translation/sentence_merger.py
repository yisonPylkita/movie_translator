"""Sentence-level grouping for seq2seq subtitle translation.

The BiDi (allegro/BiDi-eng-pol) model is a seq2seq translator that processes
its entire input as one unit. Subtitles frequently split sentences across
multiple timed lines, causing the model to produce wrong verb conjugation,
stuttering, and mistranslations when lines are translated independently.

This module groups consecutive subtitle lines into complete sentences before
translation, then splits the translated output back to match original line
boundaries. Independent sentences are separated with ' || ' (double-pipe),
which the BiDi model passes through at 93%+ fidelity, allowing reliable
output splitting.

Key rules (experimentally verified):
- Lines without terminal punctuation (.!?) are fragments → merge with next
- Ellipsis (...) is a continuation marker → merge
- Speaker dash lines (- text) are NEVER merged — model drops one speaker
- The || separator preserves sentence boundaries through translation
- Proportional word-count splitting redistributes merged output to line timings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Terminal punctuation: . ! ? optionally followed by closing quotes/parens
_TERMINAL_RE = re.compile(r'[.!?]["\')»\]]*\s*$')

# Ellipsis at end of line — this is a continuation, NOT terminal
_ELLIPSIS_RE = re.compile(r'\.{2,}["\')»\]]*\s*$')

# Speaker dash at start of line
_SPEAKER_DASH_RE = re.compile(r'^[\-\u2014\u2013]\s*\S')


def is_sentence_end(text: str) -> bool:
    """Return True if *text* ends a complete sentence.

    Terminal: . ! ? (optionally followed by closing quotes/parens).
    Non-terminal: comma, colon, semicolon, ellipsis, no punctuation,
    empty/whitespace.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Ellipsis is explicitly non-terminal (continuation)
    if _ELLIPSIS_RE.search(stripped):
        return False
    return bool(_TERMINAL_RE.search(stripped))


def is_speaker_line(text: str) -> bool:
    """Return True if *text* starts with a dialogue dash (-, \u2014, \u2013)."""
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_SPEAKER_DASH_RE.match(stripped))


@dataclass
class TranslationGroup:
    """A group of consecutive subtitle lines to be translated together.

    Attributes:
        line_indices: indices into the original texts list.
        is_fragment_merge: True when lines were merged because they form
            fragments of one sentence (space-joined). False when lines are
            independent complete sentences (joined with ``||``).
    """

    line_indices: list[int] = field(default_factory=list)
    is_fragment_merge: bool = False


def group_lines(texts: list[str]) -> list[TranslationGroup]:
    """Group consecutive subtitle lines into translation units.

    Speaker lines always get their own group.  Fragment lines (no terminal
    punctuation) merge with subsequent lines until a sentence end is found.
    Consecutive complete sentences batch into a single group (joined later
    with ``||``).
    """
    if not texts:
        return []

    groups: list[TranslationGroup] = []
    i = 0
    n = len(texts)

    while i < n:
        # Speaker lines are always solo
        if is_speaker_line(texts[i]):
            groups.append(TranslationGroup(line_indices=[i], is_fragment_merge=False))
            i += 1
            continue

        # Check if current line is a fragment (not a sentence end)
        if not is_sentence_end(texts[i]):
            # Start a fragment merge group
            group = TranslationGroup(line_indices=[i], is_fragment_merge=True)
            i += 1
            # Keep merging until we hit a sentence end or run out of lines
            while i < n:
                if is_speaker_line(texts[i]):
                    # Speaker line breaks the merge — stop before it
                    break
                group.line_indices.append(i)
                if is_sentence_end(texts[i]):
                    i += 1
                    break
                i += 1
            groups.append(group)
            continue

        # Current line is a complete sentence — batch consecutive complete
        # sentences into one group (they'll be ||‑separated)
        group = TranslationGroup(line_indices=[i], is_fragment_merge=False)
        i += 1
        while i < n:
            if is_speaker_line(texts[i]):
                break
            if not is_sentence_end(texts[i]):
                break
            group.line_indices.append(i)
            i += 1
        groups.append(group)

    return groups


def build_input(texts: list[str], group: TranslationGroup) -> str:
    """Build translation input string for a single group.

    Fragment-merged lines are space-joined.  Independent sentences are
    joined with `` || ``.
    """
    lines = [texts[idx] for idx in group.line_indices]
    if group.is_fragment_merge:
        return ' '.join(lines)
    return ' || '.join(lines)


def split_output(translated: str, group: TranslationGroup, original_texts: list[str]) -> list[str]:
    """Split translated text back to the original line count for *group*.

    For independent-sentence groups, splits on ``||``.  For fragment-merged
    groups, uses proportional word-count splitting.
    """
    n_lines = len(group.line_indices)
    if n_lines == 1:
        return [translated.strip()]

    if not group.is_fragment_merge:
        # Split on ||
        parts = [p.strip() for p in translated.split('||')]
        # If the separator was lost, fall back to proportional split
        if len(parts) == n_lines:
            return parts
        # Fallback: proportional word-count split
        return _proportional_split(translated, group, original_texts)

    return _proportional_split(translated, group, original_texts)


def _proportional_split(
    translated: str,
    group: TranslationGroup,
    original_texts: list[str],
) -> list[str]:
    """Redistribute *translated* words proportionally to original word counts."""
    words = translated.split()
    total_translated = len(words)
    if total_translated == 0:
        return [''] * len(group.line_indices)

    orig_counts = [max(len(original_texts[idx].split()), 1) for idx in group.line_indices]
    total_orig = sum(orig_counts)

    result: list[str] = []
    used = 0
    for k, count in enumerate(orig_counts):
        if k == len(orig_counts) - 1:
            # Last segment gets everything remaining
            result.append(' '.join(words[used:]))
        else:
            share = round(count / total_orig * total_translated)
            share = max(share, 1)  # at least one word per segment
            share = min(share, total_translated - used - (len(orig_counts) - 1 - k))
            result.append(' '.join(words[used : used + share]))
            used += share

    return result


# ------------------------------------------------------------------
# Top-level convenience API
# ------------------------------------------------------------------


def merge_for_translation(texts: list[str]) -> tuple[list[str], list[TranslationGroup]]:
    """Group *texts* and build merged translation inputs.

    Returns ``(merged_texts, groups)`` where each element of
    *merged_texts* corresponds to one :class:`TranslationGroup`.
    """
    groups = group_lines(texts)
    merged = [build_input(texts, g) for g in groups]
    return merged, groups


def unmerge_translations(
    translated_texts: list[str],
    groups: list[TranslationGroup],
    original_texts: list[str],
) -> list[str]:
    """Split *translated_texts* back to match the original line count.

    Returns a flat list aligned 1-to-1 with *original_texts*.
    """
    result = [''] * len(original_texts)
    for translated, group in zip(translated_texts, groups, strict=True):
        parts = split_output(translated, group, original_texts)
        for idx, part in zip(group.line_indices, parts, strict=True):
            result[idx] = part
    return result
