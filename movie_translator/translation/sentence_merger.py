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
- || groups are capped at MAX_BATCH_SENTENCES (3) — larger groups cause
  the model to enter repetition loops and corrupt separators
- Pipe characters (| ||) are stripped from output to prevent leaking
- Proportional word-count splitting redistributes merged output to line timings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Maximum sentences in a single ||-separated group.
# Beyond this the BiDi model loses separators and enters repetition loops.
MAX_BATCH_SENTENCES = 3

# Maximum total words in a single ||-separated group.
# Even with <=3 sentences, long combined inputs degrade model quality.
MAX_BATCH_WORDS = 60

# Lines with this many words or fewer get their own solo group.
# Very short lines ("Huh?", "Right.") are dropped or garbled when
# batched via || with longer sentences.
SHORT_LINE_MAX_WORDS = 3

# Terminal punctuation: . ! ? optionally followed by closing quotes/parens
_TERMINAL_RE = re.compile(r'[.!?]["\')»\]]*\s*$')

# Ellipsis at end of line — this is a continuation, NOT terminal
_ELLIPSIS_RE = re.compile(r'\.{2,}["\')»\]]*\s*$')

# Speaker dash at start of line
_SPEAKER_DASH_RE = re.compile(r'^[\-\u2014\u2013]\s*\S')

# Pipe characters that may leak from || separator
_PIPE_RE = re.compile(r'\s*\|+\s*')


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
                # If the previous line ended with ellipsis and this line
                # starts with a capital letter, it's a new sentence despite
                # the ellipsis — stop merging before it.
                prev_text = texts[group.line_indices[-1]].strip()
                curr_text = texts[i].strip()
                if _ELLIPSIS_RE.search(prev_text) and curr_text and curr_text[0].isupper():
                    break
                group.line_indices.append(i)
                if is_sentence_end(texts[i]):
                    i += 1
                    break
                i += 1
            groups.append(group)
            continue

        # Very short complete sentences get their own group — the model
        # drops or garbles them when batched via || with longer lines.
        if len(texts[i].split()) <= SHORT_LINE_MAX_WORDS:
            groups.append(TranslationGroup(line_indices=[i], is_fragment_merge=False))
            i += 1
            continue

        # Current line is a complete sentence — batch consecutive complete
        # sentences into one group (they'll be ||‑separated),
        # capped at MAX_BATCH_SENTENCES and MAX_BATCH_WORDS.
        group = TranslationGroup(line_indices=[i], is_fragment_merge=False)
        current_words = len(texts[i].split())
        i += 1
        while i < n and len(group.line_indices) < MAX_BATCH_SENTENCES:
            if is_speaker_line(texts[i]):
                break
            if not is_sentence_end(texts[i]):
                break
            next_words = len(texts[i].split())
            if current_words + next_words > MAX_BATCH_WORDS:
                break
            # Don't pull short lines into a batch with longer ones
            if next_words <= SHORT_LINE_MAX_WORDS:
                break
            group.line_indices.append(i)
            current_words += next_words
            i += 1
        groups.append(group)

    return groups


# Leading/trailing ellipsis used as continuation markers in subtitle fragments:
#   "In that matter..."  +  "...my unit, the White Dragon Knights,"
# These should be stripped when joining, producing clean input for the model:
#   "In that matter, my unit, the White Dragon Knights,"
_LEADING_ELLIPSIS_RE = re.compile(r'^\.{2,}\s*')
_TRAILING_ELLIPSIS_RE = re.compile(r'\s*\.{2,}$')


def build_input(texts: list[str], group: TranslationGroup) -> str:
    """Build translation input string for a single group.

    Fragment-merged lines are space-joined.  Independent sentences are
    joined with `` || ``.

    For fragment merges, continuation ellipses (``...`` at end of one line
    and ``...`` at start of the next) are stripped so the model sees one
    clean sentence instead of ``"matter... ...my unit"``.
    """
    lines = [texts[idx] for idx in group.line_indices]
    if group.is_fragment_merge:
        cleaned = []
        for i, line in enumerate(lines):
            text = line
            # Strip trailing ellipsis if the next line starts with one
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if _TRAILING_ELLIPSIS_RE.search(text) and next_line.startswith('.'):
                    text = _TRAILING_ELLIPSIS_RE.sub('', text)
            # Strip leading ellipsis if the previous line ended with one
            if i > 0:
                prev_line = lines[i - 1].strip()
                if _LEADING_ELLIPSIS_RE.match(text) and prev_line.endswith('.'):
                    text = _LEADING_ELLIPSIS_RE.sub('', text)
            cleaned.append(text.strip())
        return ' '.join(cleaned)
    return ' || '.join(lines)


def _strip_pipes(text: str) -> str:
    """Remove orphaned | and || tokens from text."""
    cleaned = _PIPE_RE.sub(' ', text)
    return cleaned.strip()


def split_output(translated: str, group: TranslationGroup, original_texts: list[str]) -> list[str]:
    """Split translated text back to the original line count for *group*.

    For independent-sentence groups, splits on ``||``.  For fragment-merged
    groups, uses proportional word-count splitting.  Pipe characters are
    stripped from the final output to prevent leaking.
    """
    n_lines = len(group.line_indices)
    if n_lines == 1:
        return [_strip_pipes(translated.strip())]

    if not group.is_fragment_merge:
        # Split on ||
        parts = [p.strip() for p in translated.split('||')]
        # If the separator was lost, fall back to proportional split
        if len(parts) == n_lines:
            return [_strip_pipes(p) for p in parts]
        # Fallback: proportional word-count split
        return _proportional_split(translated, group, original_texts)

    return _proportional_split(translated, group, original_texts)


def _proportional_split(
    translated: str,
    group: TranslationGroup,
    original_texts: list[str],
) -> list[str]:
    """Redistribute *translated* words proportionally to original word counts."""
    # Strip pipes BEFORE splitting into words
    cleaned = _strip_pipes(translated)
    words = cleaned.split()
    total_translated = len(words)
    if total_translated == 0:
        return [''] * len(group.line_indices)

    orig_counts = [max(len(original_texts[idx].split()), 1) for idx in group.line_indices]
    total_orig = sum(orig_counts)

    result: list[str] = []
    used = 0
    remaining_segments = len(orig_counts)
    for _k, count in enumerate(orig_counts):
        remaining_segments -= 1
        if remaining_segments == 0:
            # Last segment gets everything remaining
            result.append(' '.join(words[used:]))
        else:
            remaining_words = total_translated - used
            share = round(count / total_orig * total_translated)
            # Ensure at least 1 word if available, but leave enough for
            # remaining segments (each needs at least 0 words — we only
            # guarantee non-empty for the last segment).
            share = max(share, min(1, remaining_words))
            share = min(share, max(0, remaining_words - remaining_segments))
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
