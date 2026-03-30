# Sentence-Level Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve translation quality by merging subtitle fragments into complete sentences before translation, then splitting the output back — giving the BiDi seq2seq model natural sentence-level context without performance cost.

**Architecture:** A new `sentence_merger` module handles grouping consecutive subtitle lines into translation units based on punctuation heuristics. Fragment lines (no terminal punctuation) are space-joined into single sentences. Independent sentences within the same batch are joined with ` || ` (double-pipe separator, experimentally verified to pass through the BiDi model at 93%+ fidelity). After translation, output is split on `||` and fragments are proportionally redistributed to original line boundaries by word count.

**Tech Stack:** Python, regex for punctuation detection, existing SubtitleTranslator for model interaction.

---

## Background: Why This Exists

The `allegro/BiDi-eng-pol` model is a seq2seq (Marian-based) translation model. It translates
its entire input as a single unit — it cannot be instructed to "translate only part of the input."

Subtitles frequently split sentences across multiple timed lines:
```
Line 1 (0:00-0:03): "The Holy Britannian Empire"     ← no verb, no punctuation
Line 2 (0:03-0:06): "declared war on Japan."          ← verb conjugation depends on subject
```

When translated independently, the model lacks context:
- "The Holy Britannian Empire" → "Imperium Imperium" (stuttering bug on fragments)
- "declared war on Japan." → "wypowiedział wojnę Japonii." (masculine singular — wrong for neuter "Imperium")

When merged: "The Holy Britannian Empire declared war on Japan." → "Imperium wypowiedziało wojnę Japonii." (correct neuter conjugation, no stutter)

### Key experimental findings (from `/tmp/bidi_findings.md`):
1. **Fragment merging dramatically improves quality** — fixes verb conjugation, eliminates stuttering
2. **The `||` double-pipe separator preserves sentence boundaries** at 93%+ through translation
3. **Speaker dash lines (`- text`) must NEVER be merged** — the model drops one speaker entirely
4. **Plain concatenation of independent sentences is dangerous** — model merges/drops content
5. **Proportional word-count splitting** works for redistributing merged translations to line timings

---

## File Structure

| File | Responsibility |
|------|---------------|
| `movie_translator/translation/sentence_merger.py` (CREATE) | Grouping, merging, splitting logic. Pure functions, no model dependency. |
| `movie_translator/translation/tests/test_sentence_merger.py` (CREATE) | Tests for all grouping/splitting edge cases |
| `movie_translator/translation/translator.py` (MODIFY) | Integration: call merger before/after `translate_texts()` |

---

## Task 1: Sentence Merger — Grouping Logic

**Files:**
- Create: `movie_translator/translation/sentence_merger.py`
- Test: `movie_translator/translation/tests/test_sentence_merger.py`

- [ ] **Step 1: Write the failing tests for `is_sentence_end`**

```python
# movie_translator/translation/tests/test_sentence_merger.py
import pytest

from movie_translator.translation.sentence_merger import is_sentence_end


class TestIsSentenceEnd:
    """Detect whether a subtitle line ends a complete sentence."""

    def test_period_is_terminal(self):
        assert is_sentence_end('The war began.') is True

    def test_exclamation_is_terminal(self):
        assert is_sentence_end('Stop!') is True

    def test_question_is_terminal(self):
        assert is_sentence_end('What happened?') is True

    def test_quoted_punctuation_is_terminal(self):
        assert is_sentence_end('"I will destroy Britannia!"') is True
        assert is_sentence_end("'Really?'") is True

    def test_ellipsis_is_not_terminal(self):
        assert is_sentence_end('I thought...') is False

    def test_comma_is_not_terminal(self):
        assert is_sentence_end('In the year 2010,') is False

    def test_colon_is_not_terminal(self):
        assert is_sentence_end('He said:') is False

    def test_no_punctuation_is_not_terminal(self):
        assert is_sentence_end('The Holy Britannian Empire') is False

    def test_dash_prefix_with_period_is_terminal(self):
        assert is_sentence_end("- I'll go.") is True

    def test_empty_string(self):
        assert is_sentence_end('') is True

    def test_whitespace_only(self):
        assert is_sentence_end('   ') is True

    def test_closing_paren_after_period(self):
        assert is_sentence_end('(the war began.)') is True

    def test_semicolon_is_not_terminal(self):
        assert is_sentence_end('he ran; she followed') is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestIsSentenceEnd -v`
Expected: FAIL — `ImportError: cannot import name 'is_sentence_end'`

- [ ] **Step 3: Implement `is_sentence_end`**

```python
# movie_translator/translation/sentence_merger.py
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

_TERMINAL_RE = re.compile(r'[.!?]["\')»\]]*\s*$')


def is_sentence_end(text: str) -> bool:
    """Return True if the line ends a complete sentence.

    Terminal punctuation: . ! ?  (optionally followed by closing quotes/parens)
    Non-terminal: comma, colon, semicolon, ellipsis, no punctuation, empty
    """
    stripped = text.strip()
    if not stripped:
        return True
    # Ellipsis is a continuation, not a sentence end
    if stripped.rstrip('"\')»] ').endswith('...'):
        return False
    return bool(_TERMINAL_RE.search(stripped))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestIsSentenceEnd -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/translation/sentence_merger.py movie_translator/translation/tests/test_sentence_merger.py
git commit -m "feat(translation): add is_sentence_end for subtitle line grouping"
```

---

## Task 2: Sentence Merger — `is_speaker_line` and `group_lines`

**Files:**
- Modify: `movie_translator/translation/sentence_merger.py`
- Modify: `movie_translator/translation/tests/test_sentence_merger.py`

- [ ] **Step 1: Write the failing tests for `is_speaker_line`**

Append to `test_sentence_merger.py`:

```python
from movie_translator.translation.sentence_merger import is_speaker_line


class TestIsSpeakerLine:
    """Speaker dash lines must never be merged."""

    def test_dash_space_is_speaker(self):
        assert is_speaker_line("- I'll go.") is True

    def test_dash_no_space_is_speaker(self):
        assert is_speaker_line("-Don't move!") is True

    def test_no_dash_is_not_speaker(self):
        assert is_speaker_line('The Empire fell.') is False

    def test_dash_mid_text_is_not_speaker(self):
        assert is_speaker_line('well-known fact.') is False

    def test_em_dash_is_speaker(self):
        assert is_speaker_line('— I understand.') is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestIsSpeakerLine -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `is_speaker_line`**

Add to `sentence_merger.py`:

```python
def is_speaker_line(text: str) -> bool:
    """Return True if the line starts with a dialogue dash (speaker indicator)."""
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in '-\u2014\u2013'  # dash, em-dash, en-dash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestIsSpeakerLine -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `group_lines`**

Append to `test_sentence_merger.py`:

```python
from movie_translator.translation.sentence_merger import group_lines, TranslationGroup


class TestGroupLines:
    """Group consecutive lines into translation units."""

    def test_single_complete_sentence(self):
        groups = group_lines(['Hello world.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0]
        assert groups[0].is_fragment_merge is False

    def test_two_independent_sentences(self):
        groups = group_lines(['Hello.', 'Goodbye.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1]
        assert groups[0].is_fragment_merge is False

    def test_fragment_merge(self):
        groups = group_lines(['The Empire', 'declared war.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1]
        assert groups[0].is_fragment_merge is True

    def test_three_line_fragment(self):
        groups = group_lines(['In the decisive battle', 'for the mainland,', 'Britannian forces won.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1, 2]
        assert groups[0].is_fragment_merge is True

    def test_mixed_fragment_and_independent(self):
        groups = group_lines(['The Empire', 'declared war.', 'Japan fell.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1, 2]
        # Lines 0+1 are a fragment merge, line 2 is independent — but all in one group

    def test_speaker_lines_stay_individual(self):
        groups = group_lines(["- I'll go.", '- No, stay!'])
        assert len(groups) == 2
        assert groups[0].line_indices == [0]
        assert groups[1].line_indices == [1]

    def test_speaker_line_breaks_group(self):
        groups = group_lines(['He ran.', "- Stop!", 'She waited.'])
        assert len(groups) == 3

    def test_fragment_before_speaker_line(self):
        groups = group_lines(['The Empire', "- Stop!"])
        assert len(groups) == 2
        assert groups[0].line_indices == [0]
        assert groups[1].line_indices == [1]

    def test_empty_input(self):
        groups = group_lines([])
        assert groups == []

    def test_ellipsis_merges_with_next(self):
        groups = group_lines(['I thought...', 'we could win.'])
        assert len(groups) == 1
        assert groups[0].is_fragment_merge is True

    def test_comma_merges_with_next(self):
        groups = group_lines(['In 2010,', 'the war began.'])
        assert len(groups) == 1
        assert groups[0].is_fragment_merge is True
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestGroupLines -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement `group_lines`**

Add to `sentence_merger.py`:

```python
from dataclasses import dataclass


@dataclass
class TranslationGroup:
    """A group of consecutive subtitle lines to translate as one unit.

    Attributes:
        line_indices: Indices into the original text list.
        is_fragment_merge: True if the group contains lines merged into a
            single sentence (no terminal punctuation between them). False if
            all lines are complete sentences joined with ' || '.
    """

    line_indices: list[int]
    is_fragment_merge: bool


def group_lines(texts: list[str]) -> list[TranslationGroup]:
    """Group consecutive subtitle lines into translation units.

    Rules:
    - Speaker dash lines always get their own group (model drops content
      when dash lines are merged).
    - Lines without terminal punctuation are fragments — merge with
      subsequent lines until a sentence end is found.
    - Complete sentences are batched into a single group joined with ' || '
      until a speaker line or fragment boundary is encountered.
    """
    if not texts:
        return []

    groups: list[TranslationGroup] = []
    current_indices: list[int] = []
    has_fragment = False

    for i, text in enumerate(texts):
        # Speaker lines always break the current group and stand alone
        if is_speaker_line(text):
            if current_indices:
                groups.append(TranslationGroup(current_indices, has_fragment))
            groups.append(TranslationGroup([i], is_fragment_merge=False))
            current_indices = []
            has_fragment = False
            continue

        # If next line is a speaker line, current fragments must end here
        next_is_speaker = (i + 1 < len(texts)) and is_speaker_line(texts[i + 1])

        current_indices.append(i)

        if is_sentence_end(text) or next_is_speaker:
            # This line completes a sentence (or must break before speaker)
            if not is_sentence_end(text):
                has_fragment = True
            # Continue accumulating complete sentences into the same group
            # unless the next line is a speaker or we're at the end
            if next_is_speaker or i == len(texts) - 1:
                groups.append(TranslationGroup(current_indices, has_fragment))
                current_indices = []
                has_fragment = False
        else:
            # Fragment — will merge with next line
            has_fragment = True

    # Flush remaining
    if current_indices:
        groups.append(TranslationGroup(current_indices, has_fragment))

    return groups
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestGroupLines -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add movie_translator/translation/sentence_merger.py movie_translator/translation/tests/test_sentence_merger.py
git commit -m "feat(translation): add group_lines for sentence-level translation units"
```

---

## Task 3: Sentence Merger — `build_input` and `split_output`

**Files:**
- Modify: `movie_translator/translation/sentence_merger.py`
- Modify: `movie_translator/translation/tests/test_sentence_merger.py`

- [ ] **Step 1: Write the failing tests for `build_input`**

Append to `test_sentence_merger.py`:

```python
from movie_translator.translation.sentence_merger import build_input


class TestBuildInput:
    """Build translation input string from a group of lines."""

    def test_single_line(self):
        result = build_input(['Hello world.'])
        assert result == 'Hello world.'

    def test_fragment_merge_joins_with_space(self):
        result = build_input(['The Empire', 'declared war.'])
        assert result == 'The Empire declared war.'

    def test_independent_sentences_join_with_double_pipe(self):
        result = build_input(['He ran.', 'She followed.'])
        assert result == 'He ran. || She followed.'

    def test_mixed_fragment_then_independent(self):
        result = build_input(['The Empire', 'declared war.', 'Japan fell.'])
        assert result == 'The Empire declared war. || Japan fell.'

    def test_three_independent(self):
        result = build_input(['One.', 'Two.', 'Three.'])
        assert result == 'One. || Two. || Three.'

    def test_fragment_at_end(self):
        # Fragment at end has no sentence-ender; just space-join
        result = build_input(['He said:', 'hello world'])
        assert result == 'He said: hello world'

    def test_ellipsis_continuation(self):
        result = build_input(['I thought...', 'we could win.'])
        assert result == 'I thought... we could win.'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestBuildInput -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `build_input`**

Add to `sentence_merger.py`:

```python
SEPARATOR = ' || '


def build_input(texts: list[str]) -> str:
    """Build a single translation input string from grouped lines.

    Fragment lines (no terminal punctuation) are space-joined into a sentence.
    Complete sentences are joined with ' || ' to preserve boundaries through
    translation — the BiDi model passes '||' through at 93%+ fidelity.
    """
    if len(texts) == 1:
        return texts[0]

    # Walk through texts, space-joining fragments and || -joining complete sentences
    parts: list[str] = []
    current_fragment: list[str] = []

    for text in texts:
        current_fragment.append(text)
        if is_sentence_end(text):
            parts.append(' '.join(current_fragment))
            current_fragment = []

    # Flush any trailing fragment (line without terminal punctuation at end)
    if current_fragment:
        parts.append(' '.join(current_fragment))

    return SEPARATOR.join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestBuildInput -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `split_output`**

Append to `test_sentence_merger.py`:

```python
from movie_translator.translation.sentence_merger import split_output


class TestSplitOutput:
    """Split translated output back to match original line count."""

    def test_single_line_passthrough(self):
        result = split_output('Cześć.', original_texts=['Hello.'])
        assert result == ['Cześć.']

    def test_double_pipe_split(self):
        result = split_output(
            'Imperium wypowiedziało wojnę. || Japonia upadła.',
            original_texts=['The Empire declared war.', 'Japan fell.'],
        )
        assert result == ['Imperium wypowiedziało wojnę.', 'Japonia upadła.']

    def test_fragment_proportional_split_two_lines(self):
        result = split_output(
            'Imperium wypowiedziało wojnę Japonii.',
            original_texts=['The Empire', 'declared war on Japan.'],
        )
        assert len(result) == 2
        # Both parts should be non-empty
        assert all(r.strip() for r in result)
        # Rejoined should equal original (minus extra spaces)
        assert ' '.join(result) == 'Imperium wypowiedziało wojnę Japonii.'

    def test_fragment_proportional_split_three_lines(self):
        result = split_output(
            'W decydującej bitwie o kontynent siły Brytanii zwyciężyły.',
            original_texts=['In the decisive battle', 'for the mainland,', 'Britannian forces won.'],
        )
        assert len(result) == 3
        assert all(r.strip() for r in result)

    def test_missing_separator_fallback(self):
        # Model didn't preserve ||, fallback: give full text to each line
        result = split_output(
            'Cześć jak się masz dobrze.',
            original_texts=['Hello.', 'How are you?', 'Fine.'],
        )
        # When || is expected but missing, assign full text to first line,
        # empty strings to the rest — caller handles fallback
        assert len(result) == 3

    def test_double_pipe_with_whitespace(self):
        result = split_output(
            'Tak.  ||  Nie.',
            original_texts=['Yes.', 'No.'],
        )
        assert result == ['Tak.', 'Nie.']

    def test_mixed_group_fragment_then_independent(self):
        # Input was: ["The Empire", "declared war.", "Japan fell."]
        # build_input produced: "The Empire declared war. || Japan fell."
        # Model translated to: "Imperium wypowiedziało wojnę. || Japonia upadła."
        result = split_output(
            'Imperium wypowiedziało wojnę. || Japonia upadła.',
            original_texts=['The Empire', 'declared war.', 'Japan fell.'],
        )
        assert len(result) == 3
        # Lines 0+1 were a fragment merge, line 2 was independent
        # The || split gives us 2 parts: fragment_translation and independent
        # Fragment part gets proportionally split into 2 lines
        assert result[2] == 'Japonia upadła.'
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestSplitOutput -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement `split_output`**

Add to `sentence_merger.py`:

```python
def split_output(translated: str, original_texts: list[str]) -> list[str]:
    """Split translated text back to match the original line count.

    For groups containing independent sentences (joined with ' || '):
      Split on '||' to recover per-sentence translations.
    For merged fragments within a sentence:
      Distribute words proportionally based on original English line lengths.

    Args:
        translated: The translated string (may contain '||' separators).
        original_texts: The original English lines that were grouped.

    Returns:
        List of translated strings, one per original line.
    """
    if len(original_texts) == 1:
        return [translated.strip()]

    # Determine which original lines were fragments vs sentence ends
    # to understand the group structure
    sentence_boundaries = _find_sentence_boundaries(original_texts)
    expected_parts = len(sentence_boundaries)

    # Try splitting on || separator
    if '||' in translated:
        parts = [p.strip() for p in translated.split('||')]
        if len(parts) == expected_parts:
            return _distribute_parts(parts, original_texts, sentence_boundaries)

    # Fallback: || missing or count mismatch — use proportional split
    return _proportional_word_split(translated.strip(), original_texts)


def _find_sentence_boundaries(texts: list[str]) -> list[list[int]]:
    """Group line indices by sentence boundaries.

    Returns a list of index-groups, where each group is the set of
    original line indices that form one sentence.
    E.g., ["The Empire", "declared war.", "Japan fell."] →
          [[0, 1], [2]]  (lines 0+1 are one sentence, line 2 is another)
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i, text in enumerate(texts):
        current.append(i)
        if is_sentence_end(text) or i == len(texts) - 1:
            groups.append(current)
            current = []
    return groups


def _distribute_parts(
    parts: list[str],
    original_texts: list[str],
    sentence_groups: list[list[int]],
) -> list[str]:
    """Assign translated parts to original lines, splitting fragments proportionally."""
    result = [''] * len(original_texts)
    for part, indices in zip(parts, sentence_groups, strict=True):
        if len(indices) == 1:
            result[indices[0]] = part
        else:
            sub_originals = [original_texts[i] for i in indices]
            sub_parts = _proportional_word_split(part, sub_originals)
            for idx, sub_part in zip(indices, sub_parts, strict=True):
                result[idx] = sub_part
    return result


def _proportional_word_split(translated: str, original_texts: list[str]) -> list[str]:
    """Split translated text proportionally by original line character counts."""
    words = translated.split()
    if not words:
        return [translated] + [''] * (len(original_texts) - 1)

    total_chars = sum(len(t) for t in original_texts)
    if total_chars == 0:
        # Equal distribution
        per_line = max(1, len(words) // len(original_texts))
        result = []
        for i in range(len(original_texts)):
            start = i * per_line
            end = start + per_line if i < len(original_texts) - 1 else len(words)
            result.append(' '.join(words[start:end]))
        return result

    # Distribute words proportionally
    ratios = [len(t) / total_chars for t in original_texts]
    result = []
    word_idx = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            # Last line gets all remaining words
            result.append(' '.join(words[word_idx:]))
        else:
            n_words = max(1, round(ratio * len(words)))
            result.append(' '.join(words[word_idx : word_idx + n_words]))
            word_idx = min(word_idx + n_words, len(words))

    return result
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestSplitOutput -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add movie_translator/translation/sentence_merger.py movie_translator/translation/tests/test_sentence_merger.py
git commit -m "feat(translation): add build_input and split_output for sentence merging"
```

---

## Task 4: Sentence Merger — Top-Level `merge_for_translation` and `unmerge_translations`

**Files:**
- Modify: `movie_translator/translation/sentence_merger.py`
- Modify: `movie_translator/translation/tests/test_sentence_merger.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_sentence_merger.py`:

```python
from movie_translator.translation.sentence_merger import merge_for_translation, unmerge_translations


class TestMergeForTranslation:
    """Top-level API: merge lines into translation units."""

    def test_returns_fewer_texts_than_input(self):
        texts = ['The Empire', 'declared war.', 'Japan fell.']
        merged, groups = merge_for_translation(texts)
        assert len(merged) < len(texts)

    def test_preserves_speaker_lines(self):
        texts = ["- I'll go.", '- No!']
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 2
        assert merged[0] == "- I'll go."
        assert merged[1] == '- No!'

    def test_independent_sentences_use_separator(self):
        texts = ['Hello.', 'Goodbye.']
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 1
        assert '||' in merged[0]

    def test_fragments_space_joined(self):
        texts = ['The Empire', 'declared war.']
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 1
        assert merged[0] == 'The Empire declared war.'

    def test_groups_track_original_indices(self):
        texts = ['The Empire', 'declared war.', "- Stop!"]
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 2
        assert groups[0].line_indices == [0, 1]
        assert groups[1].line_indices == [2]


class TestUnmergeTranslations:
    """Top-level API: split translations back to original line count."""

    def test_round_trip_independent(self):
        originals = ['Hello.', 'Goodbye.']
        merged, groups = merge_for_translation(originals)
        # Simulate translation: just return the merged text unchanged
        translated = merged  # In real use, these would be Polish
        result = unmerge_translations(translated, groups, originals)
        assert len(result) == 2

    def test_round_trip_fragments(self):
        originals = ['The Empire', 'declared war.']
        merged, groups = merge_for_translation(originals)
        result = unmerge_translations(
            ['Imperium wypowiedziało wojnę.'], groups, originals
        )
        assert len(result) == 2
        assert ' '.join(result) == 'Imperium wypowiedziało wojnę.'

    def test_round_trip_speaker_lines(self):
        originals = ["- I'll go.", '- No!']
        merged, groups = merge_for_translation(originals)
        result = unmerge_translations(['- Pójdę.', '- Nie!'], groups, originals)
        assert result == ['- Pójdę.', '- Nie!']

    def test_round_trip_preserves_line_count(self):
        originals = ['A.', 'B', 'C.', '- D!', 'E', 'F,', 'G.']
        merged, groups = merge_for_translation(originals)
        # Simulate: each merged text just becomes "PL_N"
        fake_translated = [f'PL_{i}' for i in range(len(merged))]
        result = unmerge_translations(fake_translated, groups, originals)
        assert len(result) == len(originals)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py::TestMergeForTranslation -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `merge_for_translation` and `unmerge_translations`**

Add to `sentence_merger.py`:

```python
def merge_for_translation(
    texts: list[str],
) -> tuple[list[str], list[TranslationGroup]]:
    """Merge subtitle lines into sentence-level translation units.

    Returns:
        merged_texts: Fewer strings to translate (fragments joined, sentences
            separated by '||').
        groups: Mapping from each merged text back to original line indices.
    """
    groups = group_lines(texts)
    merged = []
    for group in groups:
        group_texts = [texts[i] for i in group.line_indices]
        merged.append(build_input(group_texts))
    return merged, groups


def unmerge_translations(
    translated_texts: list[str],
    groups: list[TranslationGroup],
    original_texts: list[str],
) -> list[str]:
    """Split translated texts back to match the original line count.

    Args:
        translated_texts: Translated strings, one per TranslationGroup.
        groups: The groups returned by merge_for_translation().
        original_texts: The original English lines (for proportional splitting).

    Returns:
        List of translated strings with the same length as original_texts.
    """
    result = [''] * len(original_texts)
    for translated, group in zip(translated_texts, groups, strict=True):
        group_originals = [original_texts[i] for i in group.line_indices]
        split = split_output(translated, group_originals)
        for idx, text in zip(group.line_indices, split, strict=True):
            result[idx] = text
    return result
```

- [ ] **Step 4: Run all merger tests to verify they pass**

Run: `uv run pytest movie_translator/translation/tests/test_sentence_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/translation/sentence_merger.py movie_translator/translation/tests/test_sentence_merger.py
git commit -m "feat(translation): add merge_for_translation / unmerge_translations API"
```

---

## Task 5: Integrate Sentence Merger into Translator

**Files:**
- Modify: `movie_translator/translation/translator.py`
- Modify: `movie_translator/translation/tests/test_batch_translation.py`

- [ ] **Step 1: Write integration test**

Add a new test to `test_batch_translation.py`:

```python
class TestSentenceMerging:
    """Verify sentence merging is applied in translate_texts."""

    def test_fragment_lines_merged_before_translation(self):
        """When two lines form one sentence, the model should receive
        a single merged string instead of two separate strings."""
        texts = ['The Empire', 'declared war.']

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=16)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        received_inputs: list[list[str]] = []

        def mock_encode(texts_list, **kwargs):
            received_inputs.append(texts_list)
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            # Return one translation per input
            return ['Imperium wypowiedziało wojnę.'] * len(received_inputs[-1])

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        # Model should have received 1 merged text, not 2 separate texts
        assert len(received_inputs) == 1
        assert len(received_inputs[0]) == 1
        assert 'The Empire declared war.' in received_inputs[0][0]

        # Result should be split back to 2 lines
        assert len(result) == 2

    def test_speaker_lines_not_merged(self):
        """Speaker dash lines must be translated individually."""
        texts = ["- I'll go.", '- No, stay!']

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=16)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        received_inputs: list[list[str]] = []

        def mock_encode(texts_list, **kwargs):
            received_inputs.append(texts_list)
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            return ['- Pójdę.', '- Nie, zostań!'][: len(received_inputs[-1])]

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        # Model should have received 2 separate texts (not merged)
        assert len(received_inputs) == 1
        assert len(received_inputs[0]) == 2
        assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/translation/tests/test_batch_translation.py::TestSentenceMerging -v`
Expected: FAIL — model receives 2 texts instead of 1

- [ ] **Step 3: Integrate merger into `translate_texts`**

In `translator.py`, modify the `translate_texts` method. Add the import at the top:

```python
from .sentence_merger import merge_for_translation, unmerge_translations
```

Then modify `translate_texts`:

```python
    def translate_texts(
        self, texts: list[str], progress_callback: ProgressCallback | None = None
    ) -> list[str]:
        if not texts:
            return []

        # Merge subtitle fragments into sentence-level translation units.
        # This gives the seq2seq model complete sentences instead of fragments,
        # improving verb conjugation, pronoun resolution, and eliminating
        # the model's stuttering bug on short inputs.
        merged_texts, groups = merge_for_translation(texts)
        logger.debug(
            f'Sentence merging: {len(texts)} lines → {len(merged_texts)} translation units'
        )

        translations = []
        total_batches = (len(merged_texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for i in range(0, len(merged_texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_texts = merged_texts[i : i + self.batch_size]

            batch_translations = self._translate_batch(batch_texts)
            translations.extend(batch_translations)

            if progress_callback:
                elapsed = time.time() - start_time
                lines_processed = min(batch_num * self.batch_size, len(merged_texts))
                rate = lines_processed / elapsed if elapsed > 0 else 0
                progress_callback(batch_num, total_batches, rate)

            self._periodic_memory_cleanup(i)

        self._clear_memory()

        if self.enable_enhancements and self.preprocessing_stats.total_processed > 0:
            logger.info(self.preprocessing_stats.get_summary())

        # Split translations back to match original line count
        return unmerge_translations(translations, groups, texts)
```

- [ ] **Step 4: Run integration tests**

Run: `uv run pytest movie_translator/translation/tests/test_batch_translation.py -v`
Expected: PASS

- [ ] **Step 5: Run full translation test suite**

Run: `uv run pytest movie_translator/translation/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/translation/translator.py movie_translator/translation/tests/test_batch_translation.py
git commit -m "feat(translation): integrate sentence-level merging into translate_texts"
```

---

## Task 6: Full Lint Check and Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run ruff check**

Run: `uv run ruff check .`
Expected: All checks passed

- [ ] **Step 2: Run ruff format**

Run: `uv run ruff format --check .`
Expected: All formatted. If not, run `uv run ruff format .`

- [ ] **Step 3: Run ty check**

Run: `uv run ty check`
Expected: All checks passed (0 new diagnostics)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All pass (except the pre-existing OpenSubtitles flaky test)

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -A
git commit -m "chore: lint and format fixes for sentence merger"
```
