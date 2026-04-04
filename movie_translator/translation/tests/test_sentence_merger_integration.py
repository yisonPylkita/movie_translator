"""Integration tests for sentence merger with the real BiDi model.

These tests load the actual allegro/BiDi-eng-pol model to validate
that the sentence merging pipeline produces correct translations.
The model is deterministic (greedy decoding, do_sample=False), so
outputs are reproducible.

Tests cover:
- No pipe (||, |) characters leak into translated output
- Correct line count preservation
- No repetition loops in battle scenes with many short sentences
- || separator fidelity within small groups (<=3)
- Stutter prefix handling
- Phrase cache integration
"""

import re
import warnings

import pytest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from movie_translator.translation.models import get_local_model_path
from movie_translator.translation.sentence_merger import (
    MAX_BATCH_SENTENCES,
    group_lines,
    merge_for_translation,
    split_output,
    unmerge_translations,
)

warnings.filterwarnings('ignore', message='.*sacremoses.*')

MODEL_PATH = 'allegro/BiDi-eng-pol'

_PIPE_RE = re.compile(r'\|')

# Evaluate once at collection time so pytest-xdist can skip before distributing
_local_model_path = get_local_model_path('allegro')
needs_model = pytest.mark.skipif(
    _local_model_path is None,
    reason='Local allegro model not available',
)


@pytest.fixture(scope='module')
def model_and_tokenizer():
    """Load the BiDi model once for all tests in this module."""
    assert _local_model_path is not None
    path = str(_local_model_path)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return tokenizer, model, device


def translate(tokenizer, model, device, texts: list[str], max_new_tokens: int = 128) -> list[str]:
    """Translate texts using the BiDi model."""
    prefixed = [f'>>pol<< {t}' for t in texts]
    encoded = tokenizer.batch_encode_plus(
        prefixed,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    )
    if device != 'cpu':
        encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    return tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


class TestGroupSizeCap:
    """Verify that || groups are capped at MAX_BATCH_SENTENCES."""

    def test_short_exclamation_sequence_capped(self):
        """Battle scene with many short sentences must not create huge groups."""
        texts = [
            'Enemy raid!',
            'What?',
            'An enemy raid?',
            'Enemy raid!',
            "Let's go!",
            'Ready!',
            'Enemy raid!',
            'They finally showed their faces!',
            'Aye, sir!',
        ]
        groups = group_lines(texts)
        for g in groups:
            assert len(g.line_indices) <= MAX_BATCH_SENTENCES, (
                f'Group has {len(g.line_indices)} lines, exceeds cap of {MAX_BATCH_SENTENCES}'
            )

    def test_group_cap_preserves_all_lines(self):
        """Capping must not lose any lines."""
        texts = ['Line one!', 'Line two!', 'Line three!', 'Line four!', 'Line five!']
        groups = group_lines(texts)
        all_indices = []
        for g in groups:
            all_indices.extend(g.line_indices)
        assert sorted(all_indices) == list(range(len(texts)))


@needs_model
class TestNoPipeLeaks:
    """Verify that no | or || characters appear in translated output."""

    def test_battle_scene_no_pipes(self, model_and_tokenizer):
        """Translate a battle scene — the exact scenario that caused the original bug."""
        tokenizer, model, device = model_and_tokenizer
        texts = [
            'Do not cease to be vigilant!',
            'Yes, sir!',
            'Our commander is unusually alert.',
            "Yeah, it's because he's been given such an important job.",
            'I suppose he must be enthusiastic.',
            'Is it from Guts?',
            'No doubt. It is the signal for the next phase!',
            'Good!',
            "Let's move.",
            'Enemy raid!',
            'What?',
            'An enemy raid?',
            'Enemy raid!',
            "Let's go!",
            'Ready!',
            'Enemy raid!',
        ]

        merged, groups = merge_for_translation(texts)
        translations = translate(tokenizer, model, device, merged, max_new_tokens=128)
        result = unmerge_translations(translations, groups, texts)

        assert len(result) == len(texts)
        for i, text in enumerate(result):
            assert not _PIPE_RE.search(text), (
                f'Pipe character found in line {i}: "{text}" (original: "{texts[i]}")'
            )

    def test_long_dialogue_scene_no_pipes(self, model_and_tokenizer):
        """Translate a long dialogue sequence with mixed sentence types."""
        tokenizer, model, device = model_and_tokenizer
        texts = [
            'How could those upstart thieves take that from us?',
            'What is His Majesty thinking?',
            'Will you allow this affair to take place?',
            'What do you mean?',
            'The White Dragons were reputed to be the strongest unit.',
            'Nevertheless, their glory has turned insignificant...',
            '...due to the rise of the Hawks.',
            'This is a rumor.',
            'As I say, it is a rumor within the court.',
        ]

        merged, groups = merge_for_translation(texts)
        translations = translate(tokenizer, model, device, merged, max_new_tokens=128)
        result = unmerge_translations(translations, groups, texts)

        assert len(result) == len(texts)
        for i, text in enumerate(result):
            assert not _PIPE_RE.search(text), f'Pipe character found in line {i}: "{text}"'


@needs_model
class TestSeparatorFidelity:
    """Verify that || separators work correctly with small groups."""

    def test_two_sentence_separator_preserved(self, model_and_tokenizer):
        """Model should preserve || with just 2 sentences."""
        tokenizer, model, device = model_and_tokenizer
        text = 'This is a trap! || How foolish they are!'
        result = translate(tokenizer, model, device, [text])[0]
        parts = [p.strip() for p in result.split('||')]
        assert len(parts) == 2, f'Expected 2 parts, got {len(parts)}: "{result}"'

    def test_three_sentence_separator_preserved(self, model_and_tokenizer):
        """Model should preserve || with 3 sentences."""
        tokenizer, model, device = model_and_tokenizer
        text = 'Open the gate! || What happened? || Certainly!'
        result = translate(tokenizer, model, device, [text])[0]
        parts = [p.strip() for p in result.split('||')]
        assert len(parts) == 3, f'Expected 3 parts, got {len(parts)}: "{result}"'


@needs_model
class TestNoRepetitionLoops:
    """Verify the model doesn't enter repetition loops."""

    def test_repetitive_input_no_loop(self, model_and_tokenizer):
        """Repetitive short sentences should not cause model to loop."""
        tokenizer, model, device = model_and_tokenizer
        # With capped groups, this becomes multiple small groups
        texts = ['Enemy raid!', 'Enemy raid!', 'Enemy raid!']
        merged, groups = merge_for_translation(texts)
        translations = translate(tokenizer, model, device, merged, max_new_tokens=128)
        result = unmerge_translations(translations, groups, texts)

        for i, text in enumerate(result):
            # No single word should repeat 3+ times
            words = text.lower().split()
            if words:
                from collections import Counter

                counts = Counter(words)
                for word, count in counts.items():
                    if len(word) > 2:
                        assert count < 3, (
                            f'Repetition loop detected in line {i}: '
                            f'"{word}" appears {count} times in "{text}"'
                        )


@needs_model
class TestLineCountPreservation:
    """Verify that merge/unmerge round-trip preserves line count."""

    def test_line_count_preserved_mixed_content(self, model_and_tokenizer):
        """Translation should return exactly the same number of lines as input."""
        tokenizer, model, device = model_and_tokenizer
        texts = [
            'In this world...',
            'Is the destiny of mankind controlled by some transcendental entity?',
            'At least it is true...',
            '...that man has no control, even over his own will.',
            'Why does Griffith value you so much?',
            "There's no reason...",
            '...in particular.',
            'All right!',
            'Stretch out the rest of the ropes!',
            'Any sign of the enemy?',
            'Nothing at all.',
        ]

        merged, groups = merge_for_translation(texts)
        translations = translate(tokenizer, model, device, merged, max_new_tokens=128)
        result = unmerge_translations(translations, groups, texts)

        assert len(result) == len(texts), f'Expected {len(texts)} lines, got {len(result)}'


@needs_model
class TestTranslationQuality:
    """Verify basic translation quality with the real model."""

    def test_simple_sentences_translated(self, model_and_tokenizer):
        """Simple complete sentences should produce non-empty Polish output."""
        tokenizer, model, device = model_and_tokenizer
        texts = [
            'You did well.',
            'We did it!',
            'I am always prepared!',
        ]

        merged, groups = merge_for_translation(texts)
        translations = translate(tokenizer, model, device, merged, max_new_tokens=128)
        result = unmerge_translations(translations, groups, texts)

        for i, text in enumerate(result):
            assert text.strip(), f'Empty translation for line {i}: "{texts[i]}"'
            # Should contain Polish characters or at least be different from English
            # (very basic check - translation should not be identical to input)
            assert text.strip() != texts[i].strip(), (
                f'Translation identical to input for line {i}: "{text}"'
            )

    def test_known_good_translations(self, model_and_tokenizer):
        """Verify specific translations match expected model output.

        Model outputs are deterministic (greedy decoding). These expected
        values are from standalone single-sentence translation.
        """
        tokenizer, model, device = model_and_tokenizer
        test_cases = [
            ('This is their trap!', 'To ich pułapka!'),
            ('How foolish they are!', 'Jakie one są głupie!'),
        ]
        for english, expected_polish in test_cases:
            result = translate(tokenizer, model, device, [english])[0]
            assert result.strip() == expected_polish, (
                f'Expected "{expected_polish}", got "{result}" for "{english}"'
            )


class TestPipeStripping:
    """Test the pipe stripping in split_output fallback."""

    def test_proportional_split_strips_pipes(self):
        """When || split fails, proportional split must strip pipe chars."""
        from movie_translator.translation.sentence_merger import TranslationGroup

        group = TranslationGroup(line_indices=[0, 1, 2], is_fragment_merge=False)
        # Simulate a model output with corrupted separators
        translated = 'Cześć! | Jak się masz? || Dobrze!'
        original_texts = ['Hello!', 'How are you?', 'Fine!']

        # || split gives 2 parts (not 3), so falls back to proportional
        result = split_output(translated, group, original_texts)
        assert len(result) == 3
        for part in result:
            assert '|' not in part, f'Pipe found in split output: "{part}"'
