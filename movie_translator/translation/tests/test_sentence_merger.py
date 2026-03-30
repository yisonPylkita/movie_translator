"""Tests for sentence_merger — sentence-level grouping for subtitle translation."""

from __future__ import annotations

from movie_translator.translation.sentence_merger import (
    TranslationGroup,
    build_input,
    group_lines,
    is_sentence_end,
    is_speaker_line,
    merge_for_translation,
    split_output,
    unmerge_translations,
)

# ---- is_sentence_end --------------------------------------------------------


class TestIsSentenceEnd:
    def test_period(self):
        assert is_sentence_end('The end.') is True

    def test_exclamation(self):
        assert is_sentence_end('Stop!') is True

    def test_question_mark(self):
        assert is_sentence_end('Really?') is True

    def test_quoted_punctuation(self):
        assert is_sentence_end('He said "hello."') is True
        assert is_sentence_end("She yelled 'go!'") is True

    def test_ellipsis_not_terminal(self):
        assert is_sentence_end('And then...') is False

    def test_comma_not_terminal(self):
        assert is_sentence_end('However,') is False

    def test_colon_not_terminal(self):
        assert is_sentence_end('He said:') is False

    def test_semicolon_not_terminal(self):
        assert is_sentence_end('First part;') is False

    def test_no_punctuation(self):
        assert is_sentence_end('The Empire') is False

    def test_empty_string(self):
        assert is_sentence_end('') is False

    def test_whitespace_only(self):
        assert is_sentence_end('   ') is False

    def test_closing_paren_after_period(self):
        assert is_sentence_end('(the end.)') is True

    def test_closing_bracket_after_question(self):
        assert is_sentence_end('Is it true?]') is True

    def test_two_dots_not_terminal(self):
        assert is_sentence_end('Wait..') is False


# ---- is_speaker_line ---------------------------------------------------------


class TestIsSpeakerLine:
    def test_dash_space(self):
        assert is_speaker_line('- Hello there.') is True

    def test_dash_no_space(self):
        assert is_speaker_line('-Hello there.') is True

    def test_no_dash(self):
        assert is_speaker_line('Hello there.') is False

    def test_dash_mid_text(self):
        # Dash in the middle of text is NOT a speaker line
        assert is_speaker_line('Hello - there.') is False

    def test_em_dash(self):
        assert is_speaker_line('\u2014 Who goes there?') is True

    def test_en_dash(self):
        assert is_speaker_line('\u2013 Who goes there?') is True

    def test_empty(self):
        assert is_speaker_line('') is False

    def test_dash_only(self):
        # A lone dash with no following text
        assert is_speaker_line('-') is False


# ---- group_lines -------------------------------------------------------------


class TestGroupLines:
    def test_single_complete_sentence(self):
        groups = group_lines(['Hello world.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0]
        assert groups[0].is_fragment_merge is False

    def test_two_short_independent_sentences_are_solo(self):
        # Short lines (<=3 words) get solo groups to avoid model garbling
        groups = group_lines(['Hello world.', 'Goodbye world.'])
        assert len(groups) == 2
        assert groups[0].line_indices == [0]
        assert groups[1].line_indices == [1]

    def test_two_longer_independent_sentences_batched(self):
        # Lines above SHORT_LINE_MAX_WORDS still get batched with ||
        groups = group_lines(
            [
                'The Empire declared total war.',
                'The Republic responded in kind.',
            ]
        )
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1]
        assert groups[0].is_fragment_merge is False

    def test_fragment_merge_two_lines(self):
        groups = group_lines(['The Empire', 'declared war.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1]
        assert groups[0].is_fragment_merge is True

    def test_three_line_fragment(self):
        groups = group_lines(['I want', 'to tell', 'you something.'])
        assert len(groups) == 1
        assert groups[0].line_indices == [0, 1, 2]
        assert groups[0].is_fragment_merge is True

    def test_mixed_fragment_and_independent(self):
        groups = group_lines(['The Empire', 'declared war.', 'We must fight.'])
        assert len(groups) == 2
        # First group: fragment merge
        assert groups[0].line_indices == [0, 1]
        assert groups[0].is_fragment_merge is True
        # Second group: independent
        assert groups[1].line_indices == [2]
        assert groups[1].is_fragment_merge is False

    def test_speaker_lines_stay_individual(self):
        groups = group_lines(['- Hello.', '- Goodbye.'])
        assert len(groups) == 2
        assert groups[0].line_indices == [0]
        assert groups[1].line_indices == [1]

    def test_speaker_breaks_group(self):
        groups = group_lines(['Hello.', '- Speaker one.', 'Goodbye.'])
        assert len(groups) == 3
        assert groups[0].line_indices == [0]
        assert groups[1].line_indices == [1]
        assert groups[2].line_indices == [2]

    def test_fragment_before_speaker(self):
        # Fragment cannot merge with speaker line — fragment group ends
        groups = group_lines(['The Empire', '- declared war.'])
        assert len(groups) == 2
        assert groups[0].line_indices == [0]
        assert groups[0].is_fragment_merge is True
        assert groups[1].line_indices == [1]

    def test_empty_input(self):
        assert group_lines([]) == []

    def test_ellipsis_merges(self):
        groups = group_lines(['And then...', 'it happened.'])
        assert len(groups) == 1
        assert groups[0].is_fragment_merge is True
        assert groups[0].line_indices == [0, 1]

    def test_comma_merges(self):
        groups = group_lines(['However,', 'it was fine.'])
        assert len(groups) == 1
        assert groups[0].is_fragment_merge is True


# ---- build_input -------------------------------------------------------------


class TestBuildInput:
    def test_single_line(self):
        texts = ['Hello world.']
        group = TranslationGroup(line_indices=[0], is_fragment_merge=False)
        assert build_input(texts, group) == 'Hello world.'

    def test_fragment_joins_with_space(self):
        texts = ['The Empire', 'declared war.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=True)
        assert build_input(texts, group) == 'The Empire declared war.'

    def test_independent_joins_with_double_pipe(self):
        texts = ['Hello world.', 'Goodbye world.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=False)
        assert build_input(texts, group) == 'Hello world. || Goodbye world.'

    def test_mixed_fragment_then_independent(self):
        # Fragment group
        texts = ['The Empire', 'declared war.', 'We must fight.']
        frag = TranslationGroup(line_indices=[0, 1], is_fragment_merge=True)
        indep = TranslationGroup(line_indices=[2], is_fragment_merge=False)
        assert build_input(texts, frag) == 'The Empire declared war.'
        assert build_input(texts, indep) == 'We must fight.'

    def test_three_independent(self):
        texts = ['One.', 'Two.', 'Three.']
        group = TranslationGroup(line_indices=[0, 1, 2], is_fragment_merge=False)
        assert build_input(texts, group) == 'One. || Two. || Three.'

    def test_fragment_at_end(self):
        texts = ['Hello world.', 'The Empire']
        frag = TranslationGroup(line_indices=[1], is_fragment_merge=True)
        assert build_input(texts, frag) == 'The Empire'

    def test_ellipsis_continuation(self):
        texts = ['And then...', 'it happened.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=True)
        assert build_input(texts, group) == 'And then... it happened.'


# ---- split_output ------------------------------------------------------------


class TestSplitOutput:
    def test_single_line_passthrough(self):
        texts = ['Hello world.']
        group = TranslationGroup(line_indices=[0], is_fragment_merge=False)
        assert split_output('Witaj świecie.', group, texts) == ['Witaj świecie.']

    def test_double_pipe_split(self):
        texts = ['Hello.', 'Goodbye.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=False)
        result = split_output('Cześć. || Do widzenia.', group, texts)
        assert result == ['Cześć.', 'Do widzenia.']

    def test_proportional_split_2_lines(self):
        texts = ['The Empire', 'declared war on the rebels.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=True)
        # 2 words vs 5 words => ~29% vs ~71%
        result = split_output('Imperium wypowiedziało wojnę rebeliantom z północy.', group, texts)
        assert len(result) == 2
        # All words accounted for
        assert ' '.join(result) == 'Imperium wypowiedziało wojnę rebeliantom z północy.'

    def test_proportional_split_3_lines(self):
        texts = ['I want', 'to tell', 'you something important.']
        group = TranslationGroup(line_indices=[0, 1, 2], is_fragment_merge=True)
        result = split_output('Chcę ci powiedzieć coś ważnego teraz.', group, texts)
        assert len(result) == 3
        assert ' '.join(result) == 'Chcę ci powiedzieć coś ważnego teraz.'

    def test_missing_separator_fallback(self):
        # Model lost the || — should fall back to proportional split
        texts = ['Hello.', 'Goodbye.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=False)
        result = split_output('Cześć Do widzenia', group, texts)
        assert len(result) == 2
        assert ' '.join(result) == 'Cześć Do widzenia'

    def test_whitespace_around_pipes(self):
        texts = ['One.', 'Two.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=False)
        result = split_output('  Jeden.  ||  Dwa.  ', group, texts)
        assert result == ['Jeden.', 'Dwa.']

    def test_mixed_group(self):
        # Three independent sentences
        texts = ['One.', 'Two.', 'Three.']
        group = TranslationGroup(line_indices=[0, 1, 2], is_fragment_merge=False)
        result = split_output('Jeden. || Dwa. || Trzy.', group, texts)
        assert result == ['Jeden.', 'Dwa.', 'Trzy.']

    def test_empty_translation(self):
        texts = ['The Empire', 'declared war.']
        group = TranslationGroup(line_indices=[0, 1], is_fragment_merge=True)
        result = split_output('', group, texts)
        assert result == ['', '']


# ---- merge_for_translation / unmerge_translations ----------------------------


class TestRoundTrip:
    def test_round_trip_independent(self):
        # Use longer sentences so they get batched (short ones are solo now)
        texts = [
            'The Empire declared total war.',
            'The Republic responded in kind.',
        ]
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 1
        assert '||' in merged[0]

        # Simulate translation preserving ||
        translated = ['Imperium wypowiedziało totalną wojnę. || Republika odpowiedziała tym samym.']
        result = unmerge_translations(translated, groups, texts)
        assert len(result) == 2
        assert result[0] == 'Imperium wypowiedziało totalną wojnę.'
        assert result[1] == 'Republika odpowiedziała tym samym.'

    def test_round_trip_fragments(self):
        texts = ['The Empire', 'declared war.']
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 1
        assert '||' not in merged[0]
        assert merged[0] == 'The Empire declared war.'

        translated = ['Imperium wypowiedziało wojnę.']
        result = unmerge_translations(translated, groups, texts)
        assert len(result) == 2
        assert ' '.join(result) == 'Imperium wypowiedziało wojnę.'

    def test_round_trip_speaker_lines(self):
        texts = ['- Hello.', '- Goodbye.']
        merged, groups = merge_for_translation(texts)
        assert len(merged) == 2
        assert merged[0] == '- Hello.'
        assert merged[1] == '- Goodbye.'

        translated = ['- Cześć.', '- Do widzenia.']
        result = unmerge_translations(translated, groups, texts)
        assert result == ['- Cześć.', '- Do widzenia.']

    def test_preserves_line_count(self):
        texts = [
            'The Empire',
            'declared war.',
            'We must fight.',
            '- Run!',
            'And then...',
            'it was over.',
        ]
        merged, groups = merge_for_translation(texts)
        # Simulate translation (doesn't matter what — just need right count)
        translated = [f'trans_{i}' for i in range(len(merged))]
        result = unmerge_translations(translated, groups, texts)
        assert len(result) == len(texts)

    def test_round_trip_mixed_complex(self):
        texts = [
            'Hello.',
            'The Empire',
            'declared war.',
            '- Run!',
            'Goodbye.',
        ]
        merged, groups = merge_for_translation(texts)
        # groups: [Hello.] [The Empire + declared war.] [- Run!] [Goodbye.]
        assert len(groups) == 4
        assert groups[0].line_indices == [0]
        assert groups[0].is_fragment_merge is False
        assert groups[1].line_indices == [1, 2]
        assert groups[1].is_fragment_merge is True
        assert groups[2].line_indices == [3]
        assert groups[3].line_indices == [4]
