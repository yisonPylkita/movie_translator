from movie_translator.translation.enhancements import (
    postprocess_translation,
    preprocess_for_translation,
)


class TestPreprocessing:
    def test_short_phrase_direct_mapping(self):
        text, was_mapped = preprocess_for_translation('Yes.')
        assert was_mapped is True
        assert text == 'Tak.'

    def test_short_phrase_no(self):
        text, was_mapped = preprocess_for_translation('No.')
        assert was_mapped is True
        assert text == 'Nie.'

    def test_short_phrase_wait(self):
        text, was_mapped = preprocess_for_translation('Wait!')
        assert was_mapped is True
        assert text == 'Czekaj!'

    def test_punctuation_variations_exclamation(self):
        """Test that different punctuation variations all map correctly."""
        result, was_mapped = preprocess_for_translation('Sure!')
        assert result == 'Jasne!'
        assert was_mapped is True

    def test_punctuation_variations_question(self):
        result, was_mapped = preprocess_for_translation('Sure?')
        assert result == 'Jasne?'
        assert was_mapped is True

    def test_punctuation_variations_ellipsis(self):
        result, was_mapped = preprocess_for_translation('Sure...')
        assert result == 'Jasne...'
        assert was_mapped is True

    def test_punctuation_variations_multiple(self):
        result, was_mapped = preprocess_for_translation('Sure?!')
        assert result == 'Jasne?!'
        assert was_mapped is True

    def test_capitalization_preserved_upper(self):
        """Test that UPPERCASE is preserved."""
        result, was_mapped = preprocess_for_translation('WAIT')
        assert result == 'CZEKAJ'
        assert was_mapped is True

    def test_capitalization_preserved_title(self):
        """Test that Title case is preserved."""
        result, was_mapped = preprocess_for_translation('Wait')
        assert result == 'Czekaj'
        assert was_mapped is True

    def test_capitalization_preserved_lower(self):
        """Test that lowercase is preserved."""
        result, was_mapped = preprocess_for_translation('wait')
        assert result == 'czekaj'
        assert was_mapped is True

    def test_no_punctuation_works(self):
        """Test phrases without punctuation."""
        result, was_mapped = preprocess_for_translation('okay')
        assert result == 'dobrze'
        assert was_mapped is True

    def test_multi_word_phrase(self):
        """Test multi-word phrase matching."""
        result, was_mapped = preprocess_for_translation('Thank you!')
        assert result == 'Dziękuję!'
        assert was_mapped is True

    def test_multi_word_phrase_various_caps(self):
        """Test multi-word phrase with different capitalization."""
        result, was_mapped = preprocess_for_translation('THANK YOU!')
        assert result == 'DZIĘKUJĘ!'
        assert was_mapped is True

    def test_idiom_break_a_leg(self):
        text, was_mapped = preprocess_for_translation('Break a leg!')
        assert was_mapped is False
        assert 'good luck' in text.lower()
        assert 'break a leg' not in text.lower()

    def test_idiom_raining_cats_and_dogs(self):
        text, was_mapped = preprocess_for_translation('It is raining cats and dogs.')
        assert was_mapped is False
        assert 'raining heavily' in text.lower()

    def test_idiom_case_insensitive(self):
        text, was_mapped = preprocess_for_translation('BREAK A LEG!')
        assert was_mapped is False
        assert 'good luck' in text.lower()

    def test_normal_text_unchanged(self):
        original = 'This is normal dialogue.'
        text, was_mapped = preprocess_for_translation(original)
        assert was_mapped is False
        assert text == original

    def test_phrase_with_whitespace(self):
        text, was_mapped = preprocess_for_translation('  Yes.  ')
        assert was_mapped is True
        assert text == 'Tak.'


class TestPostprocessing:
    def test_remove_simple_repetition(self):
        result = postprocess_translation('Tak, tak.')
        assert result == 'Tak.'

    def test_remove_dialogue_markers_exclamation(self):
        result = postprocess_translation('- Nie! - Nie!')
        assert result == 'Nie!'

    def test_remove_dialogue_markers_period(self):
        result = postprocess_translation('- Tak. - Tak.')
        assert result == 'Tak.'

    def test_normalize_multiple_punctuation(self):
        result = postprocess_translation('Co!!!!')
        assert result == 'Co!'

    def test_normalize_space_before_punctuation(self):
        result = postprocess_translation('Tak .')
        assert result == 'Tak.'

    def test_clean_text_unchanged(self):
        original = 'To jest normalne zdanie.'
        result = postprocess_translation(original)
        assert result == original

    def test_empty_text(self):
        result = postprocess_translation('')
        assert result == ''

    def test_whitespace_only(self):
        result = postprocess_translation('   ')
        assert result == ''


class TestIntegration:
    def test_preprocessing_then_postprocessing(self):
        text = 'Yes.'
        preprocessed, was_mapped = preprocess_for_translation(text)
        assert preprocessed == 'Tak.'

        final = postprocess_translation(preprocessed)
        assert final == 'Tak.'

    def test_idiom_preprocessing(self):
        text = 'Break a leg with your performance!'
        preprocessed, was_mapped = preprocess_for_translation(text)
        assert 'good luck' in preprocessed.lower()
        assert 'break' not in preprocessed.lower()
