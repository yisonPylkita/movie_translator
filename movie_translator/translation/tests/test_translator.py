"""Tests for SubtitleTranslator."""

from movie_translator.translation.translator import SubtitleTranslator


class TestPreprocessTexts:
    """Tests for the _preprocess_texts method."""

    def test_bidi_model_adds_polish_prefix(self):
        """BiDi models require >>pol<< prefix for Polish translation."""
        translator = SubtitleTranslator(model_key='allegro')

        texts = ['Hello world', 'How are you?']
        result = translator._preprocess_texts(texts)

        assert result == ['>>pol<< Hello world', '>>pol<< How are you?']

    def test_non_bidi_model_does_not_add_prefix(self):
        """Non-BiDi models should not have any prefix added."""
        translator = SubtitleTranslator(model_key='some-other-model')

        texts = ['Hello world']
        result = translator._preprocess_texts(texts)

        assert result == ['Hello world']
