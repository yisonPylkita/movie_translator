"""Test safeguards for empty translations and short inputs."""

from unittest.mock import MagicMock

import pytest

from movie_translator.translation.translator import SubtitleTranslator


class TestTranslationSafeguards:
    @pytest.fixture
    def translator(self):
        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=3)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()
        return translator

    def test_empty_translation_fallback(self, translator):
        """Test that empty translations fall back to original text."""
        texts = ['Hello world', 'Test line', 'Another line']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['Witaj świecie', '', 'Inna linia']

        result = translator.translate_texts(texts)

        assert len(result) == 3
        assert result[0] == 'Witaj świecie'
        assert result[1] == 'Test line'
        assert result[2] == 'Inna linia'

    def test_all_empty_translations_fallback(self, translator):
        """Test that all empty translations fall back to originals."""
        texts = ['Line one', 'Line two', 'Line three']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['', '', '']

        result = translator.translate_texts(texts)

        assert result == texts

    def test_suspiciously_short_translation_fallback(self, translator):
        """Test that suspiciously short translations fall back to original."""
        texts = ['This is a longer sentence', 'Short']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['A', 'X']

        result = translator.translate_texts(texts)

        assert result[0] == 'This is a longer sentence'
        assert result[1] == 'X'

    def test_whitespace_only_translation_fallback(self, translator):
        """Test that whitespace-only translations fall back to original."""
        texts = ['Hello', 'World']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['Cześć', '   \n\t  ']

        result = translator.translate_texts(texts)

        assert result[0] == 'Cześć'
        assert result[1] == 'World'

    def test_valid_short_translations_preserved(self, translator):
        """Test that valid short translations are not incorrectly flagged."""
        texts = ['Yes', 'No', 'Ok']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['Tak', 'Nie', 'Ok']

        result = translator.translate_texts(texts)

        assert result == ['Tak', 'Nie', 'Ok']

    def test_mixed_empty_and_valid_translations(self, translator):
        """Test batch with mix of empty and valid translations."""
        texts = ['First line', 'Second line', 'Third line', 'Fourth line']

        batch_index = [0]
        expected_batches = [['Pierwsza linia', '', 'Trzecia linia'], ['']]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            result = expected_batches[batch_index[0]]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert result[0] == 'Pierwsza linia'
        assert result[1] == 'Second line'
        assert result[2] == 'Trzecia linia'
        assert result[3] == 'Fourth line'

    def test_fallback_preserves_original_formatting(self, translator):
        """Test that fallback preserves original text including whitespace."""
        texts = ['  Morning  ', '\tEvening\n']

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['', '']

        result = translator.translate_texts(texts)

        assert result[0] == '  Morning  '
        assert result[1] == '\tEvening\n'
