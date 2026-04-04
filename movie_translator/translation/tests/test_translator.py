"""Tests for SubtitleTranslator."""

from unittest.mock import MagicMock

import torch

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

    def test_nllb_model_does_not_add_prefix(self):
        """NLLB models should NOT get the >>pol<< prefix."""
        translator = SubtitleTranslator(model_key='nllb-600m')
        result = translator._preprocess_texts(['Hello world'])
        assert result == ['Hello world']
        assert '>>pol<<' not in result[0]


class TestNllbIntegration:
    """Tests for NLLB-specific behavior."""

    def test_is_nllb_true_for_nllb_models(self):
        translator = SubtitleTranslator(model_key='nllb-600m')
        assert translator._is_nllb is True

        translator = SubtitleTranslator(model_key='nllb-1.3b')
        assert translator._is_nllb is True

    def test_is_nllb_false_for_allegro(self):
        translator = SubtitleTranslator(model_key='allegro')
        assert translator._is_nllb is False

    def test_generate_translations_nllb_uses_forced_bos(self):
        """NLLB models must pass forced_bos_token_id to generate()."""
        translator = SubtitleTranslator.__new__(SubtitleTranslator)
        translator.model_config = {'huggingface_id': 'facebook/nllb-200-distilled-600M'}

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        translator.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256047
        translator.tokenizer = mock_tokenizer

        encoded = {'input_ids': torch.tensor([[1, 2]])}
        translator._generate_translations(encoded)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs['forced_bos_token_id'] == 256047
        mock_tokenizer.convert_tokens_to_ids.assert_called_once_with('pol_Latn')

    def test_generate_translations_allegro_no_forced_bos(self):
        """Allegro models should NOT pass forced_bos_token_id."""
        translator = SubtitleTranslator.__new__(SubtitleTranslator)
        translator.model_config = {'huggingface_id': 'allegro/BiDi-eng-pol'}

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        translator.model = mock_model
        translator.tokenizer = None

        encoded = {'input_ids': torch.tensor([[1, 2]])}
        translator._generate_translations(encoded)

        call_kwargs = mock_model.generate.call_args[1]
        assert 'forced_bos_token_id' not in call_kwargs
