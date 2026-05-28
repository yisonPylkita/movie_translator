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

    def test_generate_translations_passes_inputs_to_model(self):
        """generate() receives the encoded inputs and decoding kwargs."""
        translator = SubtitleTranslator.__new__(SubtitleTranslator)
        translator.model_config = {'huggingface_id': 'allegro/BiDi-eng-pol'}

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        translator.model = mock_model
        translator.tokenizer = None

        encoded = {'input_ids': torch.tensor([[1, 2]])}
        translator._generate_translations(encoded)

        call_kwargs = mock_model.generate.call_args[1]
        # No backend-specific forced_bos / language hints needed for Allegro.
        assert 'forced_bos_token_id' not in call_kwargs
        assert call_kwargs['num_beams'] == 1
        assert call_kwargs['do_sample'] is False
        assert call_kwargs['max_new_tokens'] == 128
