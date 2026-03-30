from unittest.mock import MagicMock, patch

from movie_translator.translation.translator import SubtitleTranslator
from movie_translator.types import DialogueLine


class TestBatchTranslation:
    def test_translation_across_batch_boundary_size_3_lines_8(self):
        # Speaker-dash lines each get their own translation group (no merging),
        # so all 8 lines become 8 translation units that span batch boundaries.
        texts = [f'- Line {i}.' for i in range(1, 9)]
        expected_translations = [f'- Linia {i}.' for i in range(1, 9)]

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=3)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        batch_index = [0]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            start_idx = batch_index[0] * 3
            end_idx = min(start_idx + 3, len(expected_translations))
            result = expected_translations[start_idx:end_idx]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert len(result) == 8
        for i in range(8):
            assert result[i] == expected_translations[i], f'Line {i} incorrect: {result[i]}'

    def test_translation_across_batch_boundary_size_5_lines_7(self):
        # Speaker-dash lines prevent merging, giving 7 translation units.
        texts = [f'- English {i}.' for i in range(1, 8)]
        expected_translations = [f'- Polski {i}.' for i in range(1, 8)]

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=5)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        batch_index = [0]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            start_idx = batch_index[0] * 5
            end_idx = min(start_idx + 5, len(expected_translations))
            result = expected_translations[start_idx:end_idx]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert len(result) == 7
        for i in range(7):
            assert result[i] == expected_translations[i]

    def test_translation_batch_size_1_worst_case(self):
        # Speaker-dash lines prevent merging, giving 3 individual units.
        texts = ['- Good.', '- World.', '- Test.']
        expected = ['- Dobry.', '- Swiat.', '- Testuj.']

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=1)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        batch_index = [0]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            result = [expected[batch_index[0]]]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert len(result) == 3
        assert result == expected

    def test_translation_exact_batch_multiple(self):
        # Speaker-dash lines prevent merging, giving 8 individual units.
        texts = [f'- Line {i}.' for i in range(1, 9)]
        expected = [f'- Linia {i}.' for i in range(1, 9)]

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=4)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        batch_index = [0]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            start_idx = batch_index[0] * 4
            end_idx = min(start_idx + 4, len(expected))
            result = expected[start_idx:end_idx]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert len(result) == 8
        assert result == expected

    def test_translation_single_line_edge_case(self):
        texts = ['Single line.']

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=10)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        translator.tokenizer.batch_encode_plus.return_value = {'input_ids': MagicMock()}
        translator.model.generate.return_value = MagicMock()
        translator.tokenizer.batch_decode.return_value = ['Pojedyncza linia.']

        result = translator.translate_texts(texts)

        assert len(result) == 1
        assert result[0] == 'Pojedyncza linia.'

    def test_translation_empty_list(self):
        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=3)
        result = translator.translate_texts([])
        assert result == []

    def test_all_lines_translated_no_english_remainder(self):
        # Speaker-dash lines prevent merging, giving 7 individual translation units
        # that span batch boundaries (batch_size=3 -> 3 batches).
        texts = ['- Good.', '- World.', '- How?', '- Are.', '- You.', '- Today.', '- Friend.']
        expected = [
            '- Dobry.',
            '- Swiat.',
            '- Jak?',
            '- Sie.',
            '- Masz.',
            '- Dzisiaj.',
            '- Przyjaciel.',
        ]

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=3)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        batch_index = [0]

        def mock_encode(texts_list, **kwargs):
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            start_idx = batch_index[0] * 3
            end_idx = min(start_idx + 3, len(expected))
            result = expected[start_idx:end_idx]
            batch_index[0] += 1
            return result

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        assert len(result) == 7
        assert result == expected
        for original in texts:
            assert original not in result, (
                f'Original English text "{original}" found in translated output'
            )


class TestSentenceMerging:
    def test_fragment_lines_merged_before_translation(self):
        """When two lines form one sentence, the model should receive one merged string."""
        texts = ['The Empire', 'declared war.']

        # Disable enhancements to avoid placeholder extraction altering the text
        translator = SubtitleTranslator(
            model_key='allegro', device='cpu', batch_size=10, enable_enhancements=False
        )
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        # Capture what texts the model receives
        received_texts = []

        def mock_encode(texts_list, **kwargs):
            received_texts.extend(texts_list)
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            # Return a translation for the single merged unit
            return ['Imperium wypowiedzialo wojne.']

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        # Model should have received 1 merged text, not 2 separate lines.
        # The BiDi model adds '>>pol<< ' prefix during preprocessing.
        assert len(received_texts) == 1
        assert 'The Empire declared war.' in received_texts[0]

        # Result must have 2 lines (split back to original count)
        assert len(result) == 2

    def test_speaker_lines_not_merged(self):
        """Speaker dash lines must be translated individually."""
        texts = ["- I'll go.", '- No, stay!']

        translator = SubtitleTranslator(model_key='allegro', device='cpu', batch_size=10)
        translator.tokenizer = MagicMock()
        translator.model = MagicMock()

        received_texts = []

        def mock_encode(texts_list, **kwargs):
            received_texts.extend(texts_list)
            return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

        def mock_generate(**kwargs):
            return MagicMock()

        def mock_decode(outputs, **kwargs):
            # Each speaker line is its own group, so model receives 2 texts
            # in one batch and must return 2 translations.
            return ['- Pojde.', '- Nie, zostań!']

        translator.tokenizer.batch_encode_plus.side_effect = mock_encode
        translator.model.generate.side_effect = mock_generate
        translator.tokenizer.batch_decode.side_effect = mock_decode

        result = translator.translate_texts(texts)

        # Speaker lines should each be in their own group, meaning the model
        # receives 2 separate texts (not space-joined into one).
        assert len(received_texts) == 2
        assert len(result) == 2


class TestDialogueLineBatchTranslation:
    def test_translate_dialogue_lines_preserves_timing(self):
        dialogue_lines = [
            DialogueLine(1000, 2000, 'Hello'),
            DialogueLine(2000, 3000, 'World'),
            DialogueLine(3000, 4000, 'Test'),
        ]

        def mock_translate(texts, callback=None):
            if callback:
                callback(1, 1, 1.0)
            return ['Witaj', 'Swiat', 'Testuj']

        mock_translator = MagicMock()
        mock_translator.load_model.return_value = True
        mock_translator.translate_texts.side_effect = mock_translate

        with patch(
            'movie_translator.translation.translator.SubtitleTranslator',
            return_value=mock_translator,
        ):
            from movie_translator.translation.translator import translate_dialogue_lines

            result = translate_dialogue_lines(dialogue_lines, 'cpu', 2, 'allegro')

        assert len(result) == 3
        assert result[0] == DialogueLine(1000, 2000, 'Witaj')
        assert result[1] == DialogueLine(2000, 3000, 'Swiat')
        assert result[2] == DialogueLine(3000, 4000, 'Testuj')

    def test_translate_dialogue_lines_batch_size_smaller_than_lines(self):
        dialogue_lines = [DialogueLine(i * 1000, (i + 1) * 1000, f'Line {i}') for i in range(8)]

        def mock_translate(texts, callback=None):
            for batch_num in range(1, 4):
                if callback:
                    callback(batch_num, 3, 2.5)
            return [f'Linia {i}' for i in range(8)]

        mock_translator = MagicMock()
        mock_translator.load_model.return_value = True
        mock_translator.translate_texts.side_effect = mock_translate

        with patch(
            'movie_translator.translation.translator.SubtitleTranslator',
            return_value=mock_translator,
        ):
            from movie_translator.translation.translator import translate_dialogue_lines

            result = translate_dialogue_lines(dialogue_lines, 'cpu', 3, 'allegro')

        assert len(result) == 8
        for i, line in enumerate(result):
            assert line.start_ms == i * 1000
            assert line.end_ms == (i + 1) * 1000
            assert line.text == f'Linia {i}'
