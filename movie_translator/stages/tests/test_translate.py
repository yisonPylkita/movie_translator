from unittest.mock import MagicMock, patch

import pytest

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.stages.translate import TranslateStage
from movie_translator.types import DialogueLine


class TestTranslateStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        eng_src = tmp_path / 'eng.srt'
        eng_src.touch()
        mock_cache = MagicMock()
        mock_cache.get_translator.return_value = (MagicMock(), False)
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(model_cache=mock_cache),
        )
        ctx.english_source = eng_src
        ctx.dialogue_lines = [
            DialogueLine(1000, 3000, 'Hello'),
            DialogueLine(4000, 6000, 'Goodbye'),
        ]
        return ctx

    def test_sets_translated_lines_and_font_info(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        translated = [
            DialogueLine(1000, 3000, 'Cześć'),
            DialogueLine(4000, 6000, 'Do widzenia'),
        ]

        with (
            patch(
                'movie_translator.stages.translate.translate_dialogue_lines',
                return_value=translated,
            ),
            patch(
                'movie_translator.stages.translate.check_embedded_fonts_support_polish',
                return_value=False,
            ),
            patch('movie_translator.stages.translate.get_ass_font_names', return_value={'Arial'}),
            patch(
                'movie_translator.stages.translate.find_system_font_for_polish', return_value=None
            ),
        ):
            result = TranslateStage().run(ctx)

        assert result.translated_lines == translated
        assert result.font_info is not None
        assert result.font_info.supports_polish is False

    def test_raises_on_empty_translation(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with (
            patch('movie_translator.stages.translate.translate_dialogue_lines', return_value=[]),
            patch(
                'movie_translator.stages.translate.check_embedded_fonts_support_polish',
                return_value=False,
            ),
            patch('movie_translator.stages.translate.get_ass_font_names', return_value=set()),
            patch(
                'movie_translator.stages.translate.find_system_font_for_polish', return_value=None
            ),
        ):
            with pytest.raises(RuntimeError, match='Translation failed'):
                TranslateStage().run(ctx)
