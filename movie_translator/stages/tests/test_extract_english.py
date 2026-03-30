from unittest.mock import MagicMock, patch

import pytest

from movie_translator.context import FetchedSubtitle, PipelineConfig, PipelineContext
from movie_translator.stages.extract_english import ExtractEnglishStage
from movie_translator.subtitles import SubtitleProcessor
from movie_translator.types import DialogueLine


class TestExtractEnglishStage:
    def _make_ctx(self, tmp_path, fetched_eng=None, reference=None):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)
        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        if fetched_eng:
            ctx.fetched_subtitles = {
                'eng': [FetchedSubtitle(path=fetched_eng, source='opensubtitles')]
            }
        else:
            ctx.fetched_subtitles = {}
        ctx.reference_path = reference
        return ctx

    def test_prefers_fetched_english(self, tmp_path):
        fetched = tmp_path / 'fetched.srt'
        fetched.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ref = tmp_path / 'ref.srt'
        ref.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = self._make_ctx(tmp_path, fetched_eng=fetched, reference=ref)

        lines = [DialogueLine(1000, 2000, 'Hello')]
        with patch('movie_translator.stages.extract_english.SubtitleProcessor') as MockProc:
            MockProc.extract_dialogue_lines.return_value = lines
            result = ExtractEnglishStage().run(ctx)

        assert result.english_source == fetched
        assert result.dialogue_lines == lines

    def test_falls_back_to_reference(self, tmp_path):
        ref = tmp_path / 'ref.srt'
        ref.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = self._make_ctx(tmp_path, reference=ref)

        lines = [DialogueLine(1000, 2000, 'Hello')]
        with patch('movie_translator.stages.extract_english.SubtitleProcessor') as MockProc:
            MockProc.extract_dialogue_lines.return_value = lines
            result = ExtractEnglishStage().run(ctx)

        assert result.english_source == ref

    def test_no_source_raises(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        with (
            patch('movie_translator.stages.extract_english.SubtitleExtractor') as MockExtractor,
            patch(
                'movie_translator.stages.extract_english.is_vision_ocr_available',
                return_value=False,
            ),
        ):
            mock_ext = MagicMock()
            mock_ext.get_track_info.return_value = {'tracks': []}
            mock_ext.find_english_track.return_value = None
            MockExtractor.return_value = mock_ext

            with pytest.raises(RuntimeError, match='No English subtitle source'):
                ExtractEnglishStage().run(ctx)


class TestExtractEnglishRunIo:
    def test_run_io_uses_fetched_source(self, tmp_path):
        """Fetched English source used, no pending_ocr."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)
        eng_sub = tmp_path / 'eng.srt'
        eng_sub.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        ctx.fetched_subtitles = {'eng': [FetchedSubtitle(path=eng_sub, source='test')]}

        with patch.object(
            SubtitleProcessor,
            'extract_dialogue_lines',
            return_value=[DialogueLine(1000, 2000, 'Hello')],
        ):
            result = ExtractEnglishStage().run_io(ctx)

        assert result.english_source == eng_sub
        assert result.pending_ocr is None
        assert result.dialogue_lines is not None and len(result.dialogue_lines) == 1

    def test_run_io_defers_ocr_when_no_source(self, tmp_path):
        """No sources available, sets pending_ocr instead of raising."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)
        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        ctx.fetched_subtitles = {}

        with (
            patch('movie_translator.stages.extract_english.SubtitleExtractor') as MockExtractor,
            patch(
                'movie_translator.stages.extract_english.is_vision_ocr_available',
                return_value=True,
            ),
        ):
            mock_ext = MagicMock()
            mock_ext.get_track_info.return_value = {'tracks': []}
            mock_ext.find_english_track.return_value = None
            MockExtractor.return_value = mock_ext

            result = ExtractEnglishStage().run_io(ctx)

        assert result.pending_ocr is not None
        assert result.pending_ocr['type'] == 'burned_in'
        assert result.pending_ocr['track_id'] is None
        assert result.pending_ocr['output_dir'] == str(work)
        assert result.english_source is None
        assert result.dialogue_lines is None
