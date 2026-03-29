from unittest.mock import patch

from movie_translator.context import (
    FetchedSubtitle,
    FontInfo,
    PipelineConfig,
    PipelineContext,
)
from movie_translator.stages.create_tracks import CreateTracksStage
from movie_translator.types import DialogueLine


class TestCreateTracksStage:
    def _make_ctx(self, tmp_path, fetched_pol=None):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        eng_src = tmp_path / 'eng.srt'
        eng_src.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)

        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        ctx.english_source = eng_src
        ctx.dialogue_lines = [DialogueLine(1000, 2000, 'Hello')]
        ctx.translated_lines = [DialogueLine(1000, 2000, 'Cześć')]
        ctx.font_info = FontInfo(supports_polish=True)

        if fetched_pol:
            ctx.fetched_subtitles = {'pol': fetched_pol}
        else:
            ctx.fetched_subtitles = {}
        return ctx

    def test_creates_ai_polish_track(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as _MockProc:
            result = CreateTracksStage().run(ctx)

        assert result.subtitle_tracks is not None
        titles = [t.title for t in result.subtitle_tracks]
        assert 'Polish (AI)' in titles

    def test_fetched_polish_includes_source(self, tmp_path):
        pol_file = tmp_path / 'pol.ass'
        pol_file.write_text('[Script Info]\n\n[Events]\n')
        ctx = self._make_ctx(tmp_path, fetched_pol=FetchedSubtitle(pol_file, 'animesub'))

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as _MockProc:
            result = CreateTracksStage().run(ctx)

        titles = [t.title for t in result.subtitle_tracks]
        assert 'Polish (animesub)' in titles
        assert 'Polish (AI)' in titles

    def test_fetched_polish_is_default(self, tmp_path):
        pol_file = tmp_path / 'pol.ass'
        pol_file.write_text('[Script Info]\n\n[Events]\n')
        ctx = self._make_ctx(tmp_path, fetched_pol=FetchedSubtitle(pol_file, 'podnapisi'))

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as _MockProc:
            result = CreateTracksStage().run(ctx)

        defaults = [t for t in result.subtitle_tracks if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].title == 'Polish (podnapisi)'

    def test_ai_is_default_when_no_fetched(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as _MockProc:
            result = CreateTracksStage().run(ctx)

        defaults = [t for t in result.subtitle_tracks if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].title == 'Polish (AI)'
