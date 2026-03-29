from unittest.mock import patch

from movie_translator.context import (
    FontInfo,
    OriginalTrack,
    PipelineConfig,
    PipelineContext,
)
from movie_translator.stages.mux import MuxStage
from movie_translator.types import SubtitleFile


class TestMuxStage:
    def _make_ctx(self, tmp_path, dry_run=False):
        video = tmp_path / 'ep01.mkv'
        video.write_text('fake video')
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)

        pol_ass = tmp_path / 'pol.ass'
        pol_ass.touch()

        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(dry_run=dry_run),
        )
        ctx.subtitle_tracks = [SubtitleFile(pol_ass, 'pol', 'Polish (AI)', is_default=True)]
        ctx.font_info = FontInfo(supports_polish=True)
        ctx.original_english_track = OriginalTrack(
            stream_index=2,
            subtitle_index=0,
            codec='subrip',
            language='eng',
        )
        return ctx

    def test_passes_original_track_to_mux(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            # Simulate output file creation
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            assert call_kwargs.kwargs.get('original_sub_index') == 0
            assert call_kwargs.kwargs.get('original_sub_title') == 'English (Original)'

    def test_dry_run_does_not_replace_original(self, tmp_path):
        ctx = self._make_ctx(tmp_path, dry_run=True)

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

        # Original should still contain 'fake video'
        assert ctx.video_path.read_text() == 'fake video'

    def test_no_original_track_passes_none(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.original_english_track = None

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            assert call_kwargs.kwargs.get('original_sub_index') is None
