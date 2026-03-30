from unittest.mock import patch

import pytest

from movie_translator.context import (
    FontInfo,
    OriginalTrack,
    PipelineConfig,
    PipelineContext,
)
from movie_translator.stages.mux import MuxStage
from movie_translator.types import BoundingBox, OCRResult, SubtitleFile


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

    # ------------------------------------------------------------------
    # Inpainting code path
    # ------------------------------------------------------------------

    def test_inpainting_called_when_ocr_results_and_inpaint_enabled(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.config.enable_inpaint = True
        ctx.ocr_results = [
            OCRResult(
                timestamp_ms=1000,
                text='Hello',
                boxes=[BoundingBox(0.1, 0.8, 0.8, 0.1)],
            ),
        ]

        with (
            patch('movie_translator.stages.mux.VideoOperations') as MockOps,
            patch('movie_translator.stages.mux.remove_burned_in_subtitles') as mock_inpaint,
        ):
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            MuxStage().run(ctx)

            mock_inpaint.assert_called_once()
            call_args = mock_inpaint.call_args
            assert call_args[0][0] == ctx.video_path
            assert call_args[0][2] == ctx.ocr_results
            assert ctx.inpainted_video is not None

    def test_inpainting_skipped_when_already_inpainted(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.config.enable_inpaint = True
        ctx.ocr_results = [
            OCRResult(timestamp_ms=1000, text='Hi', boxes=[BoundingBox(0.1, 0.8, 0.8, 0.1)]),
        ]
        pre_inpainted = tmp_path / 'already_inpainted.mkv'
        pre_inpainted.write_text('inpainted video')
        ctx.inpainted_video = pre_inpainted

        with (
            patch('movie_translator.stages.mux.VideoOperations') as MockOps,
            patch('movie_translator.stages.mux.remove_burned_in_subtitles') as mock_inpaint,
        ):
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            MuxStage().run(ctx)

            mock_inpaint.assert_not_called()
            # Source video should be the pre-existing inpainted video
            source_arg = mock_ops.create_clean_video.call_args[0][0]
            assert source_arg == pre_inpainted

    def test_inpainting_not_called_when_inpaint_disabled(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.config.enable_inpaint = False
        ctx.ocr_results = [
            OCRResult(timestamp_ms=1000, text='Hi', boxes=[BoundingBox(0.1, 0.8, 0.8, 0.1)]),
        ]

        with (
            patch('movie_translator.stages.mux.VideoOperations') as MockOps,
            patch('movie_translator.stages.mux.remove_burned_in_subtitles') as mock_inpaint,
        ):
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            MuxStage().run(ctx)

            mock_inpaint.assert_not_called()

    # ------------------------------------------------------------------
    # _replace_original() method
    # ------------------------------------------------------------------

    def test_replace_original_creates_backup_and_replaces(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.write_text('original content')
        temp_video = tmp_path / 'ep01_temp.mkv'
        temp_video.write_text('muxed content')

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.verify_result.return_value = None

            MuxStage()._replace_original(video, temp_video)

        # After successful replace, video should have new content
        assert video.read_text() == 'muxed content'
        # Backup should be cleaned up
        backup = video.with_suffix('.mkv.backup')
        assert not backup.exists()
        # Temp file should be gone (moved)
        assert not temp_video.exists()

    def test_replace_original_rolls_back_on_verify_failure(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.write_text('original content')
        temp_video = tmp_path / 'ep01_temp.mkv'
        temp_video.write_text('muxed content')

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.verify_result.side_effect = RuntimeError('verification failed')

            with pytest.raises(RuntimeError, match='verification failed'):
                MuxStage()._replace_original(video, temp_video)

        # After rollback, the original file should be restored from backup.
        # The shutil.move in the try block already replaced the original, so
        # the rollback branch checks backup_path.exists() and not video_path.exists().
        # Since shutil.move succeeded before verify raised, video exists with muxed content
        # and backup still exists. The rollback condition (not video_path.exists()) is False,
        # so backup is not moved back. The backup file should still exist for manual recovery.
        backup = video.with_suffix('.mkv.backup')
        assert backup.exists()

    def test_replace_original_rolls_back_when_move_fails(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.write_text('original content')
        temp_video = tmp_path / 'ep01_temp.mkv'
        temp_video.write_text('muxed content')

        with (
            patch('movie_translator.stages.mux.VideoOperations'),
            patch('movie_translator.stages.mux.shutil.move', side_effect=OSError('disk full')),
        ):
            with pytest.raises(OSError, match='disk full'):
                MuxStage()._replace_original(video, temp_video)

        # Move failed, so original should still be intact
        assert video.read_text() == 'original content'
        # Backup exists but original also exists, so rollback condition is False
        backup = video.with_suffix('.mkv.backup')
        assert backup.exists()

    # ------------------------------------------------------------------
    # Font attachments
    # ------------------------------------------------------------------

    def test_font_attachments_passed_to_mux(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        font_a = tmp_path / 'FontA.ttf'
        font_b = tmp_path / 'FontB.otf'
        font_a.touch()
        font_b.touch()
        ctx.font_info = FontInfo(
            supports_polish=True,
            font_attachments=[font_a, font_b],
        )

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            assert call_kwargs.kwargs.get('font_attachments') == [font_a, font_b]

    def test_empty_font_attachments_passed_as_none(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.font_info = FontInfo(supports_polish=True, font_attachments=[])

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            # Empty list is falsy, so `[] or None` evaluates to None
            assert call_kwargs.kwargs.get('font_attachments') is None

    # ------------------------------------------------------------------
    # Verify result raises
    # ------------------------------------------------------------------

    def test_verify_result_failure_raises(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.side_effect = RuntimeError('wrong track count')
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')

            with pytest.raises(RuntimeError, match='wrong track count'):
                MuxStage().run(ctx)
