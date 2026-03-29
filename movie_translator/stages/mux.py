"""Final video muxing stage — combines video with subtitle tracks."""

import shutil
from pathlib import Path

from ..context import PipelineContext
from ..inpainting import remove_burned_in_subtitles
from ..logging import logger
from ..types import SubtitleFile
from ..video import VideoOperations


class MuxStage:
    name = 'mux'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Use inpainted video if OCR was used
        source_video = ctx.video_path
        if ctx.ocr_results and ctx.inpainted_video is None:
            logger.info('Removing burned-in subtitles...')
            inpainted = ctx.work_dir / f'{ctx.video_path.stem}_inpainted{ctx.video_path.suffix}'
            remove_burned_in_subtitles(
                ctx.video_path,
                inpainted,
                ctx.ocr_results,
                ctx.config.device,
            )
            ctx.inpainted_video = inpainted
            source_video = inpainted
        elif ctx.inpainted_video:
            source_video = ctx.inpainted_video

        # Determine original track preservation
        original_sub_index = None
        original_sub_title = None
        if ctx.original_english_track:
            original_sub_index = ctx.original_english_track.subtitle_index
            original_sub_title = 'English (Original)'

        # Mux
        temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp{ctx.video_path.suffix}'
        ops = VideoOperations()
        ops.create_clean_video(
            source_video,
            ctx.subtitle_tracks,
            temp_video,
            font_attachments=ctx.font_info.font_attachments or None,
            original_sub_index=original_sub_index,
            original_sub_title=original_sub_title,
        )

        # Build full expected track list including preserved original
        expected_tracks = list(ctx.subtitle_tracks)
        if original_sub_index is not None:
            lang = ctx.original_english_track.language if ctx.original_english_track else 'eng'
            expected_tracks.insert(
                0,
                SubtitleFile(
                    path=Path(),  # placeholder, only count and language are checked
                    language=lang,
                    title=original_sub_title or 'English (Original)',
                    is_default=False,
                ),
            )
        ops.verify_result(temp_video, expected_tracks=expected_tracks)

        if not ctx.config.dry_run:
            self._replace_original(ctx.video_path, temp_video)

        return ctx

    def _replace_original(self, video_path, temp_video):
        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)
        try:
            shutil.move(str(temp_video), str(video_path))
            ops = VideoOperations()
            ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
