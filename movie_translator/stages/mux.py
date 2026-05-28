"""Final video muxing stage — combines video with subtitle tracks."""

import os
import shutil
from pathlib import Path

from ..context import PipelineContext
from ..inpainting import remove_burned_in_subtitles
from ..logging import logger
from ..types import SubtitleFile
from ..video import VideoOperations

# Suffix marker for the in-place mux output sitting next to the original.
# Format: <stem>.translating<suffix>  e.g. Episode01.translating.mkv
IN_PLACE_TEMP_MARKER = '.translating'


def in_place_temp_path(video_path: Path) -> Path:
    """Return the sibling path used for the in-place mux temp file."""
    return video_path.with_name(f'{video_path.stem}{IN_PLACE_TEMP_MARKER}{video_path.suffix}')


class MuxStage:
    name = 'mux'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Inpaint burned-in subtitles if OCR was used and inpainting is enabled.
        # OCR is automatic (always runs when needed), but inpainting is opt-in
        # via --inpaint because it's slow (rewrites the entire video).
        source_video = ctx.video_path
        if ctx.ocr_results and ctx.config.enable_inpaint and ctx.inpainted_video is None:
            with ctx.metrics.span('inpaint') as s:
                logger.info('Removing burned-in subtitles via inpainting...')
                s.detail('frames', len(ctx.ocr_results))
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
        assert ctx.subtitle_tracks is not None
        assert ctx.font_info is not None

        # Choose temp location:
        # - in_place=True: sibling of original (same filesystem → atomic os.replace)
        # - in_place=False: inside work_dir (cleaned up later)
        if ctx.config.in_place:
            temp_video = in_place_temp_path(ctx.video_path)
        else:
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp{ctx.video_path.suffix}'

        try:
            with ctx.metrics.span('create_clean_video') as s:
                ops = VideoOperations()
                s.detail('tracks', len(ctx.subtitle_tracks))
                s.detail('font_attachments', len(ctx.font_info.font_attachments or []))
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
            with ctx.metrics.span('verify_result'):
                ops.verify_result(temp_video, expected_tracks=expected_tracks)

            if not ctx.config.dry_run:
                with ctx.metrics.span('replace_original'):
                    if ctx.config.in_place:
                        self._replace_in_place(ctx.video_path, temp_video)
                        # Free inpainted (full-size) copy promptly to honour
                        # the "≤2× peak per worker" disk budget.
                        if ctx.inpainted_video is not None and ctx.inpainted_video.exists():
                            try:
                                ctx.inpainted_video.unlink()
                            except OSError:
                                pass
                    else:
                        self._replace_original(ctx.video_path, temp_video)
        except Exception:
            # Best-effort cleanup of partial temp so it doesn't waste disk.
            # For non-in-place this is a no-op practically (work_dir is cleaned
            # by the caller); for in-place this is essential — the temp sits
            # beside the original.
            if temp_video.exists():
                try:
                    temp_video.unlink()
                except OSError:
                    pass
            raise

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

    def _replace_in_place(self, video_path: Path, temp_video: Path) -> None:
        """Atomic in-place replace: no backup, peak ≤2× original size.

        os.replace is atomic on POSIX when src and dst share a filesystem.
        Temp is verified BEFORE this call, so a separate post-move verify
        (and the backup it would protect) is redundant.
        """
        os.replace(str(temp_video), str(video_path))
