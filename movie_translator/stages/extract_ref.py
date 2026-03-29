"""Extract reference subtitle and record original English track info."""

from ..context import OriginalTrack, PipelineContext
from ..logging import logger
from ..ocr import (
    extract_burned_in_subtitles,
    is_vision_ocr_available,
    probe_for_burned_in_subtitles,
)
from ..subtitles import SubtitleExtractor


class ExtractReferenceStage:
    name = 'extract_reference'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        extractor = SubtitleExtractor()
        ref_dir = ctx.work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        track_info = extractor.get_track_info(ctx.video_path)
        eng_track = extractor.find_english_track(track_info) if track_info else None

        if eng_track:
            # Record original track info for preservation in mux stage
            ctx.original_english_track = OriginalTrack(
                stream_index=eng_track['id'],
                subtitle_index=eng_track.get('subtitle_index', 0),
                codec=eng_track.get('codec', 'unknown'),
                language=eng_track.get('properties', {}).get('language', 'eng'),
            )

            subtitle_ext = extractor.get_subtitle_extension(eng_track)
            ref_path = ref_dir / f'{ctx.video_path.stem}_reference{subtitle_ext}'
            try:
                extractor.extract_subtitle(
                    ctx.video_path,
                    eng_track['id'],
                    ref_path,
                    eng_track.get('subtitle_index', 0),
                )
                ctx.reference_path = ref_path
                logger.info(f'Extracted reference: {ref_path.name}')
            except Exception as e:
                logger.warning(f'Failed to extract reference: {e}')

        # Fall back to OCR if no embedded track
        if ctx.reference_path is None and is_vision_ocr_available():
            if probe_for_burned_in_subtitles(ctx.video_path):
                try:
                    result = extract_burned_in_subtitles(ctx.video_path, ref_dir)
                    if result:
                        ctx.reference_path = result.srt_path
                        ctx.ocr_results = result.ocr_results
                        logger.info(f'Extracted OCR reference: {result.srt_path.name}')
                except Exception as e:
                    logger.warning(f'OCR extraction failed: {e}')

        return ctx
