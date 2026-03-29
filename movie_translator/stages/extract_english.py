"""Determine English subtitle source and extract dialogue lines."""

from ..context import PipelineContext
from ..logging import logger
from ..ocr import (
    extract_burned_in_subtitles,
    is_vision_ocr_available,
    probe_for_burned_in_subtitles,
)
from ..subtitles import SubtitleExtractor, SubtitleProcessor


class ExtractEnglishStage:
    name = 'extract'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Priority: fetched English > reference > embedded > OCR
        fetched_eng = None
        if ctx.fetched_subtitles:
            fetched_eng_sub = ctx.fetched_subtitles.get('eng')
            if fetched_eng_sub:
                fetched_eng = fetched_eng_sub.path

        if fetched_eng:
            ctx.english_source = fetched_eng
        elif ctx.reference_path:
            ctx.english_source = ctx.reference_path
        else:
            ctx.english_source = self._extract_from_video(ctx)

        if ctx.english_source is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
        if not ctx.dialogue_lines:
            raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')

        logger.info(f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)')
        return ctx

    def _extract_from_video(self, ctx):
        extractor = SubtitleExtractor()
        track_info = extractor.get_track_info(ctx.video_path)
        if not track_info:
            return None

        eng_track = extractor.find_english_track(track_info)
        if eng_track:
            subtitle_ext = extractor.get_subtitle_extension(eng_track)
            output = ctx.work_dir / f'{ctx.video_path.stem}_extracted{subtitle_ext}'
            subtitle_index = eng_track.get('subtitle_index', 0)
            extractor.extract_subtitle(ctx.video_path, eng_track['id'], output, subtitle_index)
            return output

        # OCR fallback
        if is_vision_ocr_available() and probe_for_burned_in_subtitles(ctx.video_path):
            result = extract_burned_in_subtitles(ctx.video_path, ctx.work_dir)
            if result:
                ctx.ocr_results = result.ocr_results
                return result.srt_path

        return None
