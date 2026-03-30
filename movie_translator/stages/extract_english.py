"""Determine English subtitle source and extract dialogue lines."""

from ..context import PipelineContext
from ..logging import logger
from ..ocr import (
    extract_burned_in_subtitles,
    is_vision_ocr_available,
    probe_for_burned_in_subtitles,
)
from ..ocr.pgs_extractor import extract_pgs_track
from ..subtitles import SubtitleExtractor, SubtitleProcessor

_IMAGE_CODECS = ('hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle')


class ExtractEnglishStage:
    name = 'extract'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Priority: fetched English > reference > embedded > OCR
        with ctx.metrics.span('select_source') as s:
            fetched_eng = None
            if ctx.fetched_subtitles:
                eng_subs = ctx.fetched_subtitles.get('eng')
                if eng_subs:
                    fetched_eng = eng_subs[0].path

            if fetched_eng:
                ctx.english_source = fetched_eng
                s.detail('source', 'fetched')
            elif ctx.reference_path:
                ctx.english_source = ctx.reference_path
                s.detail('source', 'reference')
            else:
                ctx.english_source = self._extract_from_video(ctx)
                if ctx.english_source is not None:
                    source_label = 'ocr' if ctx.ocr_results else 'embedded'
                    s.detail('source', source_label)

        if ctx.english_source is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        with ctx.metrics.span('extract_dialogue_lines') as s:
            ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
            if not ctx.dialogue_lines:
                raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')
            s.detail('lines', len(ctx.dialogue_lines))
            s.detail('chars', sum(len(line.text) for line in ctx.dialogue_lines))

        logger.info(f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)')
        return ctx

    def _extract_from_video(self, ctx):
        extractor = SubtitleExtractor()
        track_info = extractor.get_track_info(ctx.video_path)
        if not track_info:
            return None

        eng_track = extractor.find_english_track(track_info)
        if eng_track:
            codec = eng_track.get('codec', '').lower()
            is_image = any(codec == c or codec.startswith(c) for c in _IMAGE_CODECS)

            if is_image:
                # PGS/DVD — extract via OCR
                srt = extract_pgs_track(ctx.video_path, eng_track['id'], ctx.work_dir)
                if srt:
                    return srt
            else:
                subtitle_ext = extractor.get_subtitle_extension(eng_track)
                output = ctx.work_dir / f'{ctx.video_path.stem}_extracted{subtitle_ext}'
                subtitle_index = eng_track.get('subtitle_index', 0)
                extractor.extract_subtitle(ctx.video_path, eng_track['id'], output, subtitle_index)
                return output

        # OCR fallback — only probe if the reference stage didn't already
        if not ctx.burned_in_probed and is_vision_ocr_available():
            ctx.burned_in_probed = True
            if probe_for_burned_in_subtitles(ctx.video_path):
                result = extract_burned_in_subtitles(ctx.video_path, ctx.work_dir)
                if result:
                    ctx.ocr_results = result.ocr_results
                    return result.srt_path

        return None
