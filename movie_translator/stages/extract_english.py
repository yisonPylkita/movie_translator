"""Determine English subtitle source and extract dialogue lines."""

from pathlib import Path

from ..context import PendingOcr, PipelineContext
from ..logging import logger
from ..ocr import is_vision_ocr_available
from ..subtitles import SubtitleExtractor, SubtitleProcessor

_IMAGE_CODECS = ('hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle')


class ExtractEnglishStage:
    name = 'extract'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Select the best English subtitle source and extract dialogue lines.

        Priority: fetched English > reference > embedded text track > defer OCR.
        Image-based tracks and burned-in subtitles set ctx.pending_ocr for
        the pipeline to resolve.
        """
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
                # Try embedded text track, defer OCR if needed
                ctx.english_source = self._extract_text_only(ctx)
                if ctx.english_source is not None:
                    s.detail('source', 'embedded')
                elif not ctx.burned_in_probed and is_vision_ocr_available():
                    ctx.pending_ocr = PendingOcr(
                        type='burned_in',
                        output_dir=ctx.work_dir,
                    )
                    s.detail('source', 'deferred_ocr')
                    return ctx  # Can't extract lines yet

        if ctx.english_source is None and ctx.pending_ocr is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        if ctx.english_source is not None:
            with ctx.metrics.span('extract_dialogue_lines') as s:
                ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
                if not ctx.dialogue_lines:
                    raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')
                s.detail('lines', len(ctx.dialogue_lines))
                s.detail('chars', sum(len(line.text) for line in ctx.dialogue_lines))

            logger.info(
                f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)'
            )

        return ctx

    def _extract_text_only(self, ctx: PipelineContext) -> Path | None:
        """Try to find and extract a text-based (non-image) English track.

        Returns the output path or None if no text track is found.
        """
        extractor = SubtitleExtractor()
        track_info = extractor.get_track_info(ctx.video_path)
        if not track_info:
            return None

        eng_track = extractor.find_english_track(track_info)
        if eng_track:
            codec = eng_track.get('codec', '').lower()
            is_image = any(codec == c or codec.startswith(c) for c in _IMAGE_CODECS)
            if not is_image:
                with ctx.metrics.span('extract_subtitle'):
                    subtitle_ext = extractor.get_subtitle_extension(eng_track)
                    output = ctx.work_dir / f'{ctx.video_path.stem}_extracted{subtitle_ext}'
                    subtitle_index = eng_track.get('subtitle_index', 0)
                    extractor.extract_subtitle(
                        ctx.video_path, eng_track['id'], output, subtitle_index
                    )
                    return output
        return None
