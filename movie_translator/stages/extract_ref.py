"""Extract reference subtitle and record original English track info."""

from ..context import OriginalTrack, PendingOcr, PipelineContext
from ..logging import logger
from ..ocr import is_vision_ocr_available
from ..subtitles import SubtitleExtractor

# PGS/DVD/DVB image-based codecs that need OCR extraction
_IMAGE_CODECS = ('hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle')


def _is_image_codec(track: dict) -> bool:
    codec = track.get('codec', '').lower()
    return any(codec == c or codec.startswith(c) for c in _IMAGE_CODECS)


class ExtractReferenceStage:
    name = 'extract_reference'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Extract the English reference subtitle track.

        Text-based tracks are extracted directly. Image-based (PGS/DVD)
        and burned-in tracks set ctx.pending_ocr for the pipeline to
        resolve (synchronously or via GPU queue).
        """
        extractor = SubtitleExtractor()
        ref_dir = ctx.work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        with ctx.metrics.span('get_track_info'):
            track_info = extractor.get_track_info(ctx.video_path)
            eng_track = extractor.find_english_track(track_info) if track_info else None

        if eng_track:
            ctx.original_english_track = OriginalTrack(
                stream_index=eng_track['id'],
                subtitle_index=eng_track.get('subtitle_index', 0),
                codec=eng_track.get('codec', 'unknown'),
                language=eng_track.get('properties', {}).get('language', 'eng'),
            )

            if _is_image_codec(eng_track):
                # Defer PGS/DVD OCR
                ctx.pending_ocr = PendingOcr(
                    type='pgs',
                    track_id=eng_track['id'],
                    output_dir=ref_dir,
                )
            else:
                with ctx.metrics.span('extract_subtitle'):
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

        # If no track found and Vision is available, defer burned-in OCR
        if ctx.reference_path is None and ctx.pending_ocr is None and is_vision_ocr_available():
            ctx.pending_ocr = PendingOcr(
                type='burned_in',
                output_dir=ref_dir,
            )

        return ctx
