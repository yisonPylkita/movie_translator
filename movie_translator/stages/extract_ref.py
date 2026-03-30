"""Extract reference subtitle and record original English track info."""

from pathlib import Path

from ..context import OriginalTrack, PipelineContext
from ..logging import logger
from ..ocr import (
    extract_burned_in_subtitles,
    is_vision_ocr_available,
    probe_for_burned_in_subtitles,
)
from ..ocr.pgs_extractor import extract_pgs_track
from ..subtitles import SubtitleExtractor

# PGS/DVD/DVB image-based codecs that need OCR extraction
_IMAGE_CODECS = ('hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle')


def _is_image_codec(track: dict) -> bool:
    codec = track.get('codec', '').lower()
    return any(codec == c or codec.startswith(c) for c in _IMAGE_CODECS)


def _count_srt_entries(path: Path) -> int:
    """Count numbered entries in an SRT file."""
    try:
        return sum(1 for line in path.read_text().splitlines() if line.strip().isdigit())
    except Exception:
        return 0


class ExtractReferenceStage:
    name = 'extract_reference'

    def run(self, ctx: PipelineContext) -> PipelineContext:
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
                # PGS/DVD bitmap track — extract via OCR
                with ctx.metrics.span('extract_pgs_track') as s:
                    srt_path = extract_pgs_track(ctx.video_path, eng_track['id'], ref_dir)
                    if srt_path:
                        ctx.reference_path = srt_path
                        logger.info(f'Extracted PGS reference via OCR: {srt_path.name}')
                        s.detail('subtitle_count', _count_srt_entries(srt_path))
            else:
                # Text-based track — extract directly
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

        # Fall back to burned-in subtitle OCR if no track at all
        if ctx.reference_path is None and is_vision_ocr_available():
            with ctx.metrics.span('probe_burned_in') as s:
                ctx.burned_in_probed = True
                detected = probe_for_burned_in_subtitles(ctx.video_path)
                s.detail('detected', detected)
            if detected:
                with ctx.metrics.span('extract_burned_in'):
                    try:
                        result = extract_burned_in_subtitles(ctx.video_path, ref_dir)
                        if result:
                            ctx.reference_path = result.srt_path
                            ctx.ocr_results = result.ocr_results
                            logger.info(f'Extracted OCR reference: {result.srt_path.name}')
                    except Exception as e:
                        logger.warning(f'OCR extraction failed: {e}')

        return ctx
