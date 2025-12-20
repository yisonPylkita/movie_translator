import subprocess
from pathlib import Path
from typing import Any

from ..ffmpeg import get_ffmpeg, get_video_info
from ..logging import logger
from ..types import NON_DIALOGUE_STYLES


class SubtitleExtractionError(Exception):
    pass


class SubtitleExtractor:
    TEXT_CODECS = ('ass', 'ssa', 'subrip', 'srt', 'webvtt', 'mov_text')
    IMAGE_CODECS = ('hdmv_pgs_subtitle', 'dvd_subtitle', 'dvb_subtitle')

    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr

    def get_track_info(self, video_path: Path) -> dict[str, Any]:
        if not video_path.exists():
            raise SubtitleExtractionError(f'Video file not found: {video_path}')

        info = get_video_info(video_path)
        return self._convert_ffprobe_info(info)

    def _convert_ffprobe_info(self, ffprobe_info: dict[str, Any]) -> dict[str, Any]:
        streams = ffprobe_info.get('streams', [])
        tracks = []
        subtitle_index = 0

        for stream in streams:
            codec_type = stream.get('codec_type', '')
            if codec_type == 'subtitle':
                track = {
                    'id': stream.get('index'),
                    'type': 'subtitles',
                    'codec': stream.get('codec_name', ''),
                    'properties': {
                        'language': stream.get('tags', {}).get('language', 'und'),
                        'track_name': stream.get('tags', {}).get('title', ''),
                        'codec_id': stream.get('codec_name', ''),
                        'forced_track': stream.get('disposition', {}).get('forced', 0) == 1,
                    },
                    'subtitle_index': subtitle_index,
                }
                tracks.append(track)
                subtitle_index += 1

        return {'tracks': tracks}

    def has_polish_subtitles(self, video_path: Path) -> bool:
        track_info = self.get_track_info(video_path)
        tracks = track_info.get('tracks', [])

        for track in tracks:
            if track.get('type') == 'subtitles':
                props = track.get('properties', {})
                lang = props.get('language', '')
                if lang in ('pol', 'pl'):
                    return True

        return False

    def find_english_track(self, track_info: dict[str, Any]) -> dict[str, Any] | None:
        english_tracks = self._get_english_tracks(track_info)

        if not english_tracks:
            return None
        if len(english_tracks) == 1:
            return english_tracks[0]

        return self._select_best_track(english_tracks)

    def _get_english_tracks(self, track_info: dict[str, Any]) -> list[dict]:
        tracks = track_info.get('tracks', [])
        english_tracks = []

        for track in tracks:
            if track.get('type') == 'subtitles':
                props = track.get('properties', {})
                lang = props.get('language', '')
                if lang in ('eng', 'en'):
                    english_tracks.append(track)

        return english_tracks

    def _select_best_track(self, english_tracks: list[dict]) -> dict | None:
        dialogue_tracks, signs_tracks = self._categorize_tracks(english_tracks)

        if dialogue_tracks:
            result = self._select_from_dialogue_tracks(dialogue_tracks, len(english_tracks))
            if result:
                return result

        return self._select_from_signs_tracks(signs_tracks, english_tracks)

    def _categorize_tracks(self, tracks: list[dict]) -> tuple[list[dict], list[dict]]:
        dialogue_tracks = []
        signs_tracks = []

        for track in tracks:
            props = track.get('properties', {})
            track_name = props.get('track_name', '').lower()

            if any(keyword in track_name for keyword in NON_DIALOGUE_STYLES):
                signs_tracks.append(track)
            else:
                dialogue_tracks.append(track)

        return dialogue_tracks, signs_tracks

    def _select_from_dialogue_tracks(
        self, dialogue_tracks: list[dict], total_count: int
    ) -> dict | None:
        text_tracks, image_tracks = self._separate_by_codec(dialogue_tracks)

        if text_tracks:
            logger.info(f'Found {total_count} English tracks, selected text-based dialogue track')
            return text_tracks[0]

        if image_tracks:
            return self._handle_image_tracks(image_tracks, total_count)

        if dialogue_tracks:
            logger.info(f'Found {total_count} English tracks, selected dialogue track')
            return dialogue_tracks[0]

        return None

    def _select_from_signs_tracks(
        self, signs_tracks: list[dict], english_tracks: list[dict]
    ) -> dict | None:
        text_signs, image_signs = self._separate_by_codec(signs_tracks)

        if text_signs:
            logger.info(
                f'Found {len(english_tracks)} English tracks, '
                'using text-based signs/songs as fallback (no dialogue tracks)'
            )
            return text_signs[0]

        if image_signs:
            logger.warning(
                f'Found {len(english_tracks)} English tracks, '
                'but only image-based signs/songs available (no text extraction possible)'
            )
            return None

        non_forced = [
            t for t in english_tracks if not t.get('properties', {}).get('forced_track', False)
        ]
        if non_forced:
            logger.info(f'Found {len(english_tracks)} English tracks, selected non-forced track')
            return non_forced[0]

        logger.warning(f'Found {len(english_tracks)} English tracks, all appear to be signs/songs')
        return english_tracks[0]

    def _separate_by_codec(self, tracks: list[dict]) -> tuple[list[dict], list[dict]]:
        text_tracks = []
        image_tracks = []

        for track in tracks:
            codec = track.get('codec', '').lower()
            if any(codec == c or codec.startswith(c) for c in self.TEXT_CODECS):
                text_tracks.append(track)
            elif any(codec == c or codec.startswith(c) for c in self.IMAGE_CODECS):
                image_tracks.append(track)
            else:
                text_tracks.append(track)

        return text_tracks, image_tracks

    def _handle_image_tracks(self, image_tracks: list[dict], total_count: int) -> dict | None:
        if not self.enable_ocr:
            logger.warning(
                f'Found {total_count} English tracks, but only image-based dialogue tracks available'
            )
            logger.info('Enable OCR with --enable-ocr flag')
            return None

        from ..ocr import SubtitleOCR

        ocr_check = SubtitleOCR()

        if ocr_check.check_availability():
            logger.info(
                f'Found {total_count} English tracks, will process image-based dialogue with OCR'
            )
            image_tracks[0]['requires_ocr'] = True
            ocr_check.cleanup()
            return image_tracks[0]

        logger.warning(
            f'Found {total_count} English tracks, but only image-based dialogue tracks available'
        )
        logger.info('Install OCR support with: uv add opencv-python paddleocr')
        ocr_check.cleanup()
        return None

    def get_subtitle_extension(self, track: dict[str, Any]) -> str:
        codec = track.get('codec', '').lower()

        if codec in ('ass', 'ssa'):
            return f'.{codec}'
        elif codec in ('subrip', 'srt'):
            return '.srt'
        elif codec == 'webvtt':
            return '.vtt'
        elif codec == 'mov_text':
            return '.srt'
        return '.srt'

    def extract_subtitle(
        self, video_path: Path, track_id: int, output_path: Path, subtitle_index: int | None = None
    ) -> None:
        if not video_path.exists():
            raise SubtitleExtractionError(f'Video file not found: {video_path}')

        logger.info(f'Extracting subtitle track {track_id}...')

        ffmpeg = get_ffmpeg()

        sub_idx = subtitle_index if subtitle_index is not None else 0

        cmd = [
            ffmpeg,
            '-y',
            '-i',
            str(video_path),
            '-map',
            f'0:s:{sub_idx}',
            '-c:s',
            'copy',
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_lines = [
                line
                for line in result.stderr.split('\n')
                if 'error' in line.lower() or 'invalid' in line.lower()
            ]
            error_msg = '; '.join(error_lines) if error_lines else 'Unknown ffmpeg error'
            raise SubtitleExtractionError(
                f'Failed to extract subtitle track {track_id}: {error_msg}'
            )

        logger.info(f'Extraction successful: {output_path.name}')
