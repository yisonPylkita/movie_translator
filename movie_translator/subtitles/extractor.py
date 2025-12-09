"""Subtitle extraction from MKV files."""

import json
import subprocess
from pathlib import Path
from typing import Any

from ..utils import log_error, log_info, log_success, log_warning


class SubtitleExtractor:
    """Handles extraction of subtitle tracks from MKV files."""

    # Keywords indicating non-dialogue tracks
    SIGNS_KEYWORDS = ('sign', 'song', 'title', 'op', 'ed')
    TEXT_CODECS = ('substationalpha', 'ass', 'ssa')
    IMAGE_CODECS = ('hdmv pgs', 'pgs')

    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr

    def get_track_info(self, mkv_path: Path) -> dict[str, Any]:
        """Get track information from MKV file."""
        try:
            result = subprocess.run(
                ['mkvmerge', '-J', str(mkv_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            log_error(f'Failed to get track info from {mkv_path.name}: {e}')
            return {}

    def find_english_track(self, track_info: dict[str, Any]) -> dict[str, Any] | None:
        """Find the best English subtitle track from available tracks."""
        english_tracks = self._get_english_tracks(track_info)

        if not english_tracks:
            return None
        if len(english_tracks) == 1:
            return english_tracks[0]

        return self._select_best_track(english_tracks)

    def _get_english_tracks(self, track_info: dict[str, Any]) -> list[dict]:
        """Filter tracks to only English subtitle tracks."""
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
        """Select the best track from multiple English tracks."""
        dialogue_tracks, signs_tracks = self._categorize_tracks(english_tracks)

        # Try dialogue tracks first
        if dialogue_tracks:
            result = self._select_from_dialogue_tracks(dialogue_tracks, len(english_tracks))
            if result:
                return result

        # Fall back to signs/songs tracks
        return self._select_from_signs_tracks(signs_tracks, english_tracks)

    def _categorize_tracks(self, tracks: list[dict]) -> tuple[list[dict], list[dict]]:
        """Categorize tracks into dialogue and signs/songs."""
        dialogue_tracks = []
        signs_tracks = []

        for track in tracks:
            props = track.get('properties', {})
            track_name = props.get('track_name', '').lower()

            if any(keyword in track_name for keyword in self.SIGNS_KEYWORDS):
                signs_tracks.append(track)
            else:
                dialogue_tracks.append(track)

        return dialogue_tracks, signs_tracks

    def _select_from_dialogue_tracks(
        self, dialogue_tracks: list[dict], total_count: int
    ) -> dict | None:
        """Select best track from dialogue tracks."""
        text_tracks, image_tracks = self._separate_by_codec(dialogue_tracks)

        if text_tracks:
            log_info(f'Found {total_count} English tracks, selected text-based dialogue track')
            return text_tracks[0]

        if image_tracks:
            return self._handle_image_tracks(image_tracks, total_count)

        if dialogue_tracks:
            log_info(f'Found {total_count} English tracks, selected dialogue track')
            return dialogue_tracks[0]

        return None

    def _select_from_signs_tracks(
        self, signs_tracks: list[dict], english_tracks: list[dict]
    ) -> dict | None:
        """Select from signs/songs tracks as fallback."""
        text_signs, image_signs = self._separate_by_codec(signs_tracks)

        if text_signs:
            log_info(
                f'Found {len(english_tracks)} English tracks, '
                'using text-based signs/songs as fallback (no dialogue tracks)'
            )
            return text_signs[0]

        if image_signs:
            log_warning(
                f'Found {len(english_tracks)} English tracks, '
                'but only image-based signs/songs available (no text extraction possible)'
            )
            return None

        # Try non-forced tracks
        non_forced = [
            t for t in english_tracks if not t.get('properties', {}).get('forced_track', False)
        ]
        if non_forced:
            log_info(f'Found {len(english_tracks)} English tracks, selected non-forced track')
            return non_forced[0]

        log_warning(f'Found {len(english_tracks)} English tracks, all appear to be signs/songs')
        return english_tracks[0]

    def _separate_by_codec(self, tracks: list[dict]) -> tuple[list[dict], list[dict]]:
        """Separate tracks into text-based and image-based."""
        text_tracks = []
        image_tracks = []

        for track in tracks:
            codec = track.get('codec', '').lower()
            if any(c in codec for c in self.TEXT_CODECS):
                text_tracks.append(track)
            elif any(c in codec for c in self.IMAGE_CODECS):
                image_tracks.append(track)

        return text_tracks, image_tracks

    def _handle_image_tracks(self, image_tracks: list[dict], total_count: int) -> dict | None:
        """Handle image-based tracks, potentially with OCR."""
        if not self.enable_ocr:
            log_warning(
                f'Found {total_count} English tracks, but only image-based dialogue tracks available'
            )
            log_info('Enable OCR with --enable-ocr flag')
            return None

        # Check OCR availability (lazy import to avoid dependency issues)
        from ..ocr import SubtitleOCR

        ocr_check = SubtitleOCR()

        if ocr_check.check_availability():
            log_info(
                f'Found {total_count} English tracks, will process image-based dialogue with OCR'
            )
            image_tracks[0]['requires_ocr'] = True
            ocr_check.cleanup()
            return image_tracks[0]

        log_warning(
            f'Found {total_count} English tracks, but only image-based dialogue tracks available'
        )
        log_info('Install OCR support with: uv add opencv-python paddleocr')
        ocr_check.cleanup()
        return None

    def get_subtitle_extension(self, track: dict[str, Any]) -> str:
        """Get the appropriate subtitle file extension for a track."""
        props = track.get('properties', {})
        codec_id = props.get('codec_id', '')
        codec = track.get('codec', '').lower()
        combined = f'{codec_id} {codec}'.lower()

        if 'ass' in combined or 's_text/ass' in combined or 'substationalpha' in combined:
            return '.ass'
        elif 'ssa' in combined or 's_text/ssa' in combined:
            return '.ssa'
        return '.srt'

    def extract_subtitle(self, mkv_path: Path, track_id: int, output_path: Path) -> bool:
        """Extract subtitle track from MKV file."""
        log_info(f'Extracting subtitle track {track_id}...')

        cmd = ['mkvextract', 'tracks', str(mkv_path), f'{track_id}:{output_path}']

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            log_success(f'Extraction successful: {output_path.name}')
            return True
        except subprocess.CalledProcessError as e:
            log_error(f'Failed to extract subtitle track {track_id}: {e}')
            if e.stderr:
                log_error(f'stderr: {e.stderr}')
            return False
