from pathlib import Path
from typing import Any

from ..ffmpeg import get_video_info, mux_video_with_subtitles
from ..logging import logger
from ..types import SubtitleFile


class VideoOperationError(Exception):
    pass


class VideoOperations:
    def create_clean_video(
        self,
        original_video: Path,
        english_ass: Path,
        polish_ass: Path,
        output_video: Path,
    ) -> None:
        logger.info(f'🎬 Creating clean video: {output_video.name}')
        logger.info('   - Adding: Polish (AI) + English dialogue (Polish as default)')

        subtitle_files = [
            SubtitleFile(polish_ass, 'pol', 'Polish (AI)', is_default=True),
            SubtitleFile(english_ass, 'eng', 'English Dialogue', is_default=False),
        ]

        mux_video_with_subtitles(
            original_video,
            subtitle_files,
            output_video,
        )

        logger.info('   - Clean video merge successful')

        if output_video.exists() and output_video.stat().st_size > 0:
            size_mb = output_video.stat().st_size / 1024 / 1024
            logger.info(f'   - Output size: {size_mb:.1f} MB')

    def verify_result(self, output_video: Path) -> None:
        logger.info(f'🔍 Verifying result: {output_video.name}')

        if not output_video.exists():
            raise VideoOperationError(f'Output video not found: {output_video}')

        info = get_video_info(output_video)
        subtitle_tracks = self._get_subtitle_tracks(info)

        logger.info(f'   - Found {len(subtitle_tracks)} subtitle tracks:')
        for track in subtitle_tracks:
            logger.info(f'     * Track {track["index"]}: {track["title"]} ({track["language"]})')

        self._validate_track_order(subtitle_tracks)

    def _get_subtitle_tracks(self, video_info: dict[str, Any]) -> list[dict[str, Any]]:
        streams = video_info.get('streams', [])
        subtitle_tracks = []

        for stream in streams:
            if stream.get('codec_type') == 'subtitle':
                tags = stream.get('tags', {})
                subtitle_tracks.append(
                    {
                        'index': stream.get('index'),
                        'language': tags.get('language', 'unknown'),
                        'title': tags.get('title', 'unnamed'),
                    }
                )

        return subtitle_tracks

    def _validate_track_order(self, subtitle_tracks: list[dict[str, Any]]) -> None:
        if len(subtitle_tracks) != 2:
            raise VideoOperationError(f'Expected 2 subtitle tracks, found {len(subtitle_tracks)}')

        polish_first = subtitle_tracks[0]['language'] == 'pol'
        english_second = subtitle_tracks[1]['language'] == 'eng'

        if polish_first and english_second:
            logger.info('   ✅ Perfect! Polish (AI) as default track + English dialogue')
            return

        raise VideoOperationError(
            f'Incorrect track order. Expected: Polish first, English second. '
            f'Found: Track 1={subtitle_tracks[0]["language"]}, '
            f'Track 2={subtitle_tracks[1]["language"]}'
        )
