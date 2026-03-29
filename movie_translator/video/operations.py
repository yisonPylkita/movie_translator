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
        subtitle_files: list[SubtitleFile],
        output_video: Path,
        font_attachments: list[Path] | None = None,
        original_sub_index: int | None = None,
        original_sub_title: str | None = None,
    ) -> None:
        logger.info(f'🎬 Creating clean video: {output_video.name}')
        track_desc = ', '.join(f'{s.title} ({s.language})' for s in subtitle_files)
        if original_sub_title:
            track_desc = f'{original_sub_title}, {track_desc}'
        logger.info(f'   - Adding: {track_desc}')

        mux_video_with_subtitles(
            original_video,
            subtitle_files,
            output_video,
            font_attachments=font_attachments,
            original_sub_index=original_sub_index,
            original_sub_title=original_sub_title,
        )

        logger.info('   - Clean video merge successful')

        if output_video.exists() and output_video.stat().st_size > 0:
            size_mb = output_video.stat().st_size / 1024 / 1024
            logger.info(f'   - Output size: {size_mb:.1f} MB')

    def verify_result(
        self, output_video: Path, expected_tracks: list[SubtitleFile] | None = None
    ) -> None:
        logger.info(f'🔍 Verifying result: {output_video.name}')

        if not output_video.exists():
            raise VideoOperationError(f'Output video not found: {output_video}')

        info = get_video_info(output_video)
        subtitle_tracks = self._get_subtitle_tracks(info)

        logger.info(f'   - Found {len(subtitle_tracks)} subtitle tracks:')
        for track in subtitle_tracks:
            logger.debug(f'     * Track {track["index"]}: {track["title"]} ({track["language"]})')

        if expected_tracks is not None:
            self._validate_tracks(subtitle_tracks, expected_tracks)

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

    def _validate_tracks(self, actual: list[dict[str, Any]], expected: list[SubtitleFile]) -> None:
        if len(actual) != len(expected):
            raise VideoOperationError(
                f'Expected {len(expected)} subtitle tracks, found {len(actual)}'
            )

        for i, (track, exp) in enumerate(zip(actual, expected, strict=True)):
            if track['language'] != exp.language:
                raise VideoOperationError(
                    f'Track {i + 1}: expected language "{exp.language}", '
                    f'found "{track["language"]}"'
                )

        track_names = ', '.join(f'{t["title"]} ({t["language"]})' for t in actual)
        logger.info(f'   ✅ Verified: {track_names}')
