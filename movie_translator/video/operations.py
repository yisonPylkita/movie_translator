"""Video file operations using ffmpeg."""

from pathlib import Path

from ..ffmpeg import get_video_info, mux_video_with_subtitles
from ..utils import log_error, log_info, log_success


class VideoOperations:
    """Handles video file creation, merging, and verification."""

    def create_clean_video(
        self,
        original_video: Path,
        english_ass: Path,
        polish_ass: Path,
        output_video: Path,
    ) -> bool:
        """Create clean video with only video/audio + Polish (AI) + English dialogue.

        Works with any ffmpeg-supported video format (MKV, MP4, etc.)
        """
        log_info(f'🎬 Creating clean video: {output_video.name}')
        log_info('   - Adding: Polish (AI) + English dialogue (Polish as default)')

        # Subtitle files: (path, language, title, is_default)
        subtitle_files = [
            (polish_ass, 'pol', 'Polish (AI)', True),
            (english_ass, 'eng', 'English Dialogue', False),
        ]

        try:
            success = mux_video_with_subtitles(
                original_video,
                subtitle_files,
                output_video,
            )

            if success:
                log_success('   - Clean video merge successful')

                if output_video.exists() and output_video.stat().st_size > 0:
                    size_mb = output_video.stat().st_size / 1024 / 1024
                    log_info(f'   - Output size: {size_mb:.1f} MB')

                return True
            else:
                log_error('Failed to merge video')
                return False

        except Exception as e:
            log_error(f'Failed to merge: {e}')
            return False

    def verify_result(self, output_video: Path) -> bool:
        """Verify the clean video has only the desired tracks."""
        log_info(f'🔍 Verifying result: {output_video.name}')

        try:
            info = get_video_info(output_video)
            subtitle_tracks = self._get_subtitle_tracks(info)

            log_info(f'   - Found {len(subtitle_tracks)} subtitle tracks:')
            for track in subtitle_tracks:
                log_info(f'     * Track {track["index"]}: {track["title"]} ({track["language"]})')

            return self._validate_track_order(subtitle_tracks)

        except Exception as e:
            log_error(f'Failed to verify: {e}')
            return False

    def _get_subtitle_tracks(self, video_info: dict) -> list[dict]:
        """Extract subtitle track information from ffprobe output."""
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

    def _validate_track_order(self, subtitle_tracks: list[dict]) -> bool:
        """Validate that tracks are in correct order: Polish first, English second."""
        if len(subtitle_tracks) != 2:
            log_error(f'   ❌ Expected 2 subtitle tracks, found {len(subtitle_tracks)}')
            return False

        polish_first = subtitle_tracks[0]['language'] == 'pol'
        english_second = subtitle_tracks[1]['language'] == 'eng'

        if polish_first and english_second:
            log_success('   ✅ Perfect! Polish (AI) as default track + English dialogue')
            return True

        log_error('   ❌ Incorrect track order. Expected: Polish first, English second')
        log_error(
            f'   ❌ Found: Track 1={subtitle_tracks[0]["language"]}, '
            f'Track 2={subtitle_tracks[1]["language"]}'
        )
        return False
