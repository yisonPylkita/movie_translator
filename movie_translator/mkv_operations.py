#!/usr/bin/env python3
"""
MKV file operations for the Movie Translator pipeline.
Handles MKV creation, merging, and verification.
"""

import json
import subprocess
from pathlib import Path

from .utils import log_error, log_info, log_success


def create_clean_mkv(original_mkv: Path, english_ass: Path, polish_ass: Path, output_mkv: Path):
    """Create clean MKV with only video/audio + Polish (AI) + English dialogue (Polish as default)."""
    log_info(f'🎬 Creating clean MKV: {output_mkv.name}')
    log_info('   - Adding: Polish (AI) + English dialogue (Polish as default)')

    cmd = [
        'mkvmerge',
        '-o',
        str(output_mkv),
        '--no-subtitles',
        str(original_mkv),
        '--language',
        '0:pol',
        '--track-name',
        '0:Polish (AI)',
        '--default-track-flag',
        '0:yes',  # Make Polish the default subtitle track
        str(polish_ass),
        '--language',
        '0:eng',
        '--track-name',
        '0:English Dialogue',
        '--default-track-flag',
        '0:no',  # Make English non-default
        str(english_ass),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success('   - Clean MKV merge successful')

        if output_mkv.stat().st_size == 0:
            size_mb = output_mkv.stat().st_size / 1024 / 1024
            log_info(f'   - Output size: {size_mb:.1f} MB')

        return True
    except subprocess.CalledProcessError as e:
        log_error(f'Failed to merge: {e}')
        if e.stderr:
            log_error(f'   stderr: {e.stderr}')
        return False


def verify_result(output_mkv: Path):
    """Verify the clean MKV has only the desired tracks."""
    log_info(f'🔍 Verifying result: {output_mkv.name}')

    try:
        result = subprocess.run(
            ['mkvmerge', '-J', str(output_mkv)],
            capture_output=True,
            text=True,
            check=True,
        )

        track_info = json.loads(result.stdout)
        tracks = track_info.get('tracks', [])

        subtitle_tracks = []
        for track in tracks:
            if track.get('type') == 'subtitles':
                props = track.get('properties', {})
                subtitle_tracks.append(
                    {
                        'id': track.get('id'),
                        'language': props.get('language', 'unknown'),
                        'name': props.get('track_name', 'unnamed'),
                    }
                )

        log_info(f'   - Found {len(subtitle_tracks)} subtitle tracks:')
        for track in subtitle_tracks:
            log_info(f'     * Track {track["id"]}: {track["name"]} ({track["language"]})')

        if len(subtitle_tracks) == 2:
            # Check track order: Polish first (default), English second
            polish_first = subtitle_tracks[0]['language'] == 'pol'
            english_second = subtitle_tracks[1]['language'] == 'eng'

            if polish_first and english_second:
                log_success('   ✅ Perfect! Polish (AI) as default track + English dialogue')
                return True
            else:
                log_error('   ❌ Incorrect track order. Expected: Polish first, English second')
                log_error(
                    f'   ❌ Found: Track 1={subtitle_tracks[0]["language"]}, Track 2={subtitle_tracks[1]["language"]}'
                )
                return False
        else:
            log_error(f'   ❌ Expected 2 subtitle tracks, found {len(subtitle_tracks)}')
            return False

    except Exception as e:
        log_error(f'Failed to verify: {e}')
        return False
