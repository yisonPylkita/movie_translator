import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import static_ffmpeg.run

from .types import SubtitleFile


@lru_cache(maxsize=1)
def get_ffmpeg_paths() -> tuple[str, str]:
    return static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()


def get_ffmpeg() -> str:
    return get_ffmpeg_paths()[0]


def get_ffprobe() -> str:
    return get_ffmpeg_paths()[1]


def get_video_info(video_path: Path) -> dict[str, Any]:
    ffprobe = get_ffprobe()

    cmd = [
        ffprobe,
        '-v',
        'quiet',
        '-print_format',
        'json',
        '-show_streams',
        '-show_format',
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_subtitle_streams(video_path: Path) -> list[dict[str, Any]]:
    info = get_video_info(video_path)
    streams = info.get('streams', [])
    return [s for s in streams if s.get('codec_type') == 'subtitle']


def mux_video_with_subtitles(
    video_path: Path,
    subtitle_files: list[SubtitleFile],
    output_path: Path,
) -> bool:
    ffmpeg = get_ffmpeg()

    cmd = [
        ffmpeg,
        '-y',
        '-i',
        str(video_path),
    ]

    for sub in subtitle_files:
        cmd.extend(['-i', str(sub.path)])

    cmd.extend(['-map', '0:v'])
    cmd.extend(['-map', '0:a'])

    for i in range(1, len(subtitle_files) + 1):
        cmd.extend(['-map', f'{i}:0'])

    cmd.extend(['-c:v', 'copy'])
    cmd.extend(['-c:a', 'copy'])
    cmd.extend(['-c:s', 'ass'])

    for i, sub in enumerate(subtitle_files):
        cmd.extend([f'-metadata:s:s:{i}', f'language={sub.language}'])
        cmd.extend([f'-metadata:s:s:{i}', f'title={sub.title}'])
        disposition = 'default' if sub.is_default else '0'
        cmd.extend([f'-disposition:s:{i}', disposition])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_ffmpeg_version() -> str:
    ffmpeg = get_ffmpeg()
    result = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True)
    first_line = result.stdout.split('\n')[0]
    return first_line
