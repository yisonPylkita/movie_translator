import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import static_ffmpeg.run

from .types import SubtitleFile


class VideoMuxError(Exception):
    pass


@lru_cache(maxsize=1)
def get_ffmpeg_paths() -> tuple[str, str]:
    # First try to use system FFmpeg (for proper arm64 support on Apple Silicon)
    ffmpeg_path = '/opt/homebrew/bin/ffmpeg' if os.path.exists('/opt/homebrew/bin/ffmpeg') else None
    ffprobe_path = (
        '/opt/homebrew/bin/ffprobe' if os.path.exists('/opt/homebrew/bin/ffprobe') else None
    )

    # Fallback to static_ffmpeg if system FFmpeg is not available
    if not ffmpeg_path or not ffprobe_path:
        try:
            ffmpeg_path, ffprobe_path = (
                static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            )
        except Exception as err:
            raise VideoMuxError(
                "FFmpeg not found. Please install FFmpeg with 'brew install ffmpeg' or run ./setup.sh"
            ) from err

    return ffmpeg_path, ffprobe_path


@lru_cache(maxsize=1)
def get_mkvmerge() -> str | None:
    """Find mkvmerge binary. Returns path or None if not available."""
    path = shutil.which('mkvmerge')
    if path:
        return path
    homebrew = '/opt/homebrew/bin/mkvmerge'
    if os.path.exists(homebrew):
        return homebrew
    return None


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


def probe_video_encoding(video_path: Path) -> dict[str, Any]:
    """Extract video encoding parameters for re-encoding."""
    info = get_video_info(video_path)

    video_stream = next(
        (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
        None,
    )
    if not video_stream:
        raise VideoMuxError(f'No video stream found in {video_path}')

    # Parse frame rate from r_frame_rate (e.g., "24/1" or "24000/1001")
    r_frame_rate = video_stream.get('r_frame_rate', '24/1')
    num, den = map(int, r_frame_rate.split('/'))
    fps = num / den

    # Bitrate may be per-stream or in format-level
    bit_rate = video_stream.get('bit_rate') or info.get('format', {}).get('bit_rate', '5000000')

    return {
        'codec_name': video_stream.get('codec_name', 'h264'),
        'profile': video_stream.get('profile', ''),
        'width': video_stream.get('width', 1920),
        'height': video_stream.get('height', 1080),
        'bit_rate': str(bit_rate),
        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
        'fps': fps,
    }


def _mimetype_for_font(font_path: Path) -> str:
    ext = font_path.suffix.lower()
    if ext == '.otf':
        return 'application/vnd.ms-opentype'
    return 'application/x-truetype-font'


def mux_video_with_subtitles(
    video_path: Path,
    subtitle_files: list[SubtitleFile],
    output_path: Path,
    font_attachments: list[Path] | None = None,
) -> None:
    if not video_path.exists():
        raise VideoMuxError(f'Video file not found: {video_path}')

    for sub in subtitle_files:
        if not sub.path.exists():
            raise VideoMuxError(f'Subtitle file not found: {sub.path}')

    is_mkv = output_path.suffix.lower() in ('.mkv', '.mka', '.mks')
    mkvmerge = get_mkvmerge() if is_mkv else None

    if mkvmerge:
        _mux_with_mkvmerge(mkvmerge, video_path, subtitle_files, output_path, font_attachments)
    else:
        _mux_with_ffmpeg(video_path, subtitle_files, output_path, font_attachments)


def _mux_with_mkvmerge(
    mkvmerge: str,
    video_path: Path,
    subtitle_files: list[SubtitleFile],
    output_path: Path,
    font_attachments: list[Path] | None = None,
) -> None:
    """Mux using mkvmerge — properly interleaves subtitle packets with video data."""
    cmd = [
        mkvmerge,
        '-o', str(output_path),
        '--no-subtitles',
        str(video_path),
    ]

    for sub in subtitle_files:
        cmd.extend(['--language', f'0:{sub.language}'])
        cmd.extend(['--track-name', f'0:{sub.title}'])
        cmd.extend(['--default-track-flag', f'0:{"1" if sub.is_default else "0"}'])
        cmd.append(str(sub.path))

    if font_attachments:
        for font_path in font_attachments:
            cmd.extend(['--attach-file', str(font_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    # mkvmerge: 0 = success, 1 = warnings, 2 = error
    if result.returncode >= 2:
        error_msg = result.stdout.strip() or result.stderr.strip() or 'Unknown mkvmerge error'
        raise VideoMuxError(f'Failed to mux video: {error_msg}')


def _mux_with_ffmpeg(
    video_path: Path,
    subtitle_files: list[SubtitleFile],
    output_path: Path,
    font_attachments: list[Path] | None = None,
) -> None:
    """Fallback muxing with ffmpeg (for MP4 or when mkvmerge is unavailable)."""
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
    # Preserve existing font/attachment streams from the original video
    cmd.extend(['-map', '0:t?'])

    for i in range(1, len(subtitle_files) + 1):
        cmd.extend(['-map', f'{i}:0'])

    cmd.extend(['-c:v', 'copy'])
    cmd.extend(['-c:a', 'copy'])
    # Select subtitle codec based on output container
    subtitle_codec = 'mov_text' if output_path.suffix.lower() == '.mp4' else 'ass'
    cmd.extend(['-c:s', subtitle_codec])

    # Attach new fonts (MKV only)
    if font_attachments and output_path.suffix.lower() != '.mp4':
        for font_path in font_attachments:
            cmd.extend(
                [
                    '-attach',
                    str(font_path),
                    '-metadata:s:t',
                    f'mimetype={_mimetype_for_font(font_path)}',
                    '-metadata:s:t',
                    f'filename={font_path.name}',
                ]
            )

    for i, sub in enumerate(subtitle_files):
        cmd.extend([f'-metadata:s:s:{i}', f'language={sub.language}'])
        cmd.extend([f'-metadata:s:s:{i}', f'title={sub.title}'])
        disposition = 'default' if sub.is_default else '0'
        cmd.extend([f'-disposition:s:{i}', disposition])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_lines = [line for line in result.stderr.split('\n') if 'error' in line.lower()]
        error_msg = '; '.join(error_lines) if error_lines else 'Unknown ffmpeg error'
        raise VideoMuxError(f'Failed to mux video: {error_msg}')


def get_ffmpeg_version() -> str:
    ffmpeg = get_ffmpeg()
    result = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True)
    first_line = result.stdout.split('\n')[0]
    return first_line
