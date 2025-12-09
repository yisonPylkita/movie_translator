"""FFmpeg utilities for video processing.

This module provides a unified interface to ffmpeg/ffprobe using static-ffmpeg,
which bundles platform-specific binaries - no system installation required.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import static_ffmpeg


@lru_cache(maxsize=1)
def get_ffmpeg_paths() -> tuple[str, str]:
    """Get paths to ffmpeg and ffprobe executables.

    Uses static-ffmpeg to provide platform-specific binaries.
    Results are cached for performance.

    Returns:
        Tuple of (ffmpeg_path, ffprobe_path)
    """
    ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    return ffmpeg_path, ffprobe_path


def get_ffmpeg() -> str:
    """Get path to ffmpeg executable."""
    return get_ffmpeg_paths()[0]


def get_ffprobe() -> str:
    """Get path to ffprobe executable."""
    return get_ffmpeg_paths()[1]


def get_video_info(video_path: Path) -> dict[str, Any]:
    """Get detailed information about a video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing streams and format information
    """
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
    """Get all subtitle streams from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        List of subtitle stream dictionaries
    """
    info = get_video_info(video_path)
    streams = info.get('streams', [])
    return [s for s in streams if s.get('codec_type') == 'subtitle']


def extract_subtitle(
    video_path: Path,
    stream_index: int,
    output_path: Path,
) -> bool:
    """Extract a subtitle stream from a video file.

    Args:
        video_path: Path to the video file
        stream_index: Index of the subtitle stream to extract
        output_path: Path for the extracted subtitle file

    Returns:
        True if extraction was successful
    """
    ffmpeg = get_ffmpeg()

    cmd = [
        ffmpeg,
        '-y',  # Overwrite output
        '-i',
        str(video_path),
        '-map',
        f'0:s:{stream_index}',
        '-c:s',
        'copy',
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def mux_video_with_subtitles(
    video_path: Path,
    subtitle_files: list[tuple[Path, str, str, bool]],
    output_path: Path,
    copy_audio: bool = True,
    copy_video: bool = True,
) -> bool:
    """Mux video with new subtitle tracks, removing existing subtitles.

    Args:
        video_path: Path to the source video file
        subtitle_files: List of (path, language, title, is_default) tuples
        output_path: Path for the output video file
        copy_audio: Whether to copy audio streams
        copy_video: Whether to copy video streams

    Returns:
        True if muxing was successful
    """
    ffmpeg = get_ffmpeg()

    cmd = [
        ffmpeg,
        '-y',  # Overwrite output
        '-i',
        str(video_path),
    ]

    # Add subtitle inputs
    for sub_path, _, _, _ in subtitle_files:
        cmd.extend(['-i', str(sub_path)])

    # Map video and audio from source (no subtitles)
    if copy_video:
        cmd.extend(['-map', '0:v'])
    if copy_audio:
        cmd.extend(['-map', '0:a'])

    # Map each subtitle file
    for i in range(1, len(subtitle_files) + 1):
        cmd.extend(['-map', f'{i}:0'])

    # Copy codecs (no re-encoding)
    if copy_video:
        cmd.extend(['-c:v', 'copy'])
    if copy_audio:
        cmd.extend(['-c:a', 'copy'])
    cmd.extend(['-c:s', 'ass'])  # Subtitle codec

    # Set metadata for each subtitle track
    for i, (_, lang, title, is_default) in enumerate(subtitle_files):
        cmd.extend([f'-metadata:s:s:{i}', f'language={lang}'])
        cmd.extend([f'-metadata:s:s:{i}', f'title={title}'])
        # Set default flag
        disposition = 'default' if is_default else '0'
        cmd.extend([f'-disposition:s:{i}', disposition])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_ffmpeg_version() -> str:
    """Get the version of ffmpeg."""
    ffmpeg = get_ffmpeg()
    result = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True)
    # First line contains version info
    first_line = result.stdout.split('\n')[0]
    return first_line


# Supported video extensions (can be expanded in the future)
SUPPORTED_VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.webm', '.mov'}


def is_supported_video(path: Path) -> bool:
    """Check if a file is a supported video format."""
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
