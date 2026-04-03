import subprocess
from pathlib import Path

from ..ffmpeg import get_ffmpeg, get_video_info
from ..logging import logger


class FrameExtractionError(Exception):
    pass


def get_video_duration_ms(video_path: Path) -> int:
    info = get_video_info(video_path)
    duration = float(info.get('format', {}).get('duration', 0))
    return int(duration * 1000)


def extract_subtitle_region_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 10,
    crop_ratio: float = 0.25,
    scale_width: int | None = None,
) -> list[tuple[Path, int]]:
    if not video_path.exists():
        raise FrameExtractionError(f'Video file not found: {video_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    # Crop bottom portion of frame, optionally scale, extract at target fps
    crop_y_start = f'ih*{1 - crop_ratio}'
    crop_height = f'ih*{crop_ratio}'
    vf_parts = [f'crop=iw:{crop_height}:0:{crop_y_start}']
    if scale_width is not None:
        vf_parts.append(f"scale='min({scale_width},iw)':-1")
    vf_parts.append(f'fps={fps}')
    vf = ','.join(vf_parts)

    pattern = str(output_dir / '%06d.jpg')

    cmd = [
        ffmpeg,
        '-y',
        '-i',
        str(video_path),
        '-vf',
        vf,
        '-q:v',
        '2',
        pattern,
    ]

    logger.info(
        f'Extracting subtitle region frames at {fps}fps (bottom {int(crop_ratio * 100)}%)...'
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_lines = [line for line in result.stderr.split('\n') if 'error' in line.lower()]
        error_msg = '; '.join(error_lines) if error_lines else 'Unknown FFmpeg error'
        raise FrameExtractionError(f'Frame extraction failed: {error_msg}')

    # Build frame list with timestamps
    frame_interval_ms = 1000 // fps
    frames: list[tuple[Path, int]] = []

    for frame_path in sorted(output_dir.glob('*.jpg')):
        # FFmpeg numbers frames starting at 1
        frame_num = int(frame_path.stem)
        timestamp_ms = (frame_num - 1) * frame_interval_ms
        frames.append((frame_path, timestamp_ms))

    logger.info(f'Extracted {len(frames)} frames')
    return frames
