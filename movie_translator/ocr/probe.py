"""Quick probe to detect burned-in subtitles before committing to full OCR.

Extracts a random sample of frames from the subtitle region and runs OCR
on each. If none contain text, the video likely has no burned-in subtitles.
"""

import random
import shutil
import subprocess
from pathlib import Path

from ..ffmpeg import get_ffmpeg, get_video_info
from ..logging import logger
from .vision_ocr import recognize_text


def probe_for_burned_in_subtitles(
    video_path: Path,
    num_samples: int = 100,
    crop_ratio: float = 0.25,
    min_text_frames: int = 1,
) -> bool:
    """Quick-check whether a video has burned-in subtitles.

    Extracts num_samples random frames from the subtitle region (bottom crop_ratio
    of the frame), runs OCR on each, and returns True if at least min_text_frames
    contain text.

    This takes ~5-10 seconds instead of minutes for full OCR extraction.
    """
    info = get_video_info(video_path)
    duration = float(info.get('format', {}).get('duration', 0))
    if duration <= 0:
        logger.warning('Could not determine video duration for OCR probe')
        return True  # Assume yes if we can't check

    # Generate random timestamps spread across the video
    # Skip first and last 5% to avoid intros/credits
    margin = duration * 0.05
    start = margin
    end = duration - margin
    if end <= start:
        start, end = 0, duration

    timestamps = sorted(
        random.sample(
            [start + (end - start) * i / (num_samples * 10) for i in range(num_samples * 10)],
            min(num_samples, num_samples * 10),
        )
    )

    probe_dir = video_path.parent / '.translate_temp' / '_ocr_probe'
    probe_dir.mkdir(parents=True, exist_ok=True)

    try:
        ffmpeg = get_ffmpeg()
        crop_y_start = f'ih*{1 - crop_ratio}'
        crop_height = f'ih*{crop_ratio}'

        text_count = 0
        logger.info(f'Probing for burned-in subtitles ({num_samples} random frames)...')

        for i, ts in enumerate(timestamps):
            frame_path = probe_dir / f'probe_{i:04d}.jpg'
            cmd = [
                ffmpeg,
                '-ss',
                f'{ts:.3f}',
                '-i',
                str(video_path),
                '-vf',
                f'crop=iw:{crop_height}:0:{crop_y_start}',
                '-vframes',
                '1',
                '-q:v',
                '2',
                '-y',
                str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not frame_path.exists():
                continue

            text = recognize_text(frame_path)
            frame_path.unlink(missing_ok=True)

            if text and len(text.strip()) > 1:
                text_count += 1
                if text_count >= min_text_frames:
                    logger.info(
                        f'Burned-in subtitles detected (found text in {text_count}/{i + 1} '
                        f'probed frames): "{text.strip()[:60]}..."'
                    )
                    return True

        logger.info(f'No burned-in subtitles detected ({text_count}/{num_samples} frames had text)')
        return False

    finally:
        if probe_dir.exists():
            shutil.rmtree(probe_dir)
