import shutil
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine
from .frame_extractor import extract_subtitle_region_frames
from .vision_ocr import recognize_text


def _build_dialogue_lines_from_ocr(
    frame_texts: list[tuple[int, str]],
) -> list[DialogueLine]:
    """Build dialogue lines from per-frame OCR results using text-based deduplication."""
    lines: list[DialogueLine] = []
    prev_text = ''
    start_ms = 0

    for timestamp_ms, text in frame_texts:
        if text != prev_text:
            # Previous subtitle ended — save it if non-garbage
            if prev_text and len(prev_text) > 1:
                lines.append(DialogueLine(start_ms, timestamp_ms, prev_text))
            start_ms = timestamp_ms
            prev_text = text

    # Handle last subtitle
    if prev_text and len(prev_text) > 1:
        last_ts = frame_texts[-1][0] if frame_texts else start_ms
        lines.append(DialogueLine(start_ms, last_ts + 1000, prev_text))

    return lines


def _format_srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'


def _write_srt(lines: list[DialogueLine], output_path: Path) -> None:
    parts: list[str] = []
    for i, line in enumerate(lines, 1):
        start = _format_srt_time(line.start_ms)
        end = _format_srt_time(line.end_ms)
        parts.append(f'{i}\n{start} --> {end}\n{line.text}\n')

    output_path.write_text('\n'.join(parts), encoding='utf-8')


def extract_burned_in_subtitles(
    video_path: Path,
    output_dir: Path,
    crop_ratio: float = 0.25,
    fps: int = 1,
) -> Path | None:
    """Extract burned-in subtitles by OCR-ing every frame and deduplicating by text.

    Uses 1fps by default — fast enough with Apple Vision's Neural Engine (~44ms/frame)
    and avoids false positives from pixel-based change detection on animated content.
    """
    frames_dir = output_dir / '_ocr_frames'

    try:
        # Step 1: Extract cropped frames
        frames = extract_subtitle_region_frames(
            video_path, frames_dir, fps=fps, crop_ratio=crop_ratio
        )
        if not frames:
            logger.error('No frames extracted from video')
            return None

        # Step 2: OCR every frame
        logger.info(f'Running OCR on {len(frames)} frames...')
        frame_texts: list[tuple[int, str]] = []
        for i, (frame_path, timestamp_ms) in enumerate(frames):
            text = recognize_text(frame_path).strip()
            frame_texts.append((timestamp_ms, text))
            if (i + 1) % 100 == 0:
                logger.info(f'  OCR progress: {i + 1}/{len(frames)}')

        # Step 3: Build dialogue lines via text-based deduplication
        lines = _build_dialogue_lines_from_ocr(frame_texts)
        if not lines:
            logger.warning('OCR produced no usable subtitle lines')
            return None

        logger.info(f'Extracted {len(lines)} subtitle lines via OCR')

        # Step 4: Write SRT
        srt_path = output_dir / f'{video_path.stem}_ocr.srt'
        _write_srt(lines, srt_path)

        return srt_path

    finally:
        # Clean up frame images
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
