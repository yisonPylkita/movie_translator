import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..logging import logger
from ..types import BoundingBox, BurnedInResult, DialogueLine, OCRResult
from .frame_extractor import extract_subtitle_region_frames
from .vision_ocr import recognize_text_with_boxes

# ── Configurable constants ───────────────────────────────────────────────────
OCR_EXTRACT_FPS = 3
OCR_SCALE_WIDTH = 1280  # it is 720p as - 1280w x720h
OCR_CROP_RATIO = 0.25  # bottom 25% of frame
OCR_CHANGE_THRESHOLD = 15.0  # mean absolute pixel diff to detect a change
OCR_VARIANCE_THRESHOLD = 200.0  # pixel variance threshold for "has text"


def _map_box_to_full_frame(box: BoundingBox, crop_ratio: float) -> BoundingBox:
    """Map bounding box from cropped frame coordinates to full frame coordinates."""
    return BoundingBox(
        x=box.x,
        y=(1 - crop_ratio) + (box.y * crop_ratio),
        width=box.width,
        height=box.height * crop_ratio,
    )


def _detect_transition_frames(
    frames: list[tuple[Path, int]],
) -> list[tuple[Path, int]]:
    """Identify frames where a subtitle transition occurred via pixel diff.

    Compares consecutive frames and returns only those where the subtitle
    region changed significantly, plus the first frame if it contains text.
    """
    if len(frames) < 2:
        return list(frames)

    def load_gray(p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert('L'))

    prev = load_gray(frames[0][0])
    prev_has_text = float(np.var(prev)) > OCR_VARIANCE_THRESHOLD
    transition_frames: list[tuple[Path, int]] = []

    if prev_has_text:
        transition_frames.append(frames[0])

    for i in range(1, len(frames)):
        path, ts = frames[i]
        curr = load_gray(path)
        diff = np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16)))

        if diff > OCR_CHANGE_THRESHOLD:
            curr_has_text = float(np.var(curr)) > OCR_VARIANCE_THRESHOLD
            if curr_has_text:
                transition_frames.append((path, ts))
            prev_has_text = curr_has_text

        prev = curr

    logger.info(
        f'Change detection: {len(transition_frames)} transitions out of {len(frames)} frames'
    )
    return transition_frames


def _build_dialogue_lines_from_ocr(
    frame_texts: list[tuple[int, str]],
) -> list[DialogueLine]:
    """Build dialogue lines from per-frame OCR results using text-based deduplication."""
    lines: list[DialogueLine] = []
    prev_text = ''
    start_ms = 0

    for timestamp_ms, text in frame_texts:
        if text != prev_text:
            if prev_text and len(prev_text) > 1:
                lines.append(DialogueLine(start_ms, timestamp_ms, prev_text))
            start_ms = timestamp_ms
            prev_text = text

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
    crop_ratio: float = OCR_CROP_RATIO,
    fps: int = OCR_EXTRACT_FPS,
    language: str = 'en',
) -> BurnedInResult | None:
    """Extract burned-in subtitles via OCR, returning SRT path and per-frame bounding boxes.

    Uses change detection to OCR only frames where subtitle text changed,
    and scales frames to 720p width for efficiency.
    """
    frames_dir = output_dir / '_ocr_frames'

    try:
        frames = extract_subtitle_region_frames(
            video_path,
            frames_dir,
            fps=fps,
            crop_ratio=crop_ratio,
            scale_width=OCR_SCALE_WIDTH,
        )
        if not frames:
            logger.error('No frames extracted from video')
            return None

        # Phase 1: detect transitions (cheap — numpy pixel diffs)
        transition_frames = _detect_transition_frames(frames)
        if not transition_frames:
            logger.warning('No subtitle transitions detected in video')
            return None

        # Phase 2: OCR only transition frames (expensive)
        logger.info(
            f'Running OCR on {len(transition_frames)} transition frames (lang={language})...'
        )
        frame_texts: list[tuple[int, str]] = []
        ocr_results: list[OCRResult] = []

        for i, (frame_path, timestamp_ms) in enumerate(transition_frames):
            text_boxes = recognize_text_with_boxes(frame_path, language=language)
            text = '\n'.join(t for t, _ in text_boxes).strip()
            frame_texts.append((timestamp_ms, text))

            # Map bounding boxes from crop-space to full-frame coordinates
            full_frame_boxes = [_map_box_to_full_frame(box, crop_ratio) for _, box in text_boxes]
            ocr_results.append(OCRResult(timestamp_ms, text, full_frame_boxes))

            if (i + 1) % 100 == 0:
                logger.info(f'  OCR progress: {i + 1}/{len(transition_frames)}')

        lines = _build_dialogue_lines_from_ocr(frame_texts)
        if not lines:
            logger.warning('OCR produced no usable subtitle lines')
            return None

        logger.info(f'Extracted {len(lines)} subtitle lines via OCR')

        srt_path = output_dir / f'{video_path.stem}_ocr.srt'
        _write_srt(lines, srt_path)

        return BurnedInResult(srt_path, ocr_results)

    finally:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
