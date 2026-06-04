import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..logging import logger
from ..srt import write_srt
from ..types import BoundingBox, BurnedInResult, DialogueLine, OCRResult
from .frame_extractor import extract_subtitle_region_frames
from .vision_ocr import recognize_text_with_boxes

# ── Configurable constants ───────────────────────────────────────────────────
# Tuned via scripts/ocr_golden_analysis.py against a "golden" sample (every
# frame OCR'd at 12fps) on Isekai Ojisan ep5. The OLD change metric — *mean*
# absolute pixel diff over the whole crop — drowns short lines: a 1-2 word
# subtitle changes too few pixels to move the mean, so no transition fired and
# the line was never OCR'd. Switching to the FRACTION of significantly-changed
# pixels catches them (text lights up enough pixels regardless of the mean):
# golden coverage went from 4 missing -> 0 missing.
OCR_EXTRACT_FPS = 6
OCR_SCALE_WIDTH = 1280  # it is 720p as - 1280w x720h
OCR_CROP_RATIO = 0.25  # bottom 25% of frame
OCR_PIXEL_DELTA = 25  # per-pixel |Δ| (0-255) that counts as "changed"
OCR_CHANGE_FRACTION = 0.006  # fraction of changed pixels that flags a transition
OCR_VARIANCE_THRESHOLD = 200.0  # pixel variance threshold for "has text"


def _is_sign_text(text: str) -> bool:
    """True for an on-screen sign/logo line (e.g. ``CALENDAR``) — a single
    all-caps token with no spaces. Real dialogue is multi-word/lowercase, so
    this drops sign text that sits inside the subtitle crop without touching
    real lines."""
    stripped = text.strip()
    if not stripped or ' ' in stripped:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


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
        # Fraction of pixels that changed appreciably — robust for short lines
        # (a few words still light up enough pixels) where a mean diff would be
        # diluted to near-zero by the unchanged background.
        delta = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
        changed_fraction = float(np.mean(delta > OCR_PIXEL_DELTA))

        if changed_fraction > OCR_CHANGE_FRACTION:
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
            # Drop on-screen sign/logo boxes (e.g. CALENDAR) that fall inside
            # the subtitle crop, keeping only real dialogue text.
            text_boxes = [(t, b) for (t, b) in text_boxes if not _is_sign_text(t)]
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
        write_srt(lines, srt_path)

        return BurnedInResult(srt_path, ocr_results)

    finally:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
