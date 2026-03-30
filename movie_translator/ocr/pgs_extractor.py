"""PGS (Presentation Graphic Stream) subtitle extraction via OCR.

Parses PGS bitmap subtitles from .sup files (extracted from MKV via
mkvextract), renders each subtitle event to a grayscale image using
the embedded palette, and OCRs the result with Apple Vision.

PGS subtitles are image-based tracks (codec: hdmv_pgs_subtitle) found
in Blu-ray rips. Unlike burned-in subtitles, PGS tracks have exact
presentation timestamps — no frame-by-frame probing needed.

The parser handles the PGS binary format directly (zero external
dependencies beyond numpy for image arrays).
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..logging import logger
from ..types import BoundingBox, DialogueLine, OCRResult
from .vision_ocr import is_available as is_ocr_available

_VISION_AVAILABLE = False
Quartz: Any = None
Vision: Any = None
try:
    import Quartz  # type: ignore[no-redef]
    import Vision  # type: ignore[no-redef]

    _VISION_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PGS binary format parsing
# ---------------------------------------------------------------------------

_SEG_PCS = 0x16  # Presentation Composition Segment
_SEG_WDS = 0x17  # Window Definition Segment
_SEG_PDS = 0x14  # Palette Definition Segment
_SEG_ODS = 0x15  # Object Definition Segment
_SEG_END = 0x80  # End of Display Set


def _parse_segments(data: bytes) -> list[dict]:
    """Parse raw PGS .sup data into segments."""
    segments = []
    pos = 0
    while pos < len(data) - 13:
        if data[pos : pos + 2] != b'PG':
            break
        pts = struct.unpack('>I', data[pos + 2 : pos + 6])[0] / 90.0  # 90kHz → ms
        seg_type = data[pos + 10]
        seg_size = struct.unpack('>H', data[pos + 11 : pos + 13])[0]
        seg_data = data[pos + 13 : pos + 13 + seg_size]
        segments.append({'pts': pts, 'type': seg_type, 'data': seg_data})
        pos += 13 + seg_size
    return segments


def _decode_rle(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode PGS RLE-encoded bitmap into a palette-indexed array."""
    pixels = []
    total = width * height
    i = 0
    while i < len(data) and len(pixels) < total:
        byte = data[i]
        i += 1
        if byte != 0:
            pixels.append(byte)
        else:
            if i >= len(data):
                break
            flag = data[i]
            i += 1
            if flag == 0:
                # End of line — pad to width boundary
                while len(pixels) % width != 0 and len(pixels) < total:
                    pixels.append(0)
            elif flag & 0xC0 == 0x40:
                length = ((flag & 0x3F) << 8) | data[i]
                i += 1
                pixels.extend([0] * length)
            elif flag & 0xC0 == 0x80:
                length = flag & 0x3F
                color = data[i]
                i += 1
                pixels.extend([color] * length)
            elif flag & 0xC0 == 0xC0:
                length = ((flag & 0x3F) << 8) | data[i]
                i += 1
                color = data[i]
                i += 1
                pixels.extend([color] * length)
            else:
                length = flag & 0x3F
                pixels.extend([0] * length)

    if len(pixels) < total:
        pixels.extend([0] * (total - len(pixels)))
    return np.array(pixels[:total], dtype=np.uint8).reshape((height, width))


# ---------------------------------------------------------------------------
# Image extraction from PGS segments
# ---------------------------------------------------------------------------


def _extract_subtitle_images(
    segments: list[dict],
) -> list[tuple[float, np.ndarray, int, int]]:
    """Extract subtitle bitmap images from parsed PGS segments.

    Returns list of (pts_ms, grayscale_image, width, height) tuples.
    Only returns events that contain actual subtitle content (skips clear events).
    """
    # Lookup tables for palette → grayscale conversion (256 entries max).
    # Updated incrementally as PDS segments arrive.
    y_lut = np.zeros(256, dtype=np.uint8)
    a_lut = np.zeros(256, dtype=np.uint8)

    results = []
    ods_data = b''
    ods_width = 0
    ods_height = 0
    current_pts = 0.0

    for seg in segments:
        if seg['type'] == _SEG_PCS:
            current_pts = seg['pts']
            d = seg['data']
            num_objects = d[8] if len(d) > 8 else 0
            if num_objects == 0:
                continue
            ods_data = b''

        elif seg['type'] == _SEG_PDS:
            d = seg['data']
            i = 2  # Skip palette_id + version
            while i + 4 < len(d):
                entry_id = d[i]
                y_lut[entry_id] = d[i + 1]
                a_lut[entry_id] = d[i + 4]
                i += 5

        elif seg['type'] == _SEG_ODS:
            d = seg['data']
            seq_flag = d[3]
            if seq_flag & 0x80:  # First in sequence
                ods_width = struct.unpack('>H', d[7:9])[0]
                ods_height = struct.unpack('>H', d[9:11])[0]
                ods_data = d[11:]
            else:
                ods_data += d[4:]

            if seq_flag & 0x40:  # Last (or only) in sequence
                if ods_width > 0 and ods_height > 0 and ods_data:
                    indexed = _decode_rle(ods_data, ods_width, ods_height)

                    # Vectorized palette lookup: single numpy operation
                    gray = y_lut[indexed]
                    alpha = a_lut[indexed]
                    img = np.where(alpha > 128, gray, 0).astype(np.uint8)
                    results.append((current_pts, img, ods_width, ods_height))

    return results


# ---------------------------------------------------------------------------
# In-memory OCR (avoids PNG write/read cycle)
# ---------------------------------------------------------------------------


def _ocr_grayscale_image(img: np.ndarray) -> tuple[str, list[BoundingBox]]:
    """OCR a grayscale numpy array directly via Apple Vision, no disk I/O."""
    if not _VISION_AVAILABLE:
        return '', []

    h, w = img.shape[:2]
    color_space = Quartz.CGColorSpaceCreateDeviceGray()  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]
    provider = Quartz.CGDataProviderCreateWithData(None, img.tobytes(), w * h, None)  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]
    cg_image = Quartz.CGImageCreate(w, h, 8, 8, w, color_space, 0, provider, None, False, 0)  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]

    if not cg_image:
        return '', []

    request = Vision.VNRecognizeTextRequest.alloc().init()  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]
    request.setRecognitionLanguages_(['en'])
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return '', []

    texts = []
    boxes = []
    for obs in request.results():
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        texts.append(candidates[0].string())
        bbox = obs.boundingBox()
        boxes.append(
            BoundingBox(
                x=bbox.origin.x,
                y=1.0 - bbox.origin.y - bbox.size.height,
                width=bbox.size.width,
                height=bbox.size.height,
            )
        )

    return '\n'.join(texts).strip(), boxes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pgs_track(
    video_path: Path,
    track_index: int,
    work_dir: Path,
) -> Path | None:
    """Extract a PGS subtitle track from an MKV and produce an SRT file.

    Uses mkvextract to get the .sup stream, parses PGS binary format,
    renders each subtitle event to a grayscale image, and OCRs with
    Apple Vision.

    Args:
        video_path: Path to the MKV file.
        track_index: The MKV track index of the PGS subtitle stream.
        work_dir: Directory for temporary files and output.

    Returns:
        Path to the generated .srt file, or None if extraction failed.
    """
    if not is_ocr_available():
        logger.warning('PGS extraction requires macOS with Vision framework')
        return None

    pgs_dir = work_dir / 'pgs_ocr'
    pgs_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract .sup stream from MKV
    sup_path = pgs_dir / 'track.sup'
    logger.info(f'Extracting PGS track {track_index} from {video_path.name}...')

    result = subprocess.run(
        ['mkvextract', 'tracks', str(video_path), f'{track_index}:{sup_path}'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not sup_path.exists():
        logger.warning(f'Failed to extract PGS track: {result.stderr[:200]}')
        return None

    # Step 2: Parse PGS segments
    logger.info('Parsing PGS subtitle stream...')
    data = sup_path.read_bytes()
    segments = _parse_segments(data)
    images = _extract_subtitle_images(segments)

    if not images:
        logger.warning('No subtitle images found in PGS track')
        return None

    logger.info(f'Found {len(images)} subtitle images, running OCR...')

    # Step 3: OCR each image (in-memory, no disk I/O)
    ocr_results: list[OCRResult] = []
    dialogue_lines: list[DialogueLine] = []
    prev_text = ''
    line_start_ms = 0

    for i, (pts_ms, img, _width, _height) in enumerate(images):
        text, boxes = _ocr_grayscale_image(img)

        if text and boxes:
            ocr_results.append(OCRResult(timestamp_ms=int(pts_ms), text=text, boxes=boxes))

        # Build dialogue lines by deduplicating consecutive identical text
        if text and text != prev_text:
            if prev_text and line_start_ms > 0:
                dialogue_lines.append(
                    DialogueLine(
                        start_ms=int(line_start_ms),
                        end_ms=int(pts_ms),
                        text=prev_text,
                    )
                )
            line_start_ms = pts_ms
            prev_text = text
        elif not text and prev_text:
            dialogue_lines.append(
                DialogueLine(
                    start_ms=int(line_start_ms),
                    end_ms=int(pts_ms),
                    text=prev_text,
                )
            )
            prev_text = ''

        if (i + 1) % 100 == 0:
            logger.info(f'OCR progress: {i + 1}/{len(images)}')

    # Close final line
    if prev_text:
        last_pts = images[-1][0]
        dialogue_lines.append(
            DialogueLine(
                start_ms=int(line_start_ms),
                end_ms=int(last_pts) + 3000,  # Assume 3s display for last sub
                text=prev_text,
            )
        )

    if not dialogue_lines:
        logger.warning('OCR produced no text from PGS images')
        return None

    logger.info(f'Extracted {len(dialogue_lines)} dialogue lines from PGS track')

    # Step 4: Write SRT
    srt_path = work_dir / f'{video_path.stem}_pgs_ocr.srt'
    _write_srt(dialogue_lines, srt_path)

    # Clean up
    sup_path.unlink(missing_ok=True)

    return srt_path


def _write_srt(lines: list[DialogueLine], output_path: Path) -> None:
    """Write dialogue lines as an SRT file."""
    parts = []
    for i, line in enumerate(lines, 1):
        start = _format_srt_time(line.start_ms)
        end = _format_srt_time(line.end_ms)
        parts.append(f'{i}\n{start} --> {end}\n{line.text}\n')
    output_path.write_text('\n'.join(parts), encoding='utf-8')


def _format_srt_time(ms: int) -> str:
    """Format milliseconds as SRT timestamp."""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_part = ms % 1000
    return f'{h:02d}:{m:02d}:{s:02d},{ms_part:03d}'
