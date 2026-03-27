import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ..ffmpeg import get_ffmpeg, probe_video_encoding
from ..logging import logger
from ..types import BoundingBox, OCRResult
from .inpainter import Inpainter
from .mask_generator import generate_mask

# Inpainting performance optimization: crop + downscale
#
# Video files are frequently encoded at resolutions much higher than their actual
# source content. For example, 480p anime is commonly upscaled to 1080p containers.
# Running LaMa inpainting on the full frame (e.g. 1440x1080) is extremely slow and
# wasteful — the extra pixels are interpolated upscale artifacts, not real detail.
#
# Two optimizations are applied:
#
# 1. CROP: Subtitles appear in a small region (typically the bottom ~25% of the
#    frame). Instead of feeding the entire frame to LaMa, we crop just the subtitle
#    region + padding for context. This alone reduces pixel count by ~70-75%.
#
# 2. DOWNSCALE: The crop is further downscaled to MAX_INPAINT_WIDTH before
#    inpainting if it exceeds that width. LaMa produces good results at moderate
#    resolutions and doesn't benefit from upscaled pixel data. The inpainted region
#    is upscaled back and pasted onto the original frame.
#
# Combined, these reduce the pixel count fed to LaMa by ~16x for a typical 1080p
# video (from ~1.5M pixels down to ~97K), bringing per-frame time from ~1.2s to
# well under 100ms.
MAX_INPAINT_WIDTH = 720
CROP_PADDING_PX = 40


def _build_subtitle_lookup(
    ocr_results: list[OCRResult],
    fps: float,
) -> dict[int, list[BoundingBox]]:
    """Map video frame indices to bounding boxes for frames that need inpainting."""
    lookup: dict[int, list[BoundingBox]] = {}
    for result in ocr_results:
        if not result.text or not result.boxes:
            continue
        start_frame = int(result.timestamp_ms * fps / 1000)
        end_frame = int((result.timestamp_ms + 1000) * fps / 1000)
        for frame_idx in range(start_frame, end_frame):
            lookup[frame_idx] = result.boxes
    return lookup


def _compute_crop_region(
    boxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
    padding_px: int = CROP_PADDING_PX,
) -> tuple[int, int, int, int]:
    """Compute pixel crop region (x1, y1, x2, y2) covering all boxes + padding.

    Uses full frame width since subtitles can span horizontally.
    """
    min_y = min(box.y for box in boxes)
    max_y = max(box.y + box.height for box in boxes)

    y1 = max(0, int(min_y * frame_height) - padding_px)
    y2 = min(frame_height, int(max_y * frame_height) + padding_px)

    return 0, y1, frame_width, y2


def _remap_boxes_to_crop(
    boxes: list[BoundingBox],
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> list[BoundingBox]:
    """Remap normalized bounding boxes from full-frame to crop-local coordinates."""
    remapped = []
    for box in boxes:
        px = box.x * frame_w - crop_x1
        py = box.y * frame_h - crop_y1
        pw = box.width * frame_w
        ph = box.height * frame_h
        remapped.append(
            BoundingBox(
                x=px / crop_w,
                y=py / crop_h,
                width=pw / crop_w,
                height=ph / crop_h,
            )
        )
    return remapped


def _inpaint_frame(
    image: Image.Image,
    boxes: list[BoundingBox],
    inpainter: Inpainter,
    frame_w: int,
    frame_h: int,
) -> Image.Image:
    """Inpaint subtitle regions using crop + downscale optimization."""
    # Step 1: Compute crop around subtitle region
    crop_x1, crop_y1, crop_x2, crop_y2 = _compute_crop_region(boxes, frame_w, frame_h)
    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_w, crop_h = crop.size

    # Step 2: Remap boxes to crop-local coordinates and generate mask
    crop_boxes = _remap_boxes_to_crop(boxes, crop_x1, crop_y1, crop_w, crop_h, frame_w, frame_h)
    mask = generate_mask(crop_boxes, crop_w, crop_h)

    # Step 3: Downscale for inpainting if the crop exceeds MAX_INPAINT_WIDTH
    scale = min(1.0, MAX_INPAINT_WIDTH / crop_w)
    if scale < 1.0:
        small_w = int(crop_w * scale)
        small_h = int(crop_h * scale)
        small_crop = crop.resize((small_w, small_h), Image.LANCZOS)
        small_mask = mask.resize((small_w, small_h), Image.NEAREST)
    else:
        small_crop = crop
        small_mask = mask

    # Step 4: Inpaint
    result = inpainter.inpaint(small_crop, small_mask)

    # Step 5: Upscale back to original crop size if we downscaled
    if scale < 1.0:
        result = result.resize((crop_w, crop_h), Image.LANCZOS)

    # Step 6: Paste inpainted region back onto original frame
    output = image.copy()
    output.paste(result, (crop_x1, crop_y1))
    return output


def _select_encoder(encoding: dict) -> tuple[str, list[str]]:
    """Select best available encoder. Hardware-accelerated on macOS."""
    bitrate = encoding.get('bit_rate', '5000000')

    if sys.platform == 'darwin':
        return 'h264_videotoolbox', ['-b:v', str(bitrate)]

    return 'libx264', ['-crf', '18', '-preset', 'medium']


def remove_burned_in_subtitles(
    video_path: Path,
    output_path: Path,
    ocr_results: list[OCRResult],
    device: str = 'cpu',
) -> None:
    """Remove burned-in subtitles from video using LaMa inpainting.

    Decodes the video frame-by-frame via FFmpeg pipe, inpaints frames that
    have subtitle bounding boxes, and re-encodes to the output path.
    Audio is stream-copied from the original.
    """
    encoding = probe_video_encoding(video_path)
    w = encoding['width']
    h = encoding['height']
    fps = encoding['fps']
    frame_size = w * h * 3

    subtitle_lookup = _build_subtitle_lookup(ocr_results, fps)
    if not subtitle_lookup:
        logger.warning('No subtitle frames to inpaint — copying original')
        shutil.copy2(video_path, output_path)
        return

    total_subtitle_frames = len(subtitle_lookup)
    logger.info(f'Inpainting {total_subtitle_frames} frames with burned-in subtitles...')

    inpainter = Inpainter(device=device)
    ffmpeg = get_ffmpeg()

    # Decoder: video → raw RGB24 frames via pipe
    decode_cmd = [
        ffmpeg,
        '-i',
        str(video_path),
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgb24',
        '-v',
        'quiet',
        'pipe:1',
    ]

    # Encoder: raw RGB24 frames → video, with audio from original
    codec, codec_args = _select_encoder(encoding)
    encode_cmd = [
        ffmpeg,
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgb24',
        '-s',
        f'{w}x{h}',
        '-r',
        str(fps),
        '-i',
        'pipe:0',
        '-i',
        str(video_path),
        '-map',
        '0:v',
        '-map',
        '1:a',
        '-c:v',
        codec,
        *codec_args,
        '-c:a',
        'copy',
        '-y',
        str(output_path),
    ]

    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Reader thread prevents deadlock: drains decoder stdout while main
    # thread may be blocked writing to encoder stdin.
    frame_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=30)

    def _reader():
        try:
            while True:
                data = decoder.stdout.read(frame_size)
                if len(data) < frame_size:
                    frame_queue.put(None)
                    break
                frame_queue.put(data)
        except Exception as e:
            logger.debug(f'Reader thread error: {e}')
            frame_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    frame_idx = 0
    inpainted_count = 0

    try:
        while True:
            raw = frame_queue.get()
            if raw is None:
                break

            if frame_idx in subtitle_lookup:
                frame_array = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                image = Image.fromarray(frame_array)
                result = _inpaint_frame(
                    image,
                    subtitle_lookup[frame_idx],
                    inpainter,
                    w,
                    h,
                )
                if result.size != (w, h):
                    result = result.resize((w, h))
                raw = np.array(result).tobytes()
                inpainted_count += 1

                if inpainted_count % 100 == 0:
                    logger.info(f'  Inpainted {inpainted_count}/{total_subtitle_frames} frames...')

            encoder.stdin.write(raw)
            frame_idx += 1
    finally:
        encoder.stdin.close()
        reader_thread.join(timeout=10)
        decoder.wait()
        encoder.wait()

    if encoder.returncode != 0:
        stderr = encoder.stderr.read().decode() if encoder.stderr else ''
        raise RuntimeError(f'FFmpeg encoder failed (code {encoder.returncode}): {stderr}')

    logger.info(f'Inpainting complete: {inpainted_count}/{frame_idx} frames modified')
