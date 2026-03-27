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
from .backends import InpaintBackend, create_backend
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
    inpainter: InpaintBackend,
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


SCENE_CUT_THRESHOLD = 0.4


def _detect_scene_cut(
    current: np.ndarray,
    reference: np.ndarray,
    threshold: float = SCENE_CUT_THRESHOLD,
) -> bool:
    """Detect if a scene cut occurred between reference and current frame.

    Uses color histogram comparison on the top half of the frame. This is
    robust to character animation and camera movement (which change pixel
    positions but preserve the overall color palette) while reliably detecting
    actual scene cuts (which change the color distribution entirely).

    Args:
        threshold: Histogram similarity below this value triggers scene cut.
                   Range 0.0 (completely different) to 1.0 (identical).
                   Default 0.4 is conservative — only fires on clear scene changes.
    """
    top_half = current.shape[0] // 2
    # Subsample for speed while keeping representative color distribution
    cur_top = current[:top_half:2, ::2].reshape(-1, 3)
    ref_top = reference[:top_half:2, ::2].reshape(-1, 3)

    # Compute per-channel histogram intersection (0=no overlap, 1=identical)
    similarity = 0.0
    for c in range(3):
        cur_hist, _ = np.histogram(cur_top[:, c], bins=64, range=(0, 256))
        ref_hist, _ = np.histogram(ref_top[:, c], bins=64, range=(0, 256))
        cur_norm = cur_hist.astype(np.float64) / (cur_hist.sum() + 1e-10)
        ref_norm = ref_hist.astype(np.float64) / (ref_hist.sum() + 1e-10)
        similarity += float(np.sum(np.minimum(cur_norm, ref_norm)))

    similarity /= 3.0
    return similarity < threshold


def _make_inpaint_processor(
    subtitle_lookup: dict[int, list[BoundingBox]],
    inpainter: InpaintBackend,
    w: int,
    h: int,
):
    """Create a frame processor using an inpainting backend."""

    def process(frame_idx: int, raw: bytes) -> tuple[bytes, bool]:
        if frame_idx not in subtitle_lookup:
            return raw, False
        frame_array = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        image = Image.fromarray(frame_array)
        result = _inpaint_frame(image, subtitle_lookup[frame_idx], inpainter, w, h)
        if result.size != (w, h):
            result = result.resize((w, h))
        return np.array(result).tobytes(), True

    return process


def _make_temporal_processor(
    subtitle_lookup: dict[int, list[BoundingBox]],
    w: int,
    h: int,
):
    """Create a frame processor that fills subtitle regions from nearby clean frames.

    For each subtitle frame, copies pixels from the most recent non-subtitle frame
    in the masked region. Very fast (numpy array copy) with good quality when the
    background doesn't change between the reference and subtitle frames.
    """
    last_clean: list[np.ndarray | None] = [None]

    def process(frame_idx: int, raw: bytes) -> tuple[bytes, bool]:
        if frame_idx not in subtitle_lookup:
            last_clean[0] = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            return raw, False

        if last_clean[0] is None:
            return raw, False

        frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        mask = generate_mask(subtitle_lookup[frame_idx], w, h)
        mask_arr = np.array(mask)
        frame_arr[mask_arr > 128] = last_clean[0][mask_arr > 128]
        return frame_arr.tobytes(), True

    return process


def _make_temporal_hybrid_processor(
    subtitle_lookup: dict[int, list[BoundingBox]],
    w: int,
    h: int,
):
    """Temporal fill with scene-cut detection, falling back to OpenCV Telea.

    Combines the speed of temporal fill (pixel copy from clean reference frame)
    with robustness against scene cuts. When a scene change is detected between
    the reference and current frame, falls back to OpenCV Telea inpainting
    instead of copying stale pixels from a different scene.
    """
    from .backends import OpenCVTeleaBackend

    last_clean: list[np.ndarray | None] = [None]
    fallback = OpenCVTeleaBackend()
    fallback_count = [0]

    def process(frame_idx: int, raw: bytes) -> tuple[bytes, bool]:
        if frame_idx not in subtitle_lookup:
            last_clean[0] = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            return raw, False

        if last_clean[0] is None:
            return raw, False

        frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

        if _detect_scene_cut(frame_arr, last_clean[0]):
            # Scene cut detected — use OpenCV inpainting as fallback
            image = Image.fromarray(frame_arr)
            result = _inpaint_frame(image, subtitle_lookup[frame_idx], fallback, w, h)
            if result.size != (w, h):
                result = result.resize((w, h))
            fallback_count[0] += 1
            if fallback_count[0] % 50 == 1:
                logger.debug(f'  Scene cut at frame {frame_idx}, using OpenCV fallback')
            return np.array(result).tobytes(), True

        # Same scene — fast temporal fill
        frame_arr = frame_arr.copy()
        mask = generate_mask(subtitle_lookup[frame_idx], w, h)
        mask_arr = np.array(mask)
        frame_arr[mask_arr > 128] = last_clean[0][mask_arr > 128]
        return frame_arr.tobytes(), True

    return process


def remove_burned_in_subtitles(
    video_path: Path,
    output_path: Path,
    ocr_results: list[OCRResult],
    device: str = 'cpu',
    backend: str | InpaintBackend = 'lama',
) -> None:
    """Remove burned-in subtitles from video using configurable inpainting.

    Decodes the video frame-by-frame via FFmpeg pipe, processes frames that
    have subtitle bounding boxes, and re-encodes to the output path.
    Audio is stream-copied from the original.

    Args:
        video_path: Input video file.
        output_path: Output video file path.
        ocr_results: OCR results with subtitle bounding boxes.
        device: Device for ML backends ('cpu', 'mps', 'cuda').
        backend: Backend name ('lama', 'opencv-telea', 'opencv-ns', 'temporal',
                 'temporal-hybrid') or a pre-created InpaintBackend instance.
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
    backend_label = backend if isinstance(backend, str) else type(backend).__name__
    logger.info(f'Inpainting {total_subtitle_frames} frames with {backend_label}...')

    # Create frame processor based on backend
    if isinstance(backend, str) and backend == 'temporal':
        process_frame = _make_temporal_processor(subtitle_lookup, w, h)
    elif isinstance(backend, str) and backend == 'temporal-hybrid':
        process_frame = _make_temporal_hybrid_processor(subtitle_lookup, w, h)
    else:
        if isinstance(backend, str):
            inpainter = create_backend(backend, device=device)
        else:
            inpainter = backend
        process_frame = _make_inpaint_processor(subtitle_lookup, inpainter, w, h)

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

    # Two reader threads prevent pipe deadlocks:
    # 1. Drains decoder stdout into a bounded queue for the main thread
    # 2. Drains encoder stderr so FFmpeg doesn't block on warning output
    frame_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=30)
    encoder_stderr_lines: list[str] = []

    def _decoder_reader():
        try:
            while True:
                data = decoder.stdout.read(frame_size)
                if len(data) < frame_size:
                    frame_queue.put(None)
                    break
                frame_queue.put(data)
        except Exception as e:
            logger.debug(f'Decoder reader thread error: {e}')
            frame_queue.put(None)

    def _stderr_reader():
        try:
            for line in encoder.stderr:
                encoder_stderr_lines.append(line.decode(errors='replace').rstrip())
        except Exception:
            pass

    decoder_thread = threading.Thread(target=_decoder_reader, daemon=True)
    stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
    decoder_thread.start()
    stderr_thread.start()

    frame_idx = 0
    inpainted_count = 0

    try:
        while True:
            raw = frame_queue.get()
            if raw is None:
                break

            raw, was_inpainted = process_frame(frame_idx, raw)
            if was_inpainted:
                inpainted_count += 1
                if inpainted_count % 100 == 0:
                    logger.info(f'  Inpainted {inpainted_count}/{total_subtitle_frames} frames...')

            encoder.stdin.write(raw)
            frame_idx += 1
    finally:
        encoder.stdin.close()
        decoder_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        decoder.wait()
        encoder.wait()

    if encoder.returncode != 0:
        stderr = '\n'.join(encoder_stderr_lines[-20:])
        raise RuntimeError(f'FFmpeg encoder failed (code {encoder.returncode}): {stderr}')

    logger.info(f'Inpainting complete: {inpainted_count}/{frame_idx} frames modified')
