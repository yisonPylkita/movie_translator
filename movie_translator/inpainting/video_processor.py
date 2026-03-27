import queue
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
        logger.warning('No subtitle frames to inpaint — skipping')
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
        stderr=subprocess.DEVNULL,
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
        except Exception:
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
                mask = generate_mask(subtitle_lookup[frame_idx], w, h)
                result = inpainter.inpaint(image, mask)
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

    logger.info(f'Inpainting complete: {inpainted_count}/{frame_idx} frames modified')
