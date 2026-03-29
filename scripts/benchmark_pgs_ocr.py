#!/usr/bin/env python3
"""Benchmark PGS subtitle extraction + OCR approaches.

Extracts PGS subtitle images using three backends, OCRs them with Vision,
and compares speed and quality.

Backend A: ffmpeg overlay on black canvas
Backend B: pgsrip library (Python PGS parser)
Backend C: Direct binary PGS parsing (minimal, no deps)

Usage:
    python scripts/benchmark_pgs_ocr.py /path/to/video.mkv
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Backend C: Direct PGS binary parser (zero dependencies)
# ---------------------------------------------------------------------------


def parse_pgs_segments(sup_path: Path):
    """Parse a .sup file into raw PGS segments."""
    data = sup_path.read_bytes()
    segments = []
    pos = 0
    while pos < len(data) - 13:
        if data[pos : pos + 2] != b'PG':
            break
        pts = struct.unpack('>I', data[pos + 2 : pos + 6])[0] / 90.0  # ms
        seg_type = data[pos + 10]
        seg_size = struct.unpack('>H', data[pos + 11 : pos + 13])[0]
        seg_data = data[pos + 13 : pos + 13 + seg_size]
        segments.append({'pts': pts, 'type': seg_type, 'data': seg_data})
        pos += 13 + seg_size
    return segments


def decode_rle(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode PGS RLE-encoded image data into a palette-indexed array."""
    pixels = []
    i = 0
    while i < len(data) and len(pixels) < width * height:
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
                # End of line
                while len(pixels) % width != 0 and len(pixels) < width * height:
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

    # Pad or truncate
    total = width * height
    if len(pixels) < total:
        pixels.extend([0] * (total - len(pixels)))
    return np.array(pixels[:total], dtype=np.uint8).reshape((height, width))


def extract_pgs_direct(sup_path: Path, output_dir: Path) -> list[dict]:
    """Extract PGS subtitle images using direct binary parsing."""
    segments = parse_pgs_segments(sup_path)

    # Group into display sets (PCS → ... → END)
    palette = {}
    results = []
    current_ods_data = b''
    current_ods_width = 0
    current_ods_height = 0
    current_pts = 0.0

    for seg in segments:
        if seg['type'] == 0x16:  # PCS
            d = seg['data']
            current_pts = seg['pts']
            num_objects = d[8] if len(d) > 8 else 0
            if num_objects == 0:
                continue  # Clear event
            current_ods_data = b''

        elif seg['type'] == 0x14:  # PDS
            d = seg['data']
            # palette_id = d[0], palette_version = d[1]
            i = 2
            while i + 4 < len(d):
                entry_id = d[i]
                y, cr, cb, alpha = d[i + 1], d[i + 2], d[i + 3], d[i + 4]
                palette[entry_id] = (y, cr, cb, alpha)
                i += 5

        elif seg['type'] == 0x15:  # ODS
            d = seg['data']
            # obj_id(2) + version(1) + sequence_flag(1) + data_length(3) + width(2) + height(2) + rle
            seq_flag = d[3]
            if seq_flag & 0x80:  # First in sequence
                current_ods_width = struct.unpack('>H', d[7:9])[0]
                current_ods_height = struct.unpack('>H', d[9:11])[0]
                current_ods_data = d[11:]
            else:  # Continuation
                current_ods_data += d[4:]

            if seq_flag & 0x40:  # Last (or only) in sequence
                if current_ods_width > 0 and current_ods_height > 0:
                    # Decode RLE to palette-indexed image
                    indexed = decode_rle(current_ods_data, current_ods_width, current_ods_height)

                    # Convert to grayscale using palette Y channel
                    gray = np.zeros_like(indexed)
                    alpha_ch = np.zeros_like(indexed)
                    for idx, (y_val, _, _, a_val) in palette.items():
                        mask = indexed == idx
                        gray[mask] = y_val
                        alpha_ch[mask] = a_val

                    # Create white-on-black image (typical for OCR)
                    img = np.where(alpha_ch > 128, gray, 0).astype(np.uint8)

                    # Save
                    out_path = output_dir / f'direct_{len(results):04d}_{int(current_pts)}ms.png'
                    cv2.imwrite(str(out_path), img)
                    results.append(
                        {
                            'path': str(out_path),
                            'pts_ms': current_pts,
                            'width': current_ods_width,
                            'height': current_ods_height,
                        }
                    )

    return results


# ---------------------------------------------------------------------------
# Backend A: ffmpeg overlay approach
# ---------------------------------------------------------------------------


def extract_pgs_ffmpeg(video_path: Path, sub_index: int, output_dir: Path) -> list[dict]:
    """Extract PGS by rendering subs onto a black canvas via ffmpeg."""
    # Get video resolution
    probe = subprocess.run(
        [
            'ffprobe',
            '-v',
            'quiet',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height',
            '-of',
            'csv=p=0',
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    width, height = probe.stdout.strip().split(',')

    # Extract subtitle to .sup first
    sup_path = output_dir / 'temp.sup'
    subprocess.run(
        ['mkvextract', 'tracks', str(video_path), f'{sub_index}:{sup_path}'],
        capture_output=True,
    )

    # Use ffmpeg to render PGS on a black canvas at video resolution
    # Output one PNG per subtitle event
    subprocess.run(
        [
            'ffmpeg',
            '-v',
            'quiet',
            '-y',
            '-f',
            'lavfi',
            '-i',
            f'color=black:s={width}x{height}:r=1',
            '-i',
            str(video_path),
            '-filter_complex',
            f'[0:v][1:s:{0}]overlay=format=auto',
            '-fps_mode',
            'passthrough',
            '-t',
            '60',  # First 60 seconds only for benchmark
            str(output_dir / 'ffmpeg_%04d.png'),
        ],
        capture_output=True,
        timeout=120,
    )

    # Collect generated files with timing
    results = []
    for f in sorted(output_dir.glob('ffmpeg_*.png')):
        # ffmpeg numbers frames sequentially; estimate timing
        frame_num = int(f.stem.split('_')[1])
        results.append(
            {
                'path': str(f),
                'pts_ms': frame_num * 1000,  # 1fps
                'width': int(width),
                'height': int(height),
            }
        )

    sup_path.unlink(missing_ok=True)
    return results


# ---------------------------------------------------------------------------
# Backend B: pgsrip library
# ---------------------------------------------------------------------------


def extract_pgs_pgsrip(sup_path: Path, output_dir: Path) -> list[dict]:
    """Extract PGS using pgsrip library."""
    from pgsrip.pgs import MediaPath, PgsReader

    data = sup_path.read_bytes()
    media_path = MediaPath(str(sup_path))
    display_sets = list(PgsReader.decode(data, media_path))

    results = []
    palette_cache = {}

    for ds in display_sets:
        if not ds.ods_segments:
            continue

        pts_ms = ds.pcs.presentation_timestamp.ordinal

        # Build palette from PDS
        for pds in ds.pds_segments:
            for idx, p in enumerate(pds.palettes):
                palette_cache[idx] = p

        for ods in ds.ods_segments:
            if not ods.data:
                continue
            # Decode RLE using our parser — pgsrip parses segments but
            # doesn't expose a working image renderer.
            # ODS data format: obj_id(2) + version(1) + seq_flag(1) + data_len(3) + width(2) + height(2) + rle
            raw = ods.bytes[13:]  # Skip PGS header
            if len(raw) < 11:
                continue
            seq_flag = raw[3]
            if not (seq_flag & 0x80):
                continue  # Not first segment
            width = struct.unpack('>H', raw[7:9])[0]
            height = struct.unpack('>H', raw[9:11])[0]
            if width <= 0 or height <= 0:
                continue

            indexed = decode_rle(ods.data, width, height)

            # Convert using cached palette
            gray = np.zeros_like(indexed)
            alpha_ch = np.zeros_like(indexed)
            for idx, p in palette_cache.items():
                mask = indexed == idx
                gray[mask] = p.y
                alpha_ch[mask] = p.alpha

            img = np.where(alpha_ch > 128, gray, 0).astype(np.uint8)

            out_path = output_dir / f'pgsrip_{len(results):04d}_{pts_ms}ms.png'
            cv2.imwrite(str(out_path), img)
            results.append(
                {
                    'path': str(out_path),
                    'pts_ms': pts_ms,
                    'width': width,
                    'height': height,
                }
            )

    return results


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def ocr_images(images: list[dict]) -> list[dict]:
    """Run Vision OCR on extracted subtitle images."""
    from movie_translator.ocr.vision_ocr import recognize_text

    results = []
    for img_info in images:
        path = Path(img_info['path'])
        if not path.exists():
            continue
        text = recognize_text(path)
        results.append(
            {
                **img_info,
                'text': text.strip() if text else '',
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/benchmark_pgs_ocr.py /path/to/video.mkv')
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f'File not found: {video_path}')
        sys.exit(1)

    work_dir = Path('/tmp/pgs_benchmark')
    work_dir.mkdir(exist_ok=True)

    # Find PGS subtitle track index
    probe = subprocess.run(
        [
            'ffprobe',
            '-v',
            'quiet',
            '-print_format',
            'json',
            '-show_streams',
            '-select_streams',
            's',
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get('streams', [])
    pgs_stream = None
    for s in streams:
        if (
            s['codec_name'] == 'hdmv_pgs_subtitle'
            and s.get('tags', {}).get('language', '') == 'eng'
        ):
            pgs_stream = s
            break

    if not pgs_stream:
        print('No English PGS track found')
        sys.exit(1)

    pgs_index = pgs_stream['index']
    print(f'Found PGS track at index {pgs_index}')

    # Extract .sup file (shared between backends B and C)
    sup_path = work_dir / 'english.sup'
    if not sup_path.exists():
        print('Extracting .sup file...')
        subprocess.run(
            ['mkvextract', 'tracks', str(video_path), f'{pgs_index}:{sup_path}'],
            capture_output=True,
        )

    # -----------------------------------------------------------------------
    # Backend C: Direct parser
    # -----------------------------------------------------------------------
    print('\n=== Backend C: Direct PGS parser ===')
    direct_dir = work_dir / 'direct'
    direct_dir.mkdir(exist_ok=True)

    start = time.time()
    direct_images = extract_pgs_direct(sup_path, direct_dir)
    direct_extract_time = time.time() - start
    print(f'  Extracted {len(direct_images)} images in {direct_extract_time:.2f}s')

    start = time.time()
    direct_ocr = ocr_images(direct_images)
    direct_ocr_time = time.time() - start
    direct_with_text = [r for r in direct_ocr if r['text']]
    print(f'  OCR: {len(direct_with_text)}/{len(direct_ocr)} had text in {direct_ocr_time:.1f}s')

    # -----------------------------------------------------------------------
    # Backend B: pgsrip
    # -----------------------------------------------------------------------
    print('\n=== Backend B: pgsrip library ===')
    pgsrip_dir = work_dir / 'pgsrip'
    pgsrip_dir.mkdir(exist_ok=True)

    start = time.time()
    pgsrip_images = extract_pgs_pgsrip(sup_path, pgsrip_dir)
    pgsrip_extract_time = time.time() - start
    print(f'  Extracted {len(pgsrip_images)} images in {pgsrip_extract_time:.2f}s')

    start = time.time()
    pgsrip_ocr = ocr_images(pgsrip_images)
    pgsrip_ocr_time = time.time() - start
    pgsrip_with_text = [r for r in pgsrip_ocr if r['text']]
    print(f'  OCR: {len(pgsrip_with_text)}/{len(pgsrip_ocr)} had text in {pgsrip_ocr_time:.1f}s')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('BENCHMARK RESULTS')
    print('=' * 60)
    print(f'{"Backend":<25s} {"Extract":>8s} {"OCR":>8s} {"Total":>8s} {"Images":>7s} {"Text":>6s}')
    print('-' * 60)

    for name, extract_t, ocr_t, n_img, n_text in [
        (
            'C: Direct parser',
            direct_extract_time,
            direct_ocr_time,
            len(direct_images),
            len(direct_with_text),
        ),
        (
            'B: pgsrip',
            pgsrip_extract_time,
            pgsrip_ocr_time,
            len(pgsrip_images),
            len(pgsrip_with_text),
        ),
    ]:
        print(
            f'{name:<25s} {extract_t:>7.2f}s {ocr_t:>7.1f}s {extract_t + ocr_t:>7.1f}s {n_img:>7d} {n_text:>6d}'
        )

    # -----------------------------------------------------------------------
    # OCR quality comparison (first 20 lines)
    # -----------------------------------------------------------------------
    print('\n=== OCR Quality: First 20 subtitle lines ===')
    for r in direct_with_text[:20]:
        print(f'  {r["pts_ms"] / 1000:7.1f}s: "{r["text"][:70]}"')

    # Save full results
    output = {
        'direct': [{'pts_ms': r['pts_ms'], 'text': r['text']} for r in direct_ocr],
        'pgsrip': [{'pts_ms': r['pts_ms'], 'text': r['text']} for r in pgsrip_ocr],
    }
    results_path = work_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nFull results: {results_path}')


if __name__ == '__main__':
    main()
