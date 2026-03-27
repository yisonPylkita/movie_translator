# Burned-In Subtitle Removal via LaMa Inpainting

## Problem

When a video has burned-in (hardcoded) subtitles, the existing OCR pipeline detects and translates them into new subtitle tracks. But the original burned-in text remains visible in the video frames, causing overlap with the translated subtitles.

## Solution

Add a post-processing step that removes burned-in subtitles from video frames using LaMa image inpainting. The OCR step already identifies where subtitles are — we reuse those bounding boxes to generate inpainting masks, then reconstruct the background behind the text.

## Architecture

```
Existing pipeline (unchanged up to OCR):
  Video → OCR detect (with bounding boxes) → translate → mux subtitle tracks

New post-processing step (after muxing):
  Video → segment at keyframes around subtitle time ranges
    ├─ Segments WITHOUT subtitles → stream copy (lossless, fast)
    └─ Segments WITH subtitles:
         Extract full frames → generate masks from OCR bounding boxes
         → LaMa inpaint each frame → re-encode segment
         (hardware-accelerated, source-matched codec/profile)
    → concatenate all segments → final output
```

## Changes Required

### 1. Vision OCR: Preserve Bounding Boxes

**File:** `movie_translator/ocr/vision_ocr.py`

`recognize_text()` currently returns only a `str`. Modify it to also return bounding box coordinates from `VNRecognizedTextObservation.boundingBox`. This is a `CGRect` in normalized coordinates (0-1 range, origin at bottom-left in Vision framework).

Add a new function `recognize_text_with_boxes()` that returns a list of `(text, bbox)` tuples. Keep the existing `recognize_text()` as a thin wrapper for backward compatibility.

### 2. New Type: `OCRResult`

**File:** `movie_translator/types.py`

```python
class BoundingBox(NamedTuple):
    x: float      # normalized 0-1, left edge
    y: float      # normalized 0-1, top edge (converted from Vision's bottom-left origin)
    width: float   # normalized 0-1
    height: float  # normalized 0-1

class OCRResult(NamedTuple):
    timestamp_ms: int
    text: str
    boxes: list[BoundingBox]  # bounding boxes in full-frame coordinates
```

### 3. Burned-In Extractor: Emit OCR Results with Boxes

**File:** `movie_translator/ocr/burned_in_extractor.py`

The extractor currently produces an SRT file and discards frame-level data. Modify it to also return a list of `OCRResult` objects so downstream inpainting knows where text was on each frame.

Key change: the OCR currently runs on cropped frames (bottom 25%). Bounding boxes from OCR are relative to the crop. We must map them back to full-frame coordinates:
- `full_y = crop_start_ratio + (crop_box_y * crop_ratio)`
- `full_height = crop_box_height * crop_ratio`
- `x` and `width` are unchanged

### 4. New Module: `movie_translator/inpainting/`

```
inpainting/
├── __init__.py
├── mask_generator.py    # Generate binary masks from OCR bounding boxes
├── inpainter.py         # LaMa inpainting wrapper
├── video_processor.py   # Segment-based re-encoding orchestrator
└── tests/
```

#### mask_generator.py

- Takes `OCRResult` list and frame dimensions
- For each frame with text: create a binary mask (white = inpaint region)
- Dilate bounding boxes by ~20px to cover text edges and shadows
- Output: PIL Image or numpy array masks

#### inpainter.py

- Wraps `simple-lama-inpainting` (`pip install simple-lama-inpainting`)
- Input: frame image + binary mask
- Output: inpainted frame image
- Handles GPU (MPS) vs CPU selection matching the existing device config

#### video_processor.py

This is the orchestrator. Steps:

1. **Probe source video** — extract codec, profile, level, bitrate via ffprobe
2. **Identify subtitle time ranges** — from `OCRResult` list, determine which time ranges have burned-in text. Merge nearby ranges (within 2s gap) to avoid excessive segmentation.
3. **Segment video at keyframes** — use ffmpeg to split at nearest keyframes around subtitle ranges. Segments without subtitles are marked for stream copy.
4. **Process subtitle segments:**
   - Extract full frames (not cropped) for the segment
   - Generate masks for each frame using OCR bounding box data
   - Inpaint each frame with LaMa
   - Re-encode segment from inpainted frames using source-matched codec settings + hardware acceleration (VideoToolbox on macOS, fallback to libx264)
5. **Concatenate** — ffmpeg concat demuxer to join all segments
6. **Cleanup** — remove temporary segments and frames

### 5. Pipeline Integration

**File:** `movie_translator/pipeline.py`

When `enable_ocr` is True and burned-in subtitles were detected:

1. The OCR step now returns both the SRT path and the `OCRResult` list
2. After muxing subtitle tracks (existing step), invoke the inpainting post-processor
3. The inpainting step operates on the already-muxed video (which has the new subtitle tracks) and produces the final output with burned-in text removed

### 6. FFmpeg Utilities

**File:** `movie_translator/ffmpeg.py`

Add helpers for:
- `probe_video_encoding()` — extract codec/profile/level/bitrate from source
- `extract_full_frames()` — extract full (uncropped) frames for a time range
- `encode_frames_to_segment()` — re-encode frames to video segment with matched settings
- `concat_segments()` — concatenate segments via concat demuxer

### 7. Dependencies

**File:** `pyproject.toml`

Add: `simple-lama-inpainting` (Apache 2.0, pip-installable, includes LaMa model download on first use)

## Re-encoding Strategy

- Probe source for: codec_name, profile, level, bit_rate, pix_fmt
- Re-encode with: same codec + profile, CRF 18 (visually lossless)
- Hardware acceleration: `h264_videotoolbox` / `hevc_videotoolbox` on macOS when available, fallback to `libx264` / `libx265`
- Audio: always stream copy (never re-encode audio)

## Scope Boundaries

**In scope:**
- LaMa frame-by-frame inpainting using OCR bounding boxes as masks
- Segment-based re-encoding to minimize re-encode cost
- Hardware-accelerated encoding on macOS
- Visual quality review via sample frame inspection

**Out of scope (future work):**
- Video-aware temporal inpainting (STTN/ProPainter) — upgrade path documented in memory
- Per-character or per-glyph masking — bounding box rectangles are sufficient for v1
- Non-macOS OCR backends — Vision framework remains the only OCR source
- Encoding preset configurability via CLI — hardcoded CRF 18 for now
