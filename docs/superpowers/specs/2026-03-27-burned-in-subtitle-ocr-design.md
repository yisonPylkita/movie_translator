# Burned-In Subtitle OCR Extraction

**Date:** 2026-03-27
**Status:** Draft
**Scope:** Extract burned-in (hardcoded) English subtitles from video frames using OCR, producing `DialogueLine` objects that feed into the existing translation pipeline.

## Problem

Many video files (especially MP4s) have subtitles burned directly into the video stream rather than as separate subtitle tracks. The current pipeline only handles extractable subtitle tracks (ASS, SRT, PGS). When no subtitle tracks exist, processing fails silently.

The example data in `movies/` demonstrates this: two MP4 files with English subtitles baked into the video frames, zero subtitle tracks present.

## Solution Overview

Add a burned-in subtitle extraction pipeline that:

1. Extracts cropped frames (bottom 25% of video) via FFmpeg
2. Detects subtitle transitions using pixel-diff change detection
3. Runs Apple Vision OCR only on transition frames
4. Produces `list[DialogueLine]` for the existing translation pipeline

This is a **macOS-only** feature, leveraging the Apple Neural Engine for fast, accurate on-device text recognition.

## Architecture

### New Module Structure

```
movie_translator/ocr/
├── __init__.py
├── frame_extractor.py      # FFmpeg frame extraction + cropping
├── change_detector.py       # Pixel-diff subtitle region change detection
├── vision_ocr.py            # Apple Vision framework text recognition
└── burned_in_extractor.py   # Orchestrator: frames -> changes -> OCR -> DialogueLines
```

### Data Flow

```
Video file (MP4/MKV, no subtitle tracks)
  -> frame_extractor: extract bottom 25% crops at 10fps via FFmpeg
  -> change_detector: pixel-diff consecutive crops, identify transitions
  -> vision_ocr: OCR only on transition frames (Apple Vision, Neural Engine)
  -> burned_in_extractor: build DialogueLine(start_ms, end_ms, text) list
  -> (existing pipeline: translate -> create subtitles -> mux)
```

### Integration Point

In `pipeline.py:_extract_subtitles()`, after track-based extraction fails (no subtitle tracks found), fall back to burned-in OCR:

```python
if not eng_track:
    if self._can_try_burned_in_ocr():
        return self._extract_burned_in_subtitles(video_path, output_dir)
    logger.error('No English subtitle track found')
    return None
```

The burned-in path produces an SRT file (no styling to preserve), which `SubtitleProcessor.extract_dialogue_lines()` already handles.

**Trigger conditions** (all must be true):
- No English subtitle tracks found in the video
- `--enable-ocr` flag passed (opt-in, reusing existing CLI flag)
- Platform is macOS (Vision framework available)

### Prerequisite: MP4 Muxing Support

The current `mux_video_with_subtitles()` in `ffmpeg.py` hardcodes `-c:s ass`, which is incompatible with MP4 containers. For MP4 output, the subtitle codec must be `mov_text`. The mux function needs to detect the output container format and select the appropriate subtitle codec:
- `.mkv` -> `-c:s ass`
- `.mp4` -> `-c:s mov_text`

Since burned-in subtitle extraction produces plain text (no ASS styling), `mov_text` is a lossless representation — nothing is lost.

### Prerequisite: MP4 File Discovery

The current `find_mkv_files_with_temp_dirs()` in `main.py` only globs for `*.mkv`. As part of this work, widen it to also discover `*.mp4` files. This is a minimal change (adjust the glob patterns and rename the function to `find_video_files_with_temp_dirs`). Without this, the OCR pipeline can never trigger for MP4 input.

## Component Details

### 1. Frame Extractor (`frame_extractor.py`)

Uses FFmpeg to extract only the bottom 25% of each frame at 10fps:

```
ffmpeg -i video.mp4 -vf "crop=iw:ih/4:0:ih*3/4,fps=10" -q:v 2 output/%06d.jpg
```

- `crop=iw:ih/4:0:ih*3/4` — full width, bottom quarter of height
- `fps=10` — one frame every 100ms
- JPEG output (lossy is fine for OCR, fast to write/read)

**Performance estimate** for a 30min video at 480p:
- ~18,000 cropped frames
- Each crop ~640x120px, ~5-10KB JPEG
- ~100-180MB temp disk space

**Interface:**
```python
def extract_subtitle_region_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 10,
    crop_ratio: float = 0.25,
) -> list[tuple[Path, int]]:
    """Returns list of (frame_path, timestamp_ms) sorted by timestamp."""
```

### 2. Change Detector (`change_detector.py`)

Compares consecutive cropped frames to find subtitle transitions without running OCR.

**Algorithm:**
1. Load each cropped JPEG as a grayscale numpy array
2. Compute absolute pixel difference between consecutive frames
3. Calculate mean difference value
4. If mean diff exceeds `change_threshold` -> transition detected
5. Classify transitions: "appeared" (low variance -> high variance) or "disappeared" (high variance -> low variance)

**Key parameters:**
- `change_threshold`: mean pixel diff to trigger transition (default: 15 on 0-255 scale, tunable)
- `empty_variance_threshold`: frames with variance below this are "no subtitle" (default: tunable during testing)

**Interface:**
```python
@dataclass
class SubtitleTransition:
    timestamp_ms: int
    frame_path: Path
    event_type: Literal['appeared', 'disappeared']

def detect_transitions(
    frames: list[tuple[Path, int]],
    change_threshold: float = 15.0,
) -> list[SubtitleTransition]:
    """Identify frames where subtitle text changes."""
```

**Expected reduction:** ~18,000 frames -> ~100-300 transitions for a typical 30min episode.

### 3. Apple Vision OCR (`vision_ocr.py`)

Wraps macOS Vision framework via `pyobjc-framework-Vision`.

**Implementation:**
- Uses `VNRecognizeTextRequest` with accurate recognition level
- Language hints set to `["en"]`
- Processes via `VNImageRequestHandler` with `CGImage` loaded from JPEG
- Runs on Neural Engine automatically (no explicit device management)
- Returns recognized text or empty string

**Interface:**
```python
def recognize_text(image_path: Path, language: str = 'en') -> str:
    """Run Apple Vision OCR on an image. Returns recognized text."""

def is_available() -> bool:
    """Check if Apple Vision framework is available on this platform."""
```

**Dependencies (macOS-only optional):**
- `pyobjc-framework-Vision`
- `pyobjc-framework-Quartz`

Added to `pyproject.toml` as a new optional dependency group.

### 4. Orchestrator (`burned_in_extractor.py`)

Ties the components together:

1. Call `extract_subtitle_region_frames()` -> cropped frames with timestamps
2. Call `detect_transitions()` -> transition events
3. For each `appeared` transition: call `recognize_text()` on that frame
4. Build `DialogueLine` entries:
   - `start_ms` = timestamp of "appeared" event
   - `end_ms` = timestamp of next transition event (appeared or disappeared)
   - `text` = OCR result from the appeared frame
5. Filter out empty/garbage OCR results (empty string, single characters)
6. Merge consecutive lines with identical text (safety dedup)
7. Write results to SRT file
8. Return path to SRT file

**Interface:**
```python
def extract_burned_in_subtitles(
    video_path: Path,
    output_dir: Path,
    crop_ratio: float = 0.25,
    fps: int = 10,
) -> Path | None:
    """Extract burned-in subtitles via OCR. Returns path to SRT file or None."""
```

## Dependencies

New optional dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
vision-ocr = [
    "pyobjc-framework-Vision>=11.0; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz>=11.0; sys_platform == 'darwin'",
    "numpy>=1.26",
]
```

Note: `numpy` is needed for the pixel-diff change detection. It may already be installed transitively via `torch`, but we declare it explicitly for the case where someone installs only the OCR extra without the full translation stack.

The existing `ocr` optional group (opencv, paddleocr, paddlepaddle) is replaced by `vision-ocr`.

## Error Handling

- **Vision framework unavailable** (Linux, old macOS): `is_available()` returns False, feature gracefully disabled. CLI prints a clear message.
- **FFmpeg frame extraction fails:** raise with context, pipeline reports failure for this file.
- **OCR returns empty for all frames:** log warning ("no burned-in subtitles detected or OCR failed"), return None.
- **Too few transitions detected** (< 5 for a 10+ minute video): log warning suggesting the video may not have burned-in subtitles.
- **Temp directory cleanup:** frame JPEGs are cleaned up after processing, regardless of success/failure.

## Testing Strategy

### Unit Tests

- **`change_detector`**: Feed synthetic image pairs (identical frames -> no transition, different frames -> transition detected). Test threshold edge cases.
- **`vision_ocr`**: Generate an image with known text using PIL/Pillow, run OCR, verify output matches. Skip on non-macOS (pytest mark).
- **`burned_in_extractor`**: Mock frame extraction and OCR, verify DialogueLine timing/dedup logic.

### Integration Validation (Multimodal)

Use Claude's multimodal capability to validate OCR accuracy against the example data:

1. Extract ~10-15 frames at various timestamps from the example MP4 using FFmpeg
2. Visually read the burned-in text from each frame
3. Run the OCR pipeline on the same video
4. Compare visual reading against OCR output
5. Document the comparison as a validation record

### Performance Benchmark

Time the full OCR pipeline on both example MP4 files (~31 min each). Report:
- Total frames extracted
- Transitions detected
- OCR calls made
- Total wall-clock time
- Per-OCR-call average time

## Scope Boundaries

**In scope:**
- Burned-in subtitle extraction from video frames
- Bottom-crop heuristic (configurable ratio, default 25%)
- Pixel-diff change detection to minimize OCR calls
- Apple Vision OCR on macOS
- SRT output for downstream pipeline consumption
- Integration with existing `--enable-ocr` flag
- MP4 file discovery (widening the glob in `main.py`)
- MP4 muxing with `mov_text` subtitle codec

**Out of scope:**
- Linux/Windows OCR support
- Subtitle position detection (we assume bottom of frame)
- Multi-language OCR (English only for now)
- Subtitle styling extraction (colors, fonts — plain text only)
- MP4 container output support (separate feature)
- Re-encoding video to remove burned-in subtitles
