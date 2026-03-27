# Burned-In Subtitle Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove burned-in subtitles from video frames using LaMa AI inpainting, so translated subtitle tracks display without overlap.

**Architecture:** The OCR step is extended to preserve bounding box coordinates. After translation and subtitle muxing, a new post-processing step pipes the video through FFmpeg (decode → Python inpainting → encode) replacing subtitle regions with LaMa-reconstructed backgrounds. A reader thread prevents pipe deadlocks.

**Tech Stack:** simple-lama-inpainting (LaMa model), Apple Vision OCR (bounding boxes), FFmpeg (decode/encode pipeline), PIL/numpy (frame manipulation), VideoToolbox (hardware encoding on macOS)

**Note:** The design spec mentions segment-based re-encoding (only re-encoding subtitle ranges, stream-copying the rest). This plan implements the simpler full re-encode via FFmpeg pipes for v1. Frames without subtitles pass through untouched at the byte level but are still re-encoded. With hardware acceleration (VideoToolbox), this is fast enough for most movies. Segment-based optimization can be added later if needed.

---

## File Structure

### New files
- `movie_translator/inpainting/__init__.py` — Module exports
- `movie_translator/inpainting/mask_generator.py` — Binary mask generation from OCR bounding boxes
- `movie_translator/inpainting/inpainter.py` — LaMa inpainting wrapper
- `movie_translator/inpainting/video_processor.py` — Pipe-based video re-encoding with per-frame inpainting
- `movie_translator/inpainting/tests/__init__.py`
- `movie_translator/inpainting/tests/test_mask_generator.py`
- `movie_translator/inpainting/tests/test_inpainter.py`
- `movie_translator/inpainting/tests/test_video_processor.py`

### Modified files
- `movie_translator/types.py` — Add BoundingBox, OCRResult, BurnedInResult
- `movie_translator/ocr/vision_ocr.py` — Add recognize_text_with_boxes()
- `movie_translator/ocr/burned_in_extractor.py` — Return OCR results with mapped bounding boxes
- `movie_translator/ocr/__init__.py` — Update exports
- `movie_translator/ocr/tests/test_vision_ocr.py` — Test bounding box extraction
- `movie_translator/ocr/tests/test_burned_in_extractor.py` — Test coordinate mapping
- `movie_translator/ffmpeg.py` — Add probe_video_encoding()
- `movie_translator/tests/test_ffmpeg.py` — Test probe function
- `movie_translator/pipeline.py` — Wire inpainting into post-processing
- `pyproject.toml` — Add simple-lama-inpainting dependency

---

### Task 1: Add types and dependency

**Files:**
- Modify: `movie_translator/types.py`
- Modify: `pyproject.toml`
- Test: `movie_translator/ocr/tests/test_burned_in_extractor.py` (types are tested via usage in later tasks — verify import here)

- [ ] **Step 1: Add new types to types.py**

```python
# Add after DialogueLine class in movie_translator/types.py:

class BoundingBox(NamedTuple):
    x: float  # normalized 0-1, left edge
    y: float  # normalized 0-1, top edge (top-left origin)
    width: float  # normalized 0-1
    height: float  # normalized 0-1


class OCRResult(NamedTuple):
    timestamp_ms: int
    text: str
    boxes: list[BoundingBox]


class BurnedInResult(NamedTuple):
    srt_path: Path
    ocr_results: list[OCRResult]
```

- [ ] **Step 2: Add simple-lama-inpainting dependency to pyproject.toml**

Add to the `dependencies` list:

```toml
    "simple-lama-inpainting>=0.1.0",
```

- [ ] **Step 3: Install dependencies and verify import**

Run: `cd /Users/w/h_dev/movie_translator && uv sync`

Then verify:

Run: `uv run python -c "from movie_translator.types import BoundingBox, OCRResult, BurnedInResult; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add movie_translator/types.py pyproject.toml uv.lock
git commit -m "feat: add inpainting types and simple-lama-inpainting dependency"
```

---

### Task 2: Vision OCR with bounding boxes

**Files:**
- Modify: `movie_translator/ocr/vision_ocr.py`
- Test: `movie_translator/ocr/tests/test_vision_ocr.py`

- [ ] **Step 1: Write failing test for recognize_text_with_boxes**

Add to `movie_translator/ocr/tests/test_vision_ocr.py`:

```python
def test_recognize_text_with_boxes_returns_bounding_boxes(image_with_text):
    from movie_translator.ocr.vision_ocr import recognize_text_with_boxes

    results = recognize_text_with_boxes(image_with_text)

    assert len(results) >= 1
    text, box = results[0]
    assert 'hello' in text.lower() or 'Hello' in text
    # Bounding box should be normalized 0-1
    assert 0.0 <= box.x <= 1.0
    assert 0.0 <= box.y <= 1.0
    assert 0.0 < box.width <= 1.0
    assert 0.0 < box.height <= 1.0


def test_recognize_text_with_boxes_empty_for_blank(blank_image):
    from movie_translator.ocr.vision_ocr import recognize_text_with_boxes

    results = recognize_text_with_boxes(blank_image)

    assert results == []


def test_recognize_text_with_boxes_empty_for_nonexistent():
    from movie_translator.ocr.vision_ocr import recognize_text_with_boxes

    results = recognize_text_with_boxes(Path('/nonexistent/image.png'))

    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/ocr/tests/test_vision_ocr.py::test_recognize_text_with_boxes_returns_bounding_boxes -v`

Expected: FAIL — `ImportError: cannot import name 'recognize_text_with_boxes'`

- [ ] **Step 3: Implement recognize_text_with_boxes and refactor recognize_text**

Replace the full content of `movie_translator/ocr/vision_ocr.py`:

```python
import sys
from pathlib import Path

from ..logging import logger
from ..types import BoundingBox


def is_available() -> bool:
    if sys.platform != 'darwin':
        return False
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401

        return True
    except ImportError:
        return False


def recognize_text_with_boxes(
    image_path: Path, language: str = 'en'
) -> list[tuple[str, BoundingBox]]:
    """Recognize text in image, returning text and bounding boxes.

    Bounding boxes are normalized (0-1) with top-left origin.
    """
    if not image_path.exists():
        return []

    try:
        import Quartz
        import Vision

        url = Quartz.CFURLCreateWithFileSystemPath(
            None, str(image_path), Quartz.kCFURLPOSIXPathStyle, False
        )
        image_source = Quartz.CGImageSourceCreateWithURL(url, None)
        if image_source is None:
            logger.debug(f'Could not create image source for {image_path}')
            return []

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            logger.debug(f'Could not load image from {image_path}')
            return []

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        success = handler.performRequests_error_([request], None)
        if not success[0]:
            logger.debug(f'Vision request failed for {image_path}')
            return []

        observations = request.results()
        if not observations:
            return []

        results: list[tuple[str, BoundingBox]] = []
        for observation in observations:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            text = candidates[0].string()
            bbox = observation.boundingBox()
            # Vision uses bottom-left origin — convert to top-left
            box = BoundingBox(
                x=bbox.origin.x,
                y=1.0 - bbox.origin.y - bbox.size.height,
                width=bbox.size.width,
                height=bbox.size.height,
            )
            results.append((text, box))

        return results

    except Exception as e:
        logger.debug(f'OCR error for {image_path}: {e}')
        return []


def recognize_text(image_path: Path, language: str = 'en') -> str:
    """Recognize text in image, returning concatenated text string."""
    results = recognize_text_with_boxes(image_path, language)
    return '\n'.join(text for text, _ in results)
```

- [ ] **Step 4: Run all Vision OCR tests**

Run: `uv run pytest movie_translator/ocr/tests/test_vision_ocr.py -v`

Expected: All tests pass (including existing tests that use `recognize_text`).

- [ ] **Step 5: Commit**

```bash
git add movie_translator/ocr/vision_ocr.py movie_translator/ocr/tests/test_vision_ocr.py
git commit -m "feat: add recognize_text_with_boxes to Vision OCR"
```

---

### Task 3: Burned-in extractor returns bounding boxes

**Files:**
- Modify: `movie_translator/ocr/burned_in_extractor.py`
- Modify: `movie_translator/ocr/__init__.py`
- Test: `movie_translator/ocr/tests/test_burned_in_extractor.py`

- [ ] **Step 1: Write failing test for coordinate mapping**

Add to `movie_translator/ocr/tests/test_burned_in_extractor.py`:

```python
from movie_translator.types import BoundingBox


class TestMapBoxToFullFrame:
    def test_maps_crop_coordinates_to_full_frame(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        # Box in middle of crop (crop_ratio=0.25, crop covers bottom 25%)
        crop_box = BoundingBox(x=0.1, y=0.3, width=0.8, height=0.2)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        assert result.x == 0.1  # x unchanged
        assert result.width == 0.8  # width unchanged
        # y should map from crop-space to full-frame: 0.75 + 0.3*0.25 = 0.825
        assert abs(result.y - 0.825) < 1e-9
        # height scales by crop_ratio: 0.2 * 0.25 = 0.05
        assert abs(result.height - 0.05) < 1e-9

    def test_top_of_crop_maps_to_crop_start(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        crop_box = BoundingBox(x=0.0, y=0.0, width=1.0, height=0.1)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        # y=0 in crop → y=0.75 in full frame (top of bottom 25%)
        assert abs(result.y - 0.75) < 1e-9

    def test_bottom_of_crop_maps_to_frame_bottom(self):
        from movie_translator.ocr.burned_in_extractor import _map_box_to_full_frame

        crop_box = BoundingBox(x=0.0, y=0.9, width=1.0, height=0.1)
        result = _map_box_to_full_frame(crop_box, crop_ratio=0.25)

        # y=0.9 in crop → 0.75 + 0.9*0.25 = 0.975
        assert abs(result.y - 0.975) < 1e-9
        # height: 0.1 * 0.25 = 0.025, so bottom edge at 0.975+0.025 = 1.0
        assert abs(result.y + result.height - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/ocr/tests/test_burned_in_extractor.py::TestMapBoxToFullFrame -v`

Expected: FAIL — `ImportError: cannot import name '_map_box_to_full_frame'`

- [ ] **Step 3: Implement coordinate mapping and modify extractor**

Replace full content of `movie_translator/ocr/burned_in_extractor.py`:

```python
import shutil
from pathlib import Path

from ..logging import logger
from ..types import BoundingBox, BurnedInResult, DialogueLine, OCRResult
from .frame_extractor import extract_subtitle_region_frames
from .vision_ocr import recognize_text_with_boxes


def _map_box_to_full_frame(box: BoundingBox, crop_ratio: float) -> BoundingBox:
    """Map bounding box from cropped frame coordinates to full frame coordinates."""
    return BoundingBox(
        x=box.x,
        y=(1 - crop_ratio) + (box.y * crop_ratio),
        width=box.width,
        height=box.height * crop_ratio,
    )


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
    crop_ratio: float = 0.25,
    fps: int = 1,
) -> BurnedInResult | None:
    """Extract burned-in subtitles via OCR, returning SRT path and per-frame bounding boxes."""
    frames_dir = output_dir / '_ocr_frames'

    try:
        frames = extract_subtitle_region_frames(
            video_path, frames_dir, fps=fps, crop_ratio=crop_ratio
        )
        if not frames:
            logger.error('No frames extracted from video')
            return None

        logger.info(f'Running OCR on {len(frames)} frames...')
        frame_texts: list[tuple[int, str]] = []
        ocr_results: list[OCRResult] = []

        for i, (frame_path, timestamp_ms) in enumerate(frames):
            text_boxes = recognize_text_with_boxes(frame_path)
            text = '\n'.join(t for t, _ in text_boxes).strip()
            frame_texts.append((timestamp_ms, text))

            # Map bounding boxes from crop-space to full-frame coordinates
            full_frame_boxes = [
                _map_box_to_full_frame(box, crop_ratio) for _, box in text_boxes
            ]
            ocr_results.append(OCRResult(timestamp_ms, text, full_frame_boxes))

            if (i + 1) % 100 == 0:
                logger.info(f'  OCR progress: {i + 1}/{len(frames)}')

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
```

- [ ] **Step 4: Update ocr/__init__.py exports**

Replace `movie_translator/ocr/__init__.py`:

```python
from .burned_in_extractor import extract_burned_in_subtitles
from .vision_ocr import is_available as is_vision_ocr_available

__all__ = ['extract_burned_in_subtitles', 'is_vision_ocr_available']
```

(No change needed — the function signature changed but the name/export is the same.)

- [ ] **Step 5: Run all extractor tests**

Run: `uv run pytest movie_translator/ocr/tests/test_burned_in_extractor.py -v`

Expected: All tests pass (existing + new coordinate mapping tests).

- [ ] **Step 6: Commit**

```bash
git add movie_translator/ocr/burned_in_extractor.py movie_translator/ocr/__init__.py movie_translator/ocr/tests/test_burned_in_extractor.py
git commit -m "feat: return bounding boxes from burned-in extractor with coordinate mapping"
```

---

### Task 4: Mask generator

**Files:**
- Create: `movie_translator/inpainting/__init__.py`
- Create: `movie_translator/inpainting/mask_generator.py`
- Create: `movie_translator/inpainting/tests/__init__.py`
- Create: `movie_translator/inpainting/tests/test_mask_generator.py`

- [ ] **Step 1: Create module structure**

Create `movie_translator/inpainting/__init__.py`:

```python
```

Create `movie_translator/inpainting/tests/__init__.py`:

```python
```

- [ ] **Step 2: Write failing test for generate_mask**

Create `movie_translator/inpainting/tests/test_mask_generator.py`:

```python
from movie_translator.inpainting.mask_generator import generate_mask
from movie_translator.types import BoundingBox


class TestGenerateMask:
    def test_creates_mask_with_correct_dimensions(self):
        boxes = [BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)]
        mask = generate_mask(boxes, frame_width=1920, frame_height=1080)

        assert mask.size == (1920, 1080)
        assert mask.mode == 'L'

    def test_white_pixels_in_box_region(self):
        boxes = [BoundingBox(x=0.25, y=0.75, width=0.5, height=0.2)]
        mask = generate_mask(boxes, frame_width=100, frame_height=100, dilation_px=0)

        # Center of box region should be white (255)
        assert mask.getpixel((50, 85)) == 255

    def test_black_pixels_outside_box_region(self):
        boxes = [BoundingBox(x=0.25, y=0.75, width=0.5, height=0.2)]
        mask = generate_mask(boxes, frame_width=100, frame_height=100, dilation_px=0)

        # Top-left corner should be black (0)
        assert mask.getpixel((5, 5)) == 0

    def test_dilation_expands_mask(self):
        boxes = [BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1)]
        mask_no_dilation = generate_mask(boxes, 200, 200, dilation_px=0)
        mask_with_dilation = generate_mask(boxes, 200, 200, dilation_px=20)

        # Count white pixels — dilated mask should have more
        no_dil_count = sum(1 for p in mask_no_dilation.getdata() if p == 255)
        dil_count = sum(1 for p in mask_with_dilation.getdata() if p == 255)
        assert dil_count > no_dil_count

    def test_multiple_boxes(self):
        boxes = [
            BoundingBox(x=0.1, y=0.1, width=0.2, height=0.1),
            BoundingBox(x=0.6, y=0.8, width=0.3, height=0.1),
        ]
        mask = generate_mask(boxes, 100, 100, dilation_px=0)

        # Both regions should be white
        assert mask.getpixel((20, 15)) == 255
        assert mask.getpixel((75, 85)) == 255

    def test_empty_boxes_returns_black_mask(self):
        mask = generate_mask([], frame_width=100, frame_height=100)

        assert all(p == 0 for p in mask.getdata())

    def test_dilation_clamps_to_frame_bounds(self):
        # Box near edge — dilation shouldn't exceed frame bounds
        boxes = [BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)]
        mask = generate_mask(boxes, 100, 100, dilation_px=20)

        # Should not raise, and top-left should be white
        assert mask.getpixel((0, 0)) == 255
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest movie_translator/inpainting/tests/test_mask_generator.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'movie_translator.inpainting.mask_generator'`

- [ ] **Step 4: Implement mask generator**

Create `movie_translator/inpainting/mask_generator.py`:

```python
from PIL import Image, ImageDraw

from ..types import BoundingBox


def generate_mask(
    boxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
    dilation_px: int = 20,
) -> Image.Image:
    """Generate a binary mask from bounding boxes. White (255) = inpaint region."""
    mask = Image.new('L', (frame_width, frame_height), 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        x1 = max(0, int(box.x * frame_width) - dilation_px)
        y1 = max(0, int(box.y * frame_height) - dilation_px)
        x2 = min(frame_width, int((box.x + box.width) * frame_width) + dilation_px)
        y2 = min(frame_height, int((box.y + box.height) * frame_height) + dilation_px)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest movie_translator/inpainting/tests/test_mask_generator.py -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add movie_translator/inpainting/
git commit -m "feat: add mask generator for inpainting bounding box regions"
```

---

### Task 5: LaMa inpainter wrapper

**Files:**
- Create: `movie_translator/inpainting/inpainter.py`
- Create: `movie_translator/inpainting/tests/test_inpainter.py`

- [ ] **Step 1: Write failing test for Inpainter**

Create `movie_translator/inpainting/tests/test_inpainter.py`:

```python
import numpy as np
import pytest
from PIL import Image

from movie_translator.inpainting.inpainter import Inpainter


@pytest.mark.slow
class TestInpainter:
    def test_inpaints_masked_region(self):
        # Create a red image with a white rectangle (simulating subtitle text)
        image = Image.new('RGB', (256, 256), (180, 60, 60))
        pixels = image.load()
        for x in range(80, 176):
            for y in range(200, 240):
                pixels[x, y] = (255, 255, 255)

        # Mask covers the white rectangle
        mask = Image.new('L', (256, 256), 0)
        mask_pixels = mask.load()
        for x in range(70, 186):
            for y in range(190, 250):
                mask_pixels[x, y] = 255

        inpainter = Inpainter(device='cpu')
        result = inpainter.inpaint(image, mask)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)

        # The masked region should no longer be pure white
        result_array = np.array(result)
        masked_region = result_array[200:240, 80:176]
        mean_brightness = masked_region.mean()
        # Inpainted region should blend with red background, not stay white
        assert mean_brightness < 220

    def test_unmasked_image_unchanged(self):
        image = Image.new('RGB', (128, 128), (100, 150, 200))
        mask = Image.new('L', (128, 128), 0)  # empty mask

        inpainter = Inpainter(device='cpu')
        result = inpainter.inpaint(image, mask)

        # With empty mask, result should be very similar to input
        diff = np.abs(np.array(result).astype(float) - np.array(image).astype(float))
        assert diff.mean() < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/inpainting/tests/test_inpainter.py -v -m slow`

Expected: FAIL — `ModuleNotFoundError: No module named 'movie_translator.inpainting.inpainter'`

- [ ] **Step 3: Implement Inpainter**

Create `movie_translator/inpainting/inpainter.py`:

```python
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from ..logging import logger


class Inpainter:
    """Wraps LaMa model for single-image inpainting."""

    def __init__(self, device: str = 'cpu'):
        torch_device = torch.device(device)
        try:
            self._model = SimpleLama(device=torch_device)
        except Exception:
            logger.warning(f'LaMa failed to load on {device}, falling back to CPU')
            self._model = SimpleLama(device=torch.device('cpu'))

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint masked regions of the image.

        Args:
            image: RGB input image.
            mask: Grayscale mask. White (255) = regions to inpaint.

        Returns:
            Inpainted RGB image.
        """
        return self._model(image, mask.convert('L'))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/inpainting/tests/test_inpainter.py -v -m slow`

Expected: All tests pass (first run downloads ~200MB LaMa model).

- [ ] **Step 5: Commit**

```bash
git add movie_translator/inpainting/inpainter.py movie_translator/inpainting/tests/test_inpainter.py
git commit -m "feat: add LaMa inpainter wrapper"
```

---

### Task 6: FFmpeg encoding probe

**Files:**
- Modify: `movie_translator/ffmpeg.py`
- Test: `movie_translator/tests/test_ffmpeg.py`

- [ ] **Step 1: Write failing test for probe_video_encoding**

Add to `movie_translator/tests/test_ffmpeg.py` (create if it doesn't exist, or add to existing file):

```python
import subprocess

import pytest

from movie_translator.ffmpeg import get_ffmpeg, probe_video_encoding


@pytest.fixture
def sample_video(tmp_path):
    """Create a short test video with known properties."""
    ffmpeg = get_ffmpeg()
    output = tmp_path / 'test.mp4'
    cmd = [
        ffmpeg, '-y',
        '-f', 'lavfi', '-i', 'color=black:s=320x240:d=1:r=24',
        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
        '-t', '1',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-c:a', 'aac', '-b:a', '128k',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test video: {result.stderr}')
    return output


class TestProbeVideoEncoding:
    def test_returns_codec_info(self, sample_video):
        info = probe_video_encoding(sample_video)

        assert info['codec_name'] == 'h264'
        assert info['width'] == 320
        assert info['height'] == 240
        assert abs(info['fps'] - 24.0) < 0.1

    def test_returns_pix_fmt(self, sample_video):
        info = probe_video_encoding(sample_video)

        assert info['pix_fmt'] == 'yuv420p'

    def test_raises_for_nonexistent(self, tmp_path):
        with pytest.raises(Exception):
            probe_video_encoding(tmp_path / 'nonexistent.mp4')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/tests/test_ffmpeg.py::TestProbeVideoEncoding -v`

Expected: FAIL — `ImportError: cannot import name 'probe_video_encoding'`

- [ ] **Step 3: Implement probe_video_encoding**

Add to `movie_translator/ffmpeg.py` after `get_video_info`:

```python
def probe_video_encoding(video_path: Path) -> dict[str, Any]:
    """Extract video encoding parameters for re-encoding."""
    info = get_video_info(video_path)

    video_stream = next(
        (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
        None,
    )
    if not video_stream:
        raise VideoMuxError(f'No video stream found in {video_path}')

    # Parse frame rate from r_frame_rate (e.g., "24/1" or "24000/1001")
    r_frame_rate = video_stream.get('r_frame_rate', '24/1')
    num, den = map(int, r_frame_rate.split('/'))
    fps = num / den

    # Bitrate may be per-stream or in format-level
    bit_rate = video_stream.get('bit_rate') or info.get('format', {}).get('bit_rate', '5000000')

    return {
        'codec_name': video_stream.get('codec_name', 'h264'),
        'profile': video_stream.get('profile', ''),
        'width': video_stream.get('width', 1920),
        'height': video_stream.get('height', 1080),
        'bit_rate': str(bit_rate),
        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
        'fps': fps,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/tests/test_ffmpeg.py::TestProbeVideoEncoding -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/ffmpeg.py movie_translator/tests/test_ffmpeg.py
git commit -m "feat: add probe_video_encoding for re-encoding parameter matching"
```

---

### Task 7: Video inpainting processor

**Files:**
- Create: `movie_translator/inpainting/video_processor.py`
- Create: `movie_translator/inpainting/tests/test_video_processor.py`

- [ ] **Step 1: Write failing test for _build_subtitle_lookup**

Create `movie_translator/inpainting/tests/test_video_processor.py`:

```python
from movie_translator.inpainting.video_processor import _build_subtitle_lookup
from movie_translator.types import BoundingBox, OCRResult


class TestBuildSubtitleLookup:
    def test_maps_ocr_result_to_frame_range(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [OCRResult(timestamp_ms=1000, text='Hello', boxes=[box])]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        # 1000ms at 24fps = frame 24. Should cover frames 24-47 (one second)
        assert 24 in lookup
        assert 47 in lookup
        assert 23 not in lookup
        assert 48 not in lookup
        assert lookup[24] == [box]

    def test_skips_empty_text(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [
            OCRResult(timestamp_ms=0, text='', boxes=[]),
            OCRResult(timestamp_ms=1000, text='Hello', boxes=[box]),
        ]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        # Frame 0-23 should NOT be in lookup (empty text)
        assert 0 not in lookup
        assert 23 not in lookup
        # Frame 24+ should be
        assert 24 in lookup

    def test_consecutive_results_cover_continuous_range(self):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)
        results = [
            OCRResult(timestamp_ms=0, text='Hello', boxes=[box]),
            OCRResult(timestamp_ms=1000, text='Hello', boxes=[box]),
        ]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        # Should cover frames 0-47 (two full seconds)
        for f in range(48):
            assert f in lookup

    def test_empty_results(self):
        lookup = _build_subtitle_lookup([], fps=24.0)
        assert lookup == {}

    def test_skips_results_without_boxes(self):
        results = [OCRResult(timestamp_ms=0, text='Hello', boxes=[])]

        lookup = _build_subtitle_lookup(results, fps=24.0)

        assert lookup == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/inpainting/tests/test_video_processor.py::TestBuildSubtitleLookup -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement _build_subtitle_lookup**

Create `movie_translator/inpainting/video_processor.py`:

```python
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
```

- [ ] **Step 4: Run lookup tests**

Run: `uv run pytest movie_translator/inpainting/tests/test_video_processor.py::TestBuildSubtitleLookup -v`

Expected: All tests pass.

- [ ] **Step 5: Write integration test for remove_burned_in_subtitles**

Add to `movie_translator/inpainting/tests/test_video_processor.py`:

```python
import subprocess

import pytest

from movie_translator.ffmpeg import get_ffmpeg, probe_video_encoding
from movie_translator.inpainting.video_processor import remove_burned_in_subtitles


@pytest.fixture
def video_with_subtitle_text(tmp_path):
    """Create a 2-second video with burned-in text at the bottom."""
    ffmpeg = get_ffmpeg()
    output = tmp_path / 'input.mp4'
    cmd = [
        ffmpeg, '-y',
        '-f', 'lavfi', '-i', 'color=size=320x240:duration=2:rate=24:color=darkblue',
        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
        '-t', '2',
        '-vf', "drawtext=text='Test Subtitle':fontsize=20:fontcolor=white:x=(w-tw)/2:y=h-40",
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-c:a', 'aac', '-b:a', '128k',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test video: {result.stderr}')
    return output


@pytest.mark.slow
class TestRemoveBurnedInSubtitles:
    def test_produces_output_video(self, video_with_subtitle_text, tmp_path):
        box = BoundingBox(x=0.1, y=0.8, width=0.8, height=0.15)
        ocr_results = [
            OCRResult(timestamp_ms=0, text='Test Subtitle', boxes=[box]),
            OCRResult(timestamp_ms=1000, text='Test Subtitle', boxes=[box]),
        ]
        output = tmp_path / 'output.mp4'

        remove_burned_in_subtitles(
            video_with_subtitle_text, output, ocr_results, device='cpu',
        )

        assert output.exists()
        assert output.stat().st_size > 0

        info = probe_video_encoding(output)
        assert info['width'] == 320
        assert info['height'] == 240
```

- [ ] **Step 6: Implement remove_burned_in_subtitles**

Add to `movie_translator/inpainting/video_processor.py` after `_build_subtitle_lookup`:

```python
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
        ffmpeg, '-i', str(video_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        'pipe:1',
    ]

    # Encoder: raw RGB24 frames → video, with audio from original
    codec, codec_args = _select_encoder(encoding)
    encode_cmd = [
        ffmpeg,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-i', str(video_path),
        '-map', '0:v', '-map', '1:a',
        '-c:v', codec, *codec_args,
        '-c:a', 'copy',
        '-y', str(output_path),
    ]

    decoder = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
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
```

- [ ] **Step 7: Run all video processor tests**

Run: `uv run pytest movie_translator/inpainting/tests/test_video_processor.py -v`

Expected: All tests pass (slow-marked tests will download LaMa model on first run).

- [ ] **Step 8: Commit**

```bash
git add movie_translator/inpainting/video_processor.py movie_translator/inpainting/tests/test_video_processor.py
git commit -m "feat: add video inpainting processor with pipe-based re-encoding"
```

---

### Task 8: Pipeline integration

**Files:**
- Modify: `movie_translator/pipeline.py`
- Modify: `movie_translator/inpainting/__init__.py`

- [ ] **Step 1: Update inpainting module exports**

Replace `movie_translator/inpainting/__init__.py`:

```python
from .video_processor import remove_burned_in_subtitles

__all__ = ['remove_burned_in_subtitles']
```

- [ ] **Step 2: Integrate inpainting into pipeline**

Replace the full content of `movie_translator/pipeline.py`:

```python
import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .inpainting import remove_burned_in_subtitles
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .types import OCRResult
from .video import VideoOperations


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_ocr: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_ocr = enable_ocr
        self._extractor = None
        self._video_ops = None
        self._ocr_results: list[OCRResult] | None = None

    def process_video_file(self, video_path: Path, temp_dir: Path, dry_run: bool = False) -> bool:
        logger.info(f'Processing: {video_path.name}')
        self._ocr_results = None

        try:
            extracted_ass = self._extract_subtitles(video_path, temp_dir)
            if not extracted_ass:
                return False

            logger.info('Parsing dialogue...')
            dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
            if not dialogue_lines:
                logger.error('No dialogue lines found')
                return False

            logger.info(f'Translating {len(dialogue_lines)} lines...')
            try:
                translated_dialogue = translate_dialogue_lines(
                    dialogue_lines, self.device, self.batch_size, self.model
                )
                if not translated_dialogue:
                    logger.error('Translation failed')
                    return False
            except Exception as e:
                logger.error(f'Translation failed: {e}')
                return False

            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            logger.info('Creating subtitle files...')
            clean_english_ass = temp_dir / f'{video_path.stem}_english_clean.ass'
            polish_ass = temp_dir / f'{video_path.stem}_polish.ass'

            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            replace_chars = not fonts_support_polish
            SubtitleProcessor.create_polish_subtitles(
                extracted_ass, translated_dialogue, polish_ass, replace_chars
            )

            # If burned-in subtitles were detected, inpaint them out first
            source_video = video_path
            if self._ocr_results:
                logger.info('Removing burned-in subtitles from video...')
                inpainted_video = temp_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
                remove_burned_in_subtitles(
                    video_path, inpainted_video, self._ocr_results, self.device,
                )
                source_video = inpainted_video

            logger.info('Creating video...')
            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(source_video, clean_english_ass, polish_ass, temp_video)
            video_ops.verify_result(temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            logger.info(f'Completed: {video_path.name}')
            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False

    def _get_extractor(self) -> SubtitleExtractor:
        """Lazy initialization of subtitle extractor."""
        if self._extractor is None:
            self._extractor = SubtitleExtractor(enable_ocr=self.enable_ocr)
        return self._extractor

    def _get_video_ops(self) -> VideoOperations:
        """Lazy initialization of video operations."""
        if self._video_ops is None:
            self._video_ops = VideoOperations()
        return self._video_ops

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('Extracting subtitles...')

        extractor = self._get_extractor()

        track_info = extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = extractor.find_english_track(track_info)
        if not eng_track:
            if self._can_try_burned_in_ocr():
                return self._extract_burned_in_subtitles(video_path, output_dir)
            logger.error('No English subtitle track found')
            return None

        track_id = eng_track['id']
        logger.info(f'Found English track: ID {track_id}')

        subtitle_ext = extractor.get_subtitle_extension(eng_track)
        extracted_sub = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'
        subtitle_index = eng_track.get('subtitle_index', 0)
        extractor.extract_subtitle(video_path, track_id, extracted_sub, subtitle_index)

        return extracted_sub

    def _can_try_burned_in_ocr(self) -> bool:
        if not self.enable_ocr:
            return False
        if not is_vision_ocr_available():
            logger.warning('Apple Vision OCR not available on this platform')
            return False
        return True

    def _extract_burned_in_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('No subtitle tracks found — attempting burned-in subtitle OCR...')
        result = extract_burned_in_subtitles(video_path, output_dir)
        if result is None:
            return None
        self._ocr_results = result.ocr_results
        return result.srt_path

    def _replace_original(self, video_path: Path, temp_video: Path) -> None:
        logger.info('Replacing original...')

        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)

        try:
            shutil.move(str(temp_video), str(video_path))
            video_ops = self._get_video_ops()
            video_ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest -v --ignore=movie_translator/inpainting/tests/test_inpainter.py --ignore=movie_translator/inpainting/tests/test_video_processor.py -m "not slow"`

Expected: All existing tests pass. The inpainter and video processor slow tests are excluded for speed.

- [ ] **Step 4: Run full test suite including slow tests**

Run: `uv run pytest -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/pipeline.py movie_translator/inpainting/__init__.py
git commit -m "feat: integrate burned-in subtitle inpainting into translation pipeline"
```
