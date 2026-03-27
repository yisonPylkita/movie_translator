# Burned-In Subtitle OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract burned-in English subtitles from video frames using Apple Vision OCR, producing DialogueLine objects for the existing translation pipeline.

**Architecture:** A four-component OCR pipeline (frame extraction, change detection, Vision OCR, orchestrator) that integrates as a fallback in the existing `_extract_subtitles()` method when no subtitle tracks are found. Two prerequisites (MP4 file discovery + MP4 muxing) unblock the pipeline for MP4 input files.

**Tech Stack:** Apple Vision framework (pyobjc), FFmpeg (frame extraction), numpy (pixel-diff change detection), pysubs2 (SRT output)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `movie_translator/main.py` | Widen file discovery to include MP4 |
| Modify | `movie_translator/ffmpeg.py` | Container-aware subtitle codec selection |
| Modify | `movie_translator/pipeline.py` | Add burned-in OCR fallback path |
| Modify | `movie_translator/ocr/__init__.py` | Re-export new modules |
| Replace | `movie_translator/ocr/processor.py` | Delete old PaddleOCR stub |
| Create | `movie_translator/ocr/vision_ocr.py` | Apple Vision text recognition wrapper |
| Create | `movie_translator/ocr/frame_extractor.py` | FFmpeg frame extraction + cropping |
| Create | `movie_translator/ocr/change_detector.py` | Pixel-diff subtitle transition detection |
| Create | `movie_translator/ocr/burned_in_extractor.py` | Orchestrator: frames -> changes -> OCR -> SRT |
| Modify | `tests/test_integration.py` | Update for renamed discovery function, add MP4 tests |
| Create | `movie_translator/ocr/tests/__init__.py` | Test package |
| Create | `movie_translator/ocr/tests/test_change_detector.py` | Change detection unit tests |
| Create | `movie_translator/ocr/tests/test_vision_ocr.py` | Vision OCR unit tests |
| Create | `movie_translator/ocr/tests/test_burned_in_extractor.py` | Orchestrator unit tests |
| Modify | `pyproject.toml` | Replace `ocr` optional group with `vision-ocr` |

---

### Task 1: MP4 File Discovery

Widen `find_mkv_files_with_temp_dirs()` to also discover `.mp4` files. Rename to `find_video_files_with_temp_dirs()`.

**Files:**
- Modify: `movie_translator/main.py:105-121` (discovery function), `main.py:42` (help text), `main.py:138-141` (call site + error message)
- Modify: `tests/test_integration.py:4` (import), `tests/test_integration.py:112-170` (TestFindMkvFiles class)

- [ ] **Step 1: Update the existing discovery tests for the new function name and MP4 support**

In `tests/test_integration.py`, rename the import, class, and add MP4 test cases:

```python
# Line 4: change import
from movie_translator.main import find_video_files_with_temp_dirs

# Line 112-170: rename class and update all calls, add MP4 tests
class TestFindVideoFiles:
    def test_finds_mkv_in_root_directory(self, tmp_path):
        mkv1 = tmp_path / 'video1.mkv'
        mkv2 = tmp_path / 'video2.mkv'
        mkv1.touch()
        mkv2.touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 2
        assert results[0][0] == mkv1
        assert results[1][0] == mkv2
        assert results[0][1] == tmp_path / '.translate_temp'
        assert (tmp_path / '.translate_temp').exists()

    def test_finds_mp4_in_root_directory(self, tmp_path):
        mp4 = tmp_path / 'video.mp4'
        mp4.touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 1
        assert results[0][0] == mp4

    def test_finds_mixed_formats_in_root_directory(self, tmp_path):
        mkv = tmp_path / 'video1.mkv'
        mp4 = tmp_path / 'video2.mp4'
        mkv.touch()
        mp4.touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 2

    def test_finds_mp4_in_subdirectories(self, tmp_path):
        subdir = tmp_path / 'Season 1'
        subdir.mkdir()
        (subdir / 'ep01.mp4').touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 1
        assert results[0][0].suffix == '.mp4'

    def test_finds_mkv_in_subdirectories(self, tmp_path):
        season1 = tmp_path / 'Season 1'
        season2 = tmp_path / 'Season 2'
        season1.mkdir()
        season2.mkdir()

        (season1 / 'ep01.mkv').touch()
        (season1 / 'ep02.mkv').touch()
        (season2 / 'ep01.mkv').touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 3
        assert results[0][1] == season1 / '.translate_temp'
        assert results[2][1] == season2 / '.translate_temp'
        assert (season1 / '.translate_temp').exists()
        assert (season2 / '.translate_temp').exists()

    def test_returns_empty_for_no_video_files(self, tmp_path):
        (tmp_path / 'readme.txt').touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert results == []

    def test_ignores_hidden_directories(self, tmp_path):
        hidden = tmp_path / '.hidden'
        hidden.mkdir()
        (hidden / 'video.mkv').touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert results == []

    def test_prefers_root_over_subdirectories(self, tmp_path):
        (tmp_path / 'root.mkv').touch()
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        (subdir / 'sub.mkv').touch()

        results = find_video_files_with_temp_dirs(tmp_path)

        assert len(results) == 1
        assert results[0][0].name == 'root.mkv'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_integration.py::TestFindVideoFiles -v`
Expected: FAIL (import error — `find_video_files_with_temp_dirs` does not exist yet)

- [ ] **Step 3: Implement the renamed function**

In `movie_translator/main.py`, replace `find_mkv_files_with_temp_dirs` with:

```python
VIDEO_EXTENSIONS = ('*.mkv', '*.mp4')


def find_video_files_with_temp_dirs(input_dir: Path) -> list[tuple[Path, Path]]:
    video_files_direct: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        video_files_direct.extend(input_dir.glob(ext))
    video_files_direct.sort()

    if video_files_direct:
        temp_dir = input_dir / '.translate_temp'
        temp_dir.mkdir(exist_ok=True)
        return [(f, temp_dir) for f in video_files_direct]

    results: list[tuple[Path, Path]] = []
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            video_files_in_subdir: list[Path] = []
            for ext in VIDEO_EXTENSIONS:
                video_files_in_subdir.extend(subdir.glob(ext))
            video_files_in_subdir.sort()

            if video_files_in_subdir:
                temp_dir = subdir / '.translate_temp'
                temp_dir.mkdir(exist_ok=True)
                results.extend((f, temp_dir) for f in video_files_in_subdir)

    return results
```

Also update all references in `main()`:
- Line 42: `help='Directory containing video files (MKV, MP4)'`
- Line 138: `video_files_with_temps = find_video_files_with_temp_dirs(input_dir)`
- Line 140-141: `if not video_files_with_temps:` and `'No video files found in {input_dir}'`
- Line 144: `total_files = len(video_files_with_temps)`
- Line 174: `for video_path, temp_dir in video_files_with_temps:`
- Rename all remaining `mkv_path` references in the loop to `video_path` and `mkv_files_with_temps` to `video_files_with_temps`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_integration.py::TestFindVideoFiles -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest -v`
Expected: ALL PASS (106+ tests)

- [ ] **Step 6: Commit**

```bash
git add movie_translator/main.py tests/test_integration.py
git commit -m "feat: widen file discovery to include MP4 files

Rename find_mkv_files_with_temp_dirs to find_video_files_with_temp_dirs.
Now discovers both *.mkv and *.mp4 files."
```

---

### Task 2: MP4 Muxing Support

Make `mux_video_with_subtitles()` select the subtitle codec based on output container format.

**Files:**
- Modify: `movie_translator/ffmpeg.py:97` (subtitle codec selection)
- Modify: `movie_translator/video/tests/test_operations.py` (add MP4 output test)

- [ ] **Step 1: Write test for MP4 muxing**

Add to `movie_translator/video/tests/test_operations.py`:

```python
def test_create_clean_video_mp4_output(self, create_test_mkv, create_ass_file, tmp_path):
    mkv_file = create_test_mkv()
    english_ass = create_ass_file('english.ass')
    polish_ass = create_ass_file('polish.ass')
    output_path = tmp_path / 'output.mp4'

    ops = VideoOperations()
    ops.create_clean_video(mkv_file, english_ass, polish_ass, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/video/tests/test_operations.py::TestVideoOperations::test_create_clean_video_mp4_output -v`
Expected: FAIL (FFmpeg error — ASS codec not supported in MP4 container)

- [ ] **Step 3: Implement container-aware codec selection**

In `movie_translator/ffmpeg.py`, replace the hardcoded `-c:s ass` line (line 97):

```python
    # Select subtitle codec based on output container
    subtitle_codec = 'mov_text' if output_path.suffix.lower() == '.mp4' else 'ass'
    cmd.extend(['-c:s', subtitle_codec])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/video/tests/test_operations.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/ffmpeg.py movie_translator/video/tests/test_operations.py
git commit -m "feat: support MP4 subtitle muxing with mov_text codec

Select subtitle codec based on output container: ass for MKV, mov_text for MP4."
```

---

### Task 3: Apple Vision OCR Wrapper

Create `vision_ocr.py` wrapping macOS Vision framework for text recognition.

**Files:**
- Create: `movie_translator/ocr/vision_ocr.py`
- Create: `movie_translator/ocr/tests/__init__.py`
- Create: `movie_translator/ocr/tests/test_vision_ocr.py`
- Modify: `pyproject.toml` (add `vision-ocr` optional deps)

- [ ] **Step 1: Add dependencies to pyproject.toml**

In `pyproject.toml`, replace the existing `ocr` optional group:

```toml
[project.optional-dependencies]
vision-ocr = [
    "pyobjc-framework-Vision>=11.0; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz>=11.0; sys_platform == 'darwin'",
    "numpy>=1.26",
    "Pillow>=10.0",
]
```

Then install: `uv sync --extra vision-ocr --group dev`

- [ ] **Step 2: Create test package and write tests**

Create `movie_translator/ocr/tests/__init__.py` (empty file).

Create `movie_translator/ocr/tests/test_vision_ocr.py`:

```python
import platform
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Darwin',
    reason='Apple Vision framework only available on macOS',
)


@pytest.fixture
def image_with_text(tmp_path):
    """Create a simple image with known text using FFmpeg."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()
    output = tmp_path / 'test_text.png'

    cmd = [
        ffmpeg,
        '-y',
        '-f', 'lavfi',
        '-i', 'color=black:s=640x120:d=1',
        '-vf', "drawtext=text='Hello world':fontsize=48:fontcolor=white:x=(w-tw)/2:y=(h-th)/2",
        '-frames:v', '1',
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'Could not create test image: {result.stderr}')

    return output


@pytest.fixture
def blank_image(tmp_path):
    """Create a blank black image with no text."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()
    output = tmp_path / 'blank.png'

    cmd = [
        ffmpeg,
        '-y',
        '-f', 'lavfi',
        '-i', 'color=black:s=640x120:d=1',
        '-frames:v', '1',
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output


def test_is_available():
    from movie_translator.ocr.vision_ocr import is_available

    assert is_available() is True


def test_recognize_text_finds_known_text(image_with_text):
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(image_with_text)

    assert 'hello' in result.lower() or 'Hello' in result


def test_recognize_text_returns_empty_for_blank(blank_image):
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(blank_image)

    assert result == ''


def test_recognize_text_returns_empty_for_nonexistent():
    from movie_translator.ocr.vision_ocr import recognize_text

    result = recognize_text(Path('/nonexistent/image.png'))

    assert result == ''
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest movie_translator/ocr/tests/test_vision_ocr.py -v`
Expected: FAIL (import error — `vision_ocr` module does not exist)

- [ ] **Step 4: Implement vision_ocr.py**

Create `movie_translator/ocr/vision_ocr.py`:

```python
import sys
from pathlib import Path

from ..logging import logger


def is_available() -> bool:
    if sys.platform != 'darwin':
        return False
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401

        return True
    except ImportError:
        return False


def recognize_text(image_path: Path, language: str = 'en') -> str:
    if not image_path.exists():
        return ''

    try:
        import Quartz
        import Vision

        # Load image via CoreGraphics
        url = Quartz.CFURLCreateWithFileSystemPath(
            None, str(image_path), Quartz.kCFURLPOSIXPathStyle, False
        )
        image_source = Quartz.CGImageSourceCreateWithURL(url, None)
        if image_source is None:
            logger.debug(f'Could not create image source for {image_path}')
            return ''

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            logger.debug(f'Could not load image from {image_path}')
            return ''

        # Create and configure text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)

        # Execute request
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        success = handler.performRequests_error_([request], None)
        if not success[0]:
            logger.debug(f'Vision request failed for {image_path}')
            return ''

        # Extract recognized text
        results = request.results()
        if not results:
            return ''

        lines = []
        for observation in results:
            candidate = observation.topCandidates_(1)
            if candidate:
                lines.append(candidate[0].string())

        return '\n'.join(lines)

    except Exception as e:
        logger.debug(f'OCR error for {image_path}: {e}')
        return ''
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest movie_translator/ocr/tests/test_vision_ocr.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/ocr/vision_ocr.py movie_translator/ocr/tests/__init__.py movie_translator/ocr/tests/test_vision_ocr.py pyproject.toml
git commit -m "feat: add Apple Vision OCR wrapper

Uses VNRecognizeTextRequest for on-device text recognition via Neural Engine.
macOS-only, with graceful fallback when unavailable."
```

---

### Task 4: Frame Extractor

Create `frame_extractor.py` to extract cropped subtitle-region frames via FFmpeg.

**Files:**
- Create: `movie_translator/ocr/frame_extractor.py`

- [ ] **Step 1: Write a manual smoke test**

Before writing the module, verify the FFmpeg crop command works on the example data:

Run: `mkdir -p /tmp/frame_test && uv run python -c "
from movie_translator.ffmpeg import get_ffmpeg
import subprocess
ffmpeg = get_ffmpeg()
cmd = [ffmpeg, '-y', '-i', 'movies/[106-114] Whisky Peak [En Sub][480p]/[One Pace][106-109] Whisky Peak 01 [480p][En Sub][BE7349ED].mp4', '-vf', 'crop=iw:ih/4:0:ih*3/4,fps=10', '-q:v', '2', '-frames:v', '5', '/tmp/frame_test/%06d.jpg']
r = subprocess.run(cmd, capture_output=True, text=True)
print('returncode:', r.returncode)
print('stderr:', r.stderr[-200:] if r.stderr else 'none')
import os
print('files:', sorted(os.listdir('/tmp/frame_test')))
"`

Expected: 5 JPEG files created successfully.

- [ ] **Step 2: Implement frame_extractor.py**

Create `movie_translator/ocr/frame_extractor.py`:

```python
import subprocess
from pathlib import Path

from ..ffmpeg import get_ffmpeg, get_video_info
from ..logging import logger


class FrameExtractionError(Exception):
    pass


def get_video_duration_ms(video_path: Path) -> int:
    info = get_video_info(video_path)
    duration = float(info.get('format', {}).get('duration', 0))
    return int(duration * 1000)


def extract_subtitle_region_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 10,
    crop_ratio: float = 0.25,
) -> list[tuple[Path, int]]:
    if not video_path.exists():
        raise FrameExtractionError(f'Video file not found: {video_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    # Crop bottom portion of frame and extract at target fps
    crop_y_start = f'ih*{1 - crop_ratio}'
    crop_height = f'ih*{crop_ratio}'
    vf = f'crop=iw:{crop_height}:0:{crop_y_start},fps={fps}'

    pattern = str(output_dir / '%06d.jpg')

    cmd = [
        ffmpeg,
        '-y',
        '-i', str(video_path),
        '-vf', vf,
        '-q:v', '2',
        pattern,
    ]

    logger.info(f'Extracting subtitle region frames at {fps}fps (bottom {int(crop_ratio * 100)}%)...')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_lines = [l for l in result.stderr.split('\n') if 'error' in l.lower()]
        error_msg = '; '.join(error_lines) if error_lines else 'Unknown FFmpeg error'
        raise FrameExtractionError(f'Frame extraction failed: {error_msg}')

    # Build frame list with timestamps
    frame_interval_ms = 1000 // fps
    frames: list[tuple[Path, int]] = []

    for frame_path in sorted(output_dir.glob('*.jpg')):
        # FFmpeg numbers frames starting at 1
        frame_num = int(frame_path.stem)
        timestamp_ms = (frame_num - 1) * frame_interval_ms
        frames.append((frame_path, timestamp_ms))

    logger.info(f'Extracted {len(frames)} frames')
    return frames
```

- [ ] **Step 3: Verify with a quick integration check**

Run: `uv run python -c "
from movie_translator.ocr.frame_extractor import extract_subtitle_region_frames
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    frames = extract_subtitle_region_frames(
        Path('movies/[106-114] Whisky Peak [En Sub][480p]/[One Pace][106-109] Whisky Peak 01 [480p][En Sub][BE7349ED].mp4'),
        Path(td),
        fps=1,  # 1fps for quick test
    )
    print(f'Extracted {len(frames)} frames')
    print(f'First: {frames[0]}')
    print(f'Last: {frames[-1]}')
"`

Expected: ~1859 frames (31 min video at 1fps), timestamps spanning 0 to ~1858000ms.

- [ ] **Step 4: Commit**

```bash
git add movie_translator/ocr/frame_extractor.py
git commit -m "feat: add FFmpeg frame extractor for subtitle region crops

Extracts bottom portion of video frames at configurable fps using FFmpeg crop filter."
```

---

### Task 5: Change Detector

Create `change_detector.py` for pixel-diff based subtitle transition detection.

**Files:**
- Create: `movie_translator/ocr/change_detector.py`
- Create: `movie_translator/ocr/tests/test_change_detector.py`

- [ ] **Step 1: Write tests**

Create `movie_translator/ocr/tests/test_change_detector.py`:

```python
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def create_test_frame(tmp_path):
    """Create test frames using FFmpeg drawtext."""
    from movie_translator.ffmpeg import get_ffmpeg

    ffmpeg = get_ffmpeg()

    def _create(filename, text=None, bg_color='black'):
        output = tmp_path / filename
        if text:
            vf = f"drawtext=text='{text}':fontsize=36:fontcolor=white:x=(w-tw)/2:y=(h-th)/2"
            cmd = [
                ffmpeg, '-y', '-f', 'lavfi',
                '-i', f'color={bg_color}:s=640x120:d=1',
                '-vf', vf, '-frames:v', '1', str(output),
            ]
        else:
            cmd = [
                ffmpeg, '-y', '-f', 'lavfi',
                '-i', f'color={bg_color}:s=640x120:d=1',
                '-frames:v', '1', str(output),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip(f'Could not create test frame: {result.stderr}')
        return output

    return _create


class TestDetectTransitions:
    def test_no_transitions_for_identical_frames(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        blank1 = create_test_frame('blank1.jpg')
        blank2 = create_test_frame('blank2.jpg')

        frames = [(blank1, 0), (blank2, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) == 0

    def test_detects_text_appearing(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        blank = create_test_frame('blank.jpg')
        with_text = create_test_frame('text.jpg', text='Hello world')

        frames = [(blank, 0), (with_text, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[0].event_type == 'appeared'
        assert transitions[0].timestamp_ms == 100

    def test_detects_text_disappearing(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        with_text = create_test_frame('text.jpg', text='Hello world')
        blank = create_test_frame('blank.jpg')

        frames = [(with_text, 0), (blank, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[-1].event_type == 'disappeared'

    def test_detects_text_change(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        text1 = create_test_frame('text1.jpg', text='Hello world')
        text2 = create_test_frame('text2.jpg', text='Goodbye world')

        frames = [(text1, 0), (text2, 100)]
        transitions = detect_transitions(frames)

        assert len(transitions) >= 1
        assert transitions[0].event_type == 'appeared'

    def test_empty_frames_list(self):
        from movie_translator.ocr.change_detector import detect_transitions

        transitions = detect_transitions([])
        assert transitions == []

    def test_single_frame(self, create_test_frame):
        from movie_translator.ocr.change_detector import detect_transitions

        frame = create_test_frame('single.jpg', text='Hello')
        transitions = detect_transitions([(frame, 0)])

        assert transitions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/ocr/tests/test_change_detector.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement change_detector.py**

Create `movie_translator/ocr/change_detector.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..logging import logger


@dataclass
class SubtitleTransition:
    timestamp_ms: int
    frame_path: Path
    event_type: Literal['appeared', 'disappeared']


def _load_grayscale(image_path: Path) -> np.ndarray:
    """Load JPEG as grayscale numpy array without OpenCV."""
    from PIL import Image

    img = Image.open(image_path).convert('L')
    return np.array(img)


def _frame_has_text(frame: np.ndarray, variance_threshold: float = 200.0) -> bool:
    """Heuristic: frames with text have higher pixel variance than blank frames."""
    return float(np.var(frame)) > variance_threshold


def detect_transitions(
    frames: list[tuple[Path, int]],
    change_threshold: float = 15.0,
) -> list[SubtitleTransition]:
    if len(frames) < 2:
        return []

    transitions: list[SubtitleTransition] = []
    prev_frame = _load_grayscale(frames[0][0])
    prev_has_text = _frame_has_text(prev_frame)

    for i in range(1, len(frames)):
        frame_path, timestamp_ms = frames[i]
        curr_frame = _load_grayscale(frame_path)

        # Compute mean absolute pixel difference
        diff = np.mean(np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)))

        if diff > change_threshold:
            curr_has_text = _frame_has_text(curr_frame)

            if curr_has_text and not prev_has_text:
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'appeared'))
            elif not curr_has_text and prev_has_text:
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'disappeared'))
            elif curr_has_text and prev_has_text:
                # Text changed — treat as new subtitle appeared
                transitions.append(SubtitleTransition(timestamp_ms, frame_path, 'appeared'))

            prev_has_text = curr_has_text

        prev_frame = curr_frame

    logger.info(f'Detected {len(transitions)} subtitle transitions')
    return transitions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/ocr/tests/test_change_detector.py -v`
Expected: ALL PASS

Note: The `Pillow` package is needed. It is a transitive dependency of `transformers` which is already installed, but if not available, install it: `uv add Pillow`.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/ocr/change_detector.py movie_translator/ocr/tests/test_change_detector.py
git commit -m "feat: add pixel-diff change detector for subtitle transitions

Compares consecutive cropped frames to detect when subtitles appear, change, or disappear."
```

---

### Task 6: Burned-In Subtitle Orchestrator

Create `burned_in_extractor.py` that ties frame extraction, change detection, and OCR together.

**Files:**
- Create: `movie_translator/ocr/burned_in_extractor.py`
- Create: `movie_translator/ocr/tests/test_burned_in_extractor.py`

- [ ] **Step 1: Write tests for the orchestrator logic**

Create `movie_translator/ocr/tests/test_burned_in_extractor.py`:

```python
from pathlib import Path
from unittest.mock import patch

from movie_translator.ocr.burned_in_extractor import _build_dialogue_lines, _write_srt
from movie_translator.ocr.change_detector import SubtitleTransition
from movie_translator.types import DialogueLine


class TestBuildDialogueLines:
    def test_simple_appeared_disappeared_pair(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'disappeared'),
        ]
        ocr_results = {Path('f1.jpg'): 'Hello world'}

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 3000, 'Hello world')

    def test_consecutive_appeared_events(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): 'First line',
            Path('f2.jpg'): 'Second line',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 2
        assert lines[0] == DialogueLine(1000, 3000, 'First line')
        assert lines[1] == DialogueLine(3000, 5000, 'Second line')

    def test_filters_empty_ocr_results(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): '',
            Path('f2.jpg'): 'Real text',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0].text == 'Real text'

    def test_filters_single_character_garbage(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(3000, Path('f2.jpg'), 'disappeared'),
        ]
        ocr_results = {Path('f1.jpg'): 'x'}

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 0

    def test_deduplicates_consecutive_identical_text(self):
        transitions = [
            SubtitleTransition(1000, Path('f1.jpg'), 'appeared'),
            SubtitleTransition(2000, Path('f2.jpg'), 'appeared'),
            SubtitleTransition(5000, Path('f3.jpg'), 'disappeared'),
        ]
        ocr_results = {
            Path('f1.jpg'): 'Same text',
            Path('f2.jpg'): 'Same text',
        }

        lines = _build_dialogue_lines(transitions, ocr_results)

        assert len(lines) == 1
        assert lines[0] == DialogueLine(1000, 5000, 'Same text')

    def test_empty_transitions(self):
        lines = _build_dialogue_lines([], {})

        assert lines == []


class TestWriteSrt:
    def test_writes_valid_srt(self, tmp_path):
        lines = [
            DialogueLine(1000, 3000, 'Hello world'),
            DialogueLine(4000, 6500, 'Second line'),
        ]
        output = tmp_path / 'output.srt'

        _write_srt(lines, output)

        content = output.read_text()
        assert '1\n' in content
        assert '00:00:01,000 --> 00:00:03,000' in content
        assert 'Hello world' in content
        assert '2\n' in content
        assert '00:00:04,000 --> 00:00:06,500' in content
        assert 'Second line' in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest movie_translator/ocr/tests/test_burned_in_extractor.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement burned_in_extractor.py**

Create `movie_translator/ocr/burned_in_extractor.py`:

```python
import shutil
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine
from .change_detector import SubtitleTransition, detect_transitions
from .frame_extractor import extract_subtitle_region_frames, get_video_duration_ms
from .vision_ocr import recognize_text


def _build_dialogue_lines(
    transitions: list[SubtitleTransition],
    ocr_results: dict[Path, str],
) -> list[DialogueLine]:
    lines: list[DialogueLine] = []

    appeared_events = [t for t in transitions if t.event_type == 'appeared']

    for i, event in enumerate(appeared_events):
        text = ocr_results.get(event.frame_path, '').strip()

        # Filter garbage
        if len(text) <= 1:
            continue

        # Determine end time: next transition (of any type) after this appeared event
        event_idx = transitions.index(event)
        if event_idx + 1 < len(transitions):
            end_ms = transitions[event_idx + 1].timestamp_ms
        else:
            # Last event — use a reasonable default (3 seconds after start)
            end_ms = event.timestamp_ms + 3000

        start_ms = event.timestamp_ms

        # Deduplicate: if previous line has same text, extend its end time
        if lines and lines[-1].text == text:
            lines[-1] = DialogueLine(lines[-1].start_ms, end_ms, text)
        else:
            lines.append(DialogueLine(start_ms, end_ms, text))

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
    fps: int = 10,
) -> Path | None:
    frames_dir = output_dir / '_ocr_frames'

    try:
        # Step 1: Extract cropped frames
        frames = extract_subtitle_region_frames(video_path, frames_dir, fps=fps, crop_ratio=crop_ratio)
        if not frames:
            logger.error('No frames extracted from video')
            return None

        # Step 2: Detect transitions
        transitions = detect_transitions(frames)
        if not transitions:
            logger.warning('No subtitle transitions detected — video may not have burned-in subtitles')
            return None

        appeared_count = sum(1 for t in transitions if t.event_type == 'appeared')
        video_duration_ms = get_video_duration_ms(video_path)
        video_duration_min = video_duration_ms / 60_000

        if appeared_count < 5 and video_duration_min > 10:
            logger.warning(
                f'Only {appeared_count} subtitle appearances in {video_duration_min:.0f}min video — '
                'this may not have burned-in subtitles'
            )

        # Step 3: OCR on transition frames
        appeared_frames = [t for t in transitions if t.event_type == 'appeared']
        logger.info(f'Running OCR on {len(appeared_frames)} transition frames...')

        ocr_results: dict[Path, str] = {}
        for t in appeared_frames:
            text = recognize_text(t.frame_path)
            ocr_results[t.frame_path] = text

        # Step 4: Build dialogue lines
        lines = _build_dialogue_lines(transitions, ocr_results)
        if not lines:
            logger.warning('OCR produced no usable subtitle lines')
            return None

        logger.info(f'Extracted {len(lines)} subtitle lines via OCR')

        # Step 5: Write SRT
        srt_path = output_dir / f'{video_path.stem}_ocr.srt'
        _write_srt(lines, srt_path)

        return srt_path

    finally:
        # Clean up frame images
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest movie_translator/ocr/tests/test_burned_in_extractor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/ocr/burned_in_extractor.py movie_translator/ocr/tests/test_burned_in_extractor.py
git commit -m "feat: add burned-in subtitle orchestrator

Connects frame extraction, change detection, and Vision OCR to produce
DialogueLine list and SRT output from burned-in video subtitles."
```

---

### Task 7: Pipeline Integration

Wire the burned-in OCR into the existing pipeline as a fallback, and update the OCR module exports.

**Files:**
- Modify: `movie_translator/ocr/__init__.py`
- Delete: `movie_translator/ocr/processor.py`
- Modify: `movie_translator/pipeline.py:106-132` (add fallback)
- Modify: `movie_translator/subtitles/extractor.py:227-252` (remove SubtitleOCR reference)

- [ ] **Step 1: Replace ocr/__init__.py and delete processor.py**

Replace `movie_translator/ocr/__init__.py` with:

```python
from .burned_in_extractor import extract_burned_in_subtitles
from .vision_ocr import is_available as is_vision_ocr_available

__all__ = ['extract_burned_in_subtitles', 'is_vision_ocr_available']
```

Delete `movie_translator/ocr/processor.py`.

- [ ] **Step 2: Clean up SubtitleOCR reference in extractor.py**

In `movie_translator/subtitles/extractor.py`, replace the `_handle_image_tracks` method (lines 227-252) to remove the deleted `SubtitleOCR` dependency:

```python
    def _handle_image_tracks(self, image_tracks: list[dict], total_count: int) -> dict | None:
        logger.warning(
            f'Found {total_count} English tracks, but only image-based dialogue tracks available'
        )
        logger.info('Image-based subtitle track OCR (PGS/DVD) is not supported')
        logger.info('If the video has burned-in subtitles, use --enable-ocr flag')
        return None
```

- [ ] **Step 3: Update pipeline.py with burned-in OCR fallback**

In `movie_translator/pipeline.py`, update the imports at the top:

```python
import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .video import VideoOperations
```

Replace the `_extract_subtitles` method (lines 106-132):

```python
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
```

Add the new helper methods (after `_get_video_ops`):

```python
    def _can_try_burned_in_ocr(self) -> bool:
        if not self.enable_ocr:
            return False
        if not is_vision_ocr_available():
            logger.warning('Apple Vision OCR not available on this platform')
            return False
        return True

    def _extract_burned_in_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('No subtitle tracks found — attempting burned-in subtitle OCR...')
        return extract_burned_in_subtitles(video_path, output_dir)
```

Remove the old `_get_ocr` method and the `_process_ocr_subtitles` method — they referenced the deleted `SubtitleOCR` class and are fully replaced by the new methods.

Remove the `self._ocr = None` line from `__init__` as well.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS. The existing integration tests should still work because they use MKV files with subtitle tracks — the burned-in fallback path is never triggered.

- [ ] **Step 5: Commit**

```bash
git add movie_translator/ocr/__init__.py movie_translator/pipeline.py movie_translator/subtitles/extractor.py
git rm movie_translator/ocr/processor.py
git commit -m "feat: integrate burned-in OCR as fallback in translation pipeline

When no English subtitle tracks are found and --enable-ocr is passed,
the pipeline falls back to extracting burned-in subtitles via Apple Vision OCR."
```

---

### Task 8: Multimodal Validation on Example Data

Run the full OCR pipeline on the example MP4 and validate using visual frame inspection.

**Files:** None (validation only)

- [ ] **Step 1: Extract sample frames for visual comparison**

Extract 10 frames at known timestamps from the example video:

Run: `mkdir -p /tmp/validation_frames && uv run python -c "
from movie_translator.ffmpeg import get_ffmpeg
import subprocess
ffmpeg = get_ffmpeg()
# Extract full frames (not cropped) at specific timestamps for visual verification
timestamps = ['00:01:00', '00:03:00', '00:05:00', '00:07:00', '00:10:00', '00:12:00', '00:15:00', '00:18:00', '00:20:00', '00:25:00']
for i, ts in enumerate(timestamps):
    cmd = [ffmpeg, '-y', '-ss', ts, '-i', 'movies/[106-114] Whisky Peak [En Sub][480p]/[One Pace][106-109] Whisky Peak 01 [480p][En Sub][BE7349ED].mp4', '-frames:v', '1', f'/tmp/validation_frames/frame_{i:02d}_{ts.replace(\":\",\"\")}.png']
    subprocess.run(cmd, capture_output=True, text=True)
print('Done. Files:', sorted(__import__('os').listdir('/tmp/validation_frames')))
"`

Then visually inspect the frames using the Read tool to see what text is burned in.

- [ ] **Step 2: Run the OCR pipeline on the example video**

Run: `uv run python -c "
from movie_translator.ocr.burned_in_extractor import extract_burned_in_subtitles
from pathlib import Path
import tempfile

video = Path('movies/[106-114] Whisky Peak [En Sub][480p]/[One Pace][106-109] Whisky Peak 01 [480p][En Sub][BE7349ED].mp4')

with tempfile.TemporaryDirectory() as td:
    srt = extract_burned_in_subtitles(video, Path(td), fps=2)
    if srt:
        print('SUCCESS - SRT file created')
        print(srt.read_text()[:3000])
    else:
        print('FAILED - No SRT produced')
"`

Note: Using `fps=2` for initial testing (faster). Increase to `fps=10` once validated.

- [ ] **Step 3: Compare OCR output against visual frame inspection**

Read the extracted frames visually and compare against the OCR-produced SRT. Document:
- Number of frames with visible subtitles
- How many were correctly OCR'd
- Any systematic errors (partial text, wrong characters, missed lines)

- [ ] **Step 4: Tune parameters if needed**

If accuracy is poor, adjust:
- `change_threshold` in `detect_transitions()` (default 15.0)
- `variance_threshold` in `_frame_has_text()` (default 200.0)
- `crop_ratio` (default 0.25 — try 0.20 or 0.33)
- `fps` (default 10 — lower for speed, higher for accuracy)

- [ ] **Step 5: Run the full pipeline end-to-end with --enable-ocr**

Run: `uv run movie-translator movies/ --enable-ocr --dry-run --verbose`

This exercises the complete flow: MP4 discovery -> no subtitle tracks -> OCR fallback -> translation -> mux.

- [ ] **Step 6: Run the full test suite one final time**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 7: Commit any parameter adjustments**

If any thresholds were tuned in steps 3-4:

```bash
git add -u
git commit -m "tune: adjust OCR detection thresholds based on real-world testing"
```
