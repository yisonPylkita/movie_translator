# Parallel Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process multiple video files concurrently using asyncio, with IO-bound stages running in parallel and GPU-bound work serialized through a dedicated worker queue.

**Architecture:** `asyncio.run()` replaces the sync loop. Pipeline workers are coroutines (one per file, throttled by semaphore). Existing sync stages run via `asyncio.to_thread()`. A single GPU worker coroutine consumes from an `asyncio.Queue`, executing compute tasks (translation, OCR, inpainting) one at a time with model caching.

**Tech Stack:** Python asyncio (stdlib), Rich (existing dep). No new dependencies.

---

## File Structure

```
movie_translator/
├── gpu_queue.py              # NEW: GpuQueue, GpuTask, TranslateTask, OcrTask, InpaintTask
├── async_pipeline.py         # NEW: process_file coroutine, run_all orchestrator
├── context.py                # MODIFY: add workers to PipelineConfig, add pending_ocr
├── logging.py                # MODIFY: add _current_file ContextVar
├── main.py                   # MODIFY: asyncio.run(), --workers flag
├── pipeline.py               # KEEP: backward compat for --workers 1
├── progress.py               # MODIFY: multi-file tracking
├── stages/
│   ├── extract_ref.py        # MODIFY: add run_io() method
│   ├── extract_english.py    # MODIFY: add run_io() method
│   ├── translate.py          # MODIFY: extract check_fonts() as public
│   └── mux.py                # UNCHANGED: already respects ctx.inpainted_video
├── tests/
│   ├── test_gpu_queue.py     # NEW: GPU queue unit tests
│   └── test_async_pipeline.py # NEW: async pipeline integration tests
```

---

### Task 1: GPU Queue Core

**Files:**
- Create: `movie_translator/gpu_queue.py`
- Create: `movie_translator/tests/test_gpu_queue.py`

- [ ] **Step 1: Write failing tests for GpuQueue**

Create `movie_translator/tests/test_gpu_queue.py`:

```python
"""Tests for the GPU work queue."""

import asyncio

import pytest

from movie_translator.gpu_queue import GpuQueue, GpuTask


class FakeTask(GpuTask):
    """A fake GPU task that returns a fixed value."""

    model_type = 'fake'

    def __init__(self, value, file_tag='test'):
        self.value = value
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        return self.value


class SlowTask(GpuTask):
    """A task that records execution order."""

    model_type = 'slow'

    def __init__(self, task_id, order_list, file_tag='test'):
        self.task_id = task_id
        self._order_list = order_list
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        self._order_list.append(self.task_id)
        return self.task_id


class FailingTask(GpuTask):
    """A task that raises an exception."""

    model_type = 'fail'

    def __init__(self, file_tag='test'):
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        raise ValueError('GPU task failed')


class ModelTrackingTask(GpuTask):
    """A task that records model cache state."""

    def __init__(self, model_type_val, records, file_tag='test'):
        self.model_type = model_type_val
        self._records = records
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        self._records.append({
            'type': self.model_type,
            'last_type': last_model_type,
            'cache_keys': list(model_cache.keys()),
        })
        model_cache[self.model_type] = f'model_{self.model_type}'
        return True


class TestGpuQueue:
    @pytest.mark.asyncio
    async def test_submit_returns_result(self):
        queue = GpuQueue()
        worker = asyncio.create_task(queue.run_worker())
        result = await queue.submit(FakeTask(42))
        assert result == 42
        await queue.shutdown()
        await worker

    @pytest.mark.asyncio
    async def test_fifo_order(self):
        order = []
        queue = GpuQueue()
        worker = asyncio.create_task(queue.run_worker())

        results = await asyncio.gather(
            queue.submit(SlowTask('a', order)),
            queue.submit(SlowTask('b', order)),
            queue.submit(SlowTask('c', order)),
        )
        assert results == ['a', 'b', 'c']
        assert order == ['a', 'b', 'c']

        await queue.shutdown()
        await worker

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        queue = GpuQueue()
        worker = asyncio.create_task(queue.run_worker())

        # Submit: good, bad, good
        r1 = asyncio.create_task(queue.submit(FakeTask(1)))
        r2 = asyncio.create_task(queue.submit(FailingTask()))
        r3 = asyncio.create_task(queue.submit(FakeTask(3)))

        assert await r1 == 1
        with pytest.raises(ValueError, match='GPU task failed'):
            await r2
        assert await r3 == 3

        await queue.shutdown()
        await worker

    @pytest.mark.asyncio
    async def test_model_cache_passed_between_tasks(self):
        records = []
        queue = GpuQueue()
        worker = asyncio.create_task(queue.run_worker())

        await queue.submit(ModelTrackingTask('translate', records))
        await queue.submit(ModelTrackingTask('translate', records))
        await queue.submit(ModelTrackingTask('ocr', records))

        # First translate: no prior model
        assert records[0]['last_type'] is None
        # Second translate: last was translate
        assert records[1]['last_type'] == 'translate'
        assert 'translate' in records[1]['cache_keys']
        # OCR: last was translate
        assert records[2]['last_type'] == 'translate'

        await queue.shutdown()
        await worker

    @pytest.mark.asyncio
    async def test_shutdown_stops_worker(self):
        queue = GpuQueue()
        worker = asyncio.create_task(queue.run_worker())
        await queue.shutdown()
        await asyncio.wait_for(worker, timeout=2.0)
        assert worker.done()

    @pytest.mark.asyncio
    async def test_queue_size(self):
        queue = GpuQueue()
        assert queue.pending == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest movie_translator/tests/test_gpu_queue.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'movie_translator.gpu_queue'`

- [ ] **Step 3: Install pytest-asyncio**

Run: `uv pip install pytest-asyncio`

Add to `pyproject.toml` dev dependencies:
```toml
"pytest-asyncio>=0.24",
```

Add asyncio_mode to pyproject.toml pytest section:
```toml
asyncio_mode = "auto"
```

- [ ] **Step 4: Implement GpuQueue**

Create `movie_translator/gpu_queue.py`:

```python
"""GPU work queue — serializes compute-heavy tasks through a single worker.

Pipeline worker coroutines submit tasks (translation, OCR, inpainting)
and await results via Futures. The single GPU worker coroutine processes
tasks FIFO, keeping the last model loaded to avoid thrashing.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class GpuTask(ABC):
    """Base class for GPU-bound work items."""

    model_type: str  # Used by worker for model caching decisions
    file_tag: str  # Short filename for log tagging

    @abstractmethod
    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> Any:
        """Run the task. Called in a thread by the GPU worker.

        Args:
            model_cache: Mutable dict persisted across tasks. Store loaded
                models here keyed by model_type.
            last_model_type: The model_type of the previous task, or None
                if this is the first task. Use to decide whether to reload.

        Returns:
            Task-specific result delivered to the awaiting coroutine.
        """


class GpuQueue:
    """Single-consumer async queue for GPU/NPU-bound tasks."""

    def __init__(self):
        self._queue: asyncio.Queue[tuple[GpuTask, asyncio.Future] | None] = asyncio.Queue()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def submit(self, task: GpuTask) -> Any:
        """Submit a task and await its result.

        The calling coroutine suspends until the GPU worker completes
        the task. If the task fails, the exception propagates here.
        """
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((task, future))
        return await future

    async def run_worker(self) -> None:
        """GPU worker loop. Pull tasks FIFO, execute, deliver results.

        Keeps the last model loaded to avoid unnecessary reloads.
        Shuts down when it receives a None sentinel via shutdown().
        """
        last_model_type: str | None = None
        model_cache: dict[str, Any] = {}

        while True:
            item = await self._queue.get()
            if item is None:
                break
            task, future = item
            try:
                # Set file context for log tagging from GPU thread
                from .logging import current_file_tag
                current_file_tag.set(task.file_tag)
                result = await asyncio.to_thread(
                    task.execute, model_cache, last_model_type
                )
                last_model_type = task.model_type
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    async def shutdown(self) -> None:
        """Signal the worker to stop after processing pending tasks."""
        await self._queue.put(None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest movie_translator/tests/test_gpu_queue.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Lint and commit**

Run: `ruff check movie_translator/gpu_queue.py movie_translator/tests/test_gpu_queue.py && ruff format movie_translator/gpu_queue.py movie_translator/tests/test_gpu_queue.py`

```bash
git add movie_translator/gpu_queue.py movie_translator/tests/test_gpu_queue.py pyproject.toml
git commit -m "feat(parallel): add GPU work queue with FIFO ordering and model cache"
```

---

### Task 2: GPU Task Implementations (TranslateTask, OcrTask, InpaintTask)

**Files:**
- Modify: `movie_translator/gpu_queue.py`
- Create: `movie_translator/tests/test_gpu_tasks.py`

- [ ] **Step 1: Write failing tests for task types**

Create `movie_translator/tests/test_gpu_tasks.py`:

```python
"""Tests for concrete GPU task types."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from movie_translator.gpu_queue import InpaintTask, OcrTask, TranslateTask
from movie_translator.types import DialogueLine, OCRResult, BoundingBox


class TestTranslateTask:
    def test_model_type(self):
        task = TranslateTask(
            dialogue_lines=[DialogueLine(0, 1000, 'Hello')],
            device='cpu',
            batch_size=16,
            model='allegro',
            file_tag='test',
        )
        assert task.model_type == 'translate'

    def test_execute_calls_translate(self):
        lines = [DialogueLine(0, 1000, 'Hello'), DialogueLine(1000, 2000, 'World')]
        translated = [DialogueLine(0, 1000, 'Cześć'), DialogueLine(1000, 2000, 'Świat')]
        task = TranslateTask(
            dialogue_lines=lines,
            device='cpu',
            batch_size=16,
            model='allegro',
            file_tag='ep01',
        )
        with patch(
            'movie_translator.gpu_queue.translate_dialogue_lines',
            return_value=translated,
        ) as mock_translate:
            result = task.execute({}, None)

        assert result == translated
        mock_translate.assert_called_once_with(lines, 'cpu', 16, 'allegro', progress_callback=None)

    def test_execute_with_progress_callback(self):
        lines = [DialogueLine(0, 1000, 'Hello')]
        task = TranslateTask(
            dialogue_lines=lines,
            device='cpu',
            batch_size=16,
            model='allegro',
            file_tag='ep01',
            progress_callback=lambda a, b, c: None,
        )
        with patch(
            'movie_translator.gpu_queue.translate_dialogue_lines',
            return_value=[DialogueLine(0, 1000, 'Cześć')],
        ):
            task.execute({}, None)


class TestOcrTask:
    def test_model_type(self):
        task = OcrTask(
            video_path=Path('/test.mkv'),
            track_id=2,
            output_dir=Path('/tmp'),
            ocr_type='pgs',
            file_tag='test',
        )
        assert task.model_type == 'ocr'


class TestInpaintTask:
    def test_model_type(self):
        task = InpaintTask(
            video_path=Path('/test.mkv'),
            output_path=Path('/tmp/out.mkv'),
            ocr_results=[],
            device='cpu',
            file_tag='test',
        )
        assert task.model_type == 'inpaint'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest movie_translator/tests/test_gpu_tasks.py -v`
Expected: FAIL with `ImportError: cannot import name 'TranslateTask' from 'movie_translator.gpu_queue'`

- [ ] **Step 3: Implement concrete task classes**

Append to `movie_translator/gpu_queue.py`:

```python
from .translation import translate_dialogue_lines
from .types import DialogueLine, OCRResult, ProgressCallback


class TranslateTask(GpuTask):
    """Translate all dialogue lines for one episode."""

    model_type = 'translate'

    def __init__(
        self,
        dialogue_lines: list[DialogueLine],
        device: str,
        batch_size: int,
        model: str,
        file_tag: str,
        progress_callback: ProgressCallback | None = None,
    ):
        self.dialogue_lines = dialogue_lines
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.file_tag = file_tag
        self.progress_callback = progress_callback

    def execute(self, model_cache, last_model_type):
        return translate_dialogue_lines(
            self.dialogue_lines,
            self.device,
            self.batch_size,
            self.model,
            progress_callback=self.progress_callback,
        )


class OcrTask(GpuTask):
    """OCR all PGS/burned-in subtitles for one episode."""

    model_type = 'ocr'

    def __init__(
        self,
        video_path: 'Path',
        track_id: int | None,
        output_dir: 'Path',
        ocr_type: str,  # 'pgs' or 'burned_in'
        file_tag: str,
    ):
        self.video_path = video_path
        self.track_id = track_id
        self.output_dir = output_dir
        self.ocr_type = ocr_type
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        from pathlib import Path

        if self.ocr_type == 'pgs':
            from .ocr.pgs_extractor import extract_pgs_track

            srt_path = extract_pgs_track(
                self.video_path, self.track_id, self.output_dir
            )
            return {'srt_path': srt_path, 'ocr_results': None}
        else:
            from .ocr import extract_burned_in_subtitles

            result = extract_burned_in_subtitles(self.video_path, self.output_dir)
            if result:
                return {'srt_path': result.srt_path, 'ocr_results': result.ocr_results}
            return {'srt_path': None, 'ocr_results': None}


class InpaintTask(GpuTask):
    """Inpaint all burned-in subtitle frames for one episode."""

    model_type = 'inpaint'

    def __init__(
        self,
        video_path: 'Path',
        output_path: 'Path',
        ocr_results: list[OCRResult],
        device: str,
        file_tag: str,
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.ocr_results = ocr_results
        self.device = device
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        from .inpainting import remove_burned_in_subtitles

        remove_burned_in_subtitles(
            self.video_path,
            self.output_path,
            self.ocr_results,
            self.device,
        )
        return self.output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest movie_translator/tests/test_gpu_tasks.py -v`
Expected: All tests PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check movie_translator/gpu_queue.py movie_translator/tests/test_gpu_tasks.py
ruff format movie_translator/gpu_queue.py movie_translator/tests/test_gpu_tasks.py
git add movie_translator/gpu_queue.py movie_translator/tests/test_gpu_tasks.py
git commit -m "feat(parallel): add TranslateTask, OcrTask, InpaintTask implementations"
```

---

### Task 3: Context and Config Changes

**Files:**
- Modify: `movie_translator/context.py`
- Modify: `movie_translator/main.py` (just the argparse addition)

- [ ] **Step 1: Add workers to PipelineConfig and pending_ocr to PipelineContext**

In `movie_translator/context.py`, add to `PipelineConfig`:
```python
    workers: int = 4
```

Add to `PipelineContext` (after `burned_in_probed`):
```python
    pending_ocr: dict | None = None  # Set by run_io() when OCR is deferred to GPU queue
```

- [ ] **Step 2: Add --workers CLI flag**

In `movie_translator/main.py`, add argument after `--keep-artifacts`:
```python
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Concurrent pipeline workers (default: auto, min(files, 4))',
    )
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `pytest movie_translator/stages/tests/ movie_translator/tests/ -v --tb=short -q`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add movie_translator/context.py movie_translator/main.py
git commit -m "feat(parallel): add workers config and pending_ocr to pipeline context"
```

---

### Task 4: Logging — File Context ContextVar

**Files:**
- Modify: `movie_translator/logging.py`
- Modify: `movie_translator/progress.py` (update _LogCapture to use it)

- [ ] **Step 1: Add _current_file ContextVar to logging.py**

In `movie_translator/logging.py`, add after imports:

```python
from contextvars import ContextVar

current_file_tag: ContextVar[str] = ContextVar('current_file_tag', default='')
```

- [ ] **Step 2: Update _LogCapture in progress.py to prepend file tag**

In `movie_translator/progress.py`, update the `_LogCapture.emit` method:

```python
from ..logging import current_file_tag  # add to imports

class _LogCapture(logging.Handler):
    def __init__(self, tracker: ProgressTracker):
        super().__init__()
        self._tracker = tracker

    def emit(self, record):
        try:
            msg = record.getMessage()
            tag = current_file_tag.get('')
            if tag:
                msg = f'[dim]\\[{tag}][/dim] {msg}'
            self._tracker._add_log(msg, record.levelname)
        except Exception:
            pass
```

- [ ] **Step 3: Run existing tests**

Run: `pytest movie_translator/tests/ -v --tb=short -q`
Expected: All PASS (no behavior change yet — ContextVar defaults to empty string)

- [ ] **Step 4: Commit**

```bash
git add movie_translator/logging.py movie_translator/progress.py
git commit -m "feat(parallel): add current_file_tag ContextVar for log message tagging"
```

---

### Task 5: Stage Refactoring — extract_ref.run_io()

**Files:**
- Modify: `movie_translator/stages/extract_ref.py`
- Modify: `movie_translator/stages/tests/test_extract_ref.py`

- [ ] **Step 1: Write test for run_io deferring OCR**

Add to `movie_translator/stages/tests/test_extract_ref.py`:

```python
class TestExtractRefRunIo:
    def test_text_track_extracts_directly(self, tmp_path):
        """run_io() with a text-based track should extract without setting pending_ocr."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

        text_track = {
            'id': 2,
            'codec': 'subrip',
            'subtitle_index': 0,
            'properties': {'language': 'eng'},
        }

        with (
            patch.object(SubtitleExtractor, 'get_track_info', return_value=[text_track]),
            patch.object(SubtitleExtractor, 'find_english_track', return_value=text_track),
            patch.object(SubtitleExtractor, 'get_subtitle_extension', return_value='.srt'),
            patch.object(SubtitleExtractor, 'extract_subtitle'),
        ):
            result = ExtractReferenceStage().run_io(ctx)

        assert result.pending_ocr is None
        assert result.reference_path is not None

    def test_pgs_track_defers_ocr(self, tmp_path):
        """run_io() with a PGS track should set pending_ocr instead of doing OCR."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

        pgs_track = {
            'id': 3,
            'codec': 'hdmv_pgs_subtitle',
            'subtitle_index': 0,
            'properties': {'language': 'eng'},
        }

        with (
            patch.object(SubtitleExtractor, 'get_track_info', return_value=[pgs_track]),
            patch.object(SubtitleExtractor, 'find_english_track', return_value=pgs_track),
        ):
            result = ExtractReferenceStage().run_io(ctx)

        assert result.pending_ocr is not None
        assert result.pending_ocr['type'] == 'pgs'
        assert result.pending_ocr['track_id'] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest movie_translator/stages/tests/test_extract_ref.py::TestExtractRefRunIo -v`
Expected: FAIL with `AttributeError: 'ExtractReferenceStage' object has no attribute 'run_io'`

- [ ] **Step 3: Implement run_io()**

In `movie_translator/stages/extract_ref.py`, add the `run_io` method to `ExtractReferenceStage`:

```python
    def run_io(self, ctx: PipelineContext) -> PipelineContext:
        """IO-only extraction. Defers OCR to pending_ocr for GPU queue."""
        extractor = SubtitleExtractor()
        ref_dir = ctx.work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        with ctx.metrics.span('get_track_info'):
            track_info = extractor.get_track_info(ctx.video_path)
            eng_track = extractor.find_english_track(track_info) if track_info else None

        if eng_track:
            ctx.original_english_track = OriginalTrack(
                stream_index=eng_track['id'],
                subtitle_index=eng_track.get('subtitle_index', 0),
                codec=eng_track.get('codec', 'unknown'),
                language=eng_track.get('properties', {}).get('language', 'eng'),
            )

            if _is_image_codec(eng_track):
                # Defer OCR to GPU queue
                ctx.pending_ocr = {
                    'type': 'pgs',
                    'track_id': eng_track['id'],
                    'output_dir': str(ref_dir),
                }
            else:
                with ctx.metrics.span('extract_subtitle'):
                    subtitle_ext = extractor.get_subtitle_extension(eng_track)
                    ref_path = ref_dir / f'{ctx.video_path.stem}_reference{subtitle_ext}'
                    try:
                        extractor.extract_subtitle(
                            ctx.video_path,
                            eng_track['id'],
                            ref_path,
                            eng_track.get('subtitle_index', 0),
                        )
                        ctx.reference_path = ref_path
                        logger.info(f'Extracted reference: {ref_path.name}')
                    except Exception as e:
                        logger.warning(f'Failed to extract reference: {e}')

        # If no track found and Vision is available, defer burned-in probe
        if ctx.reference_path is None and ctx.pending_ocr is None and is_vision_ocr_available():
            ctx.pending_ocr = {
                'type': 'burned_in',
                'track_id': None,
                'output_dir': str(ref_dir),
            }

        return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest movie_translator/stages/tests/test_extract_ref.py -v`
Expected: All PASS (both old tests via `run()` and new tests via `run_io()`)

- [ ] **Step 5: Commit**

```bash
git add movie_translator/stages/extract_ref.py movie_translator/stages/tests/test_extract_ref.py
git commit -m "feat(parallel): add run_io() to ExtractReferenceStage for deferred OCR"
```

---

### Task 6: Stage Refactoring — extract_english.run_io()

**Files:**
- Modify: `movie_translator/stages/extract_english.py`
- Modify: `movie_translator/stages/tests/test_extract_english.py`

- [ ] **Step 1: Write test for run_io**

Add to `movie_translator/stages/tests/test_extract_english.py`:

```python
class TestExtractEnglishRunIo:
    def test_uses_fetched_source_without_ocr(self, tmp_path):
        """run_io() should use fetched source and not set pending_ocr."""
        video = tmp_path / 'ep01.mkv'
        video.touch()
        eng_sub = tmp_path / 'eng.srt'
        eng_sub.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )
        ctx.fetched_subtitles = {
            'eng': [FetchedSubtitle(path=eng_sub, source='test')]
        }

        with patch.object(
            SubtitleProcessor, 'extract_dialogue_lines',
            return_value=[DialogueLine(1000, 2000, 'Hello')],
        ):
            result = ExtractEnglishStage().run_io(ctx)

        assert result.english_source == eng_sub
        assert result.pending_ocr is None
        assert len(result.dialogue_lines) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest movie_translator/stages/tests/test_extract_english.py::TestExtractEnglishRunIo -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement run_io()**

In `movie_translator/stages/extract_english.py`, add:

```python
    def run_io(self, ctx: PipelineContext) -> PipelineContext:
        """IO-only extraction. Defers OCR to pending_ocr for GPU queue."""
        # Priority: fetched English > reference > embedded text > defer OCR
        with ctx.metrics.span('select_source') as s:
            fetched_eng = None
            if ctx.fetched_subtitles:
                eng_subs = ctx.fetched_subtitles.get('eng')
                if eng_subs:
                    fetched_eng = eng_subs[0].path

            if fetched_eng:
                ctx.english_source = fetched_eng
                s.detail('source', 'fetched')
            elif ctx.reference_path:
                ctx.english_source = ctx.reference_path
                s.detail('source', 'reference')
            else:
                # Try embedded text track (IO), defer OCR
                ctx.english_source = self._extract_text_only(ctx)
                if ctx.english_source is not None:
                    s.detail('source', 'embedded')
                elif not ctx.burned_in_probed and is_vision_ocr_available():
                    ctx.pending_ocr = {
                        'type': 'burned_in',
                        'track_id': None,
                        'output_dir': str(ctx.work_dir),
                    }
                    s.detail('source', 'deferred_ocr')
                    return ctx  # Can't extract lines yet

        if ctx.english_source is None and ctx.pending_ocr is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        if ctx.english_source is not None:
            with ctx.metrics.span('extract_dialogue_lines') as s:
                ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
                if not ctx.dialogue_lines:
                    raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')
                s.detail('lines', len(ctx.dialogue_lines))

            logger.info(f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)')

        return ctx

    def _extract_text_only(self, ctx):
        """Try to extract a text-based (non-OCR) English track."""
        extractor = SubtitleExtractor()
        track_info = extractor.get_track_info(ctx.video_path)
        if not track_info:
            return None

        eng_track = extractor.find_english_track(track_info)
        if eng_track:
            codec = eng_track.get('codec', '').lower()
            is_image = any(codec == c or codec.startswith(c) for c in _IMAGE_CODECS)
            if not is_image:
                with ctx.metrics.span('extract_subtitle'):
                    subtitle_ext = extractor.get_subtitle_extension(eng_track)
                    output = ctx.work_dir / f'{ctx.video_path.stem}_extracted{subtitle_ext}'
                    subtitle_index = eng_track.get('subtitle_index', 0)
                    extractor.extract_subtitle(
                        ctx.video_path, eng_track['id'], output, subtitle_index
                    )
                    return output
        return None
```

- [ ] **Step 4: Run tests**

Run: `pytest movie_translator/stages/tests/test_extract_english.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/stages/extract_english.py movie_translator/stages/tests/test_extract_english.py
git commit -m "feat(parallel): add run_io() to ExtractEnglishStage for deferred OCR"
```

---

### Task 7: Stage Refactoring — Translate check_fonts() extraction

**Files:**
- Modify: `movie_translator/stages/translate.py`

- [ ] **Step 1: Extract check_fonts as a public method**

Refactor `TranslateStage` to make `check_fonts` a standalone method that can be called independently from the pipeline worker coroutine:

```python
class TranslateStage:
    name = 'translate'

    def __init__(self):
        self._tracker: ProgressTracker | None = None

    def set_tracker(self, tracker: ProgressTracker):
        self._tracker = tracker

    def check_fonts(self, ctx: PipelineContext) -> FontInfo:
        """Check font support for Polish characters. IO-bound, safe to run in worker."""
        with ctx.metrics.span('check_fonts') as s:
            supports = check_embedded_fonts_support_polish(ctx.video_path, ctx.english_source)
            if supports:
                s.detail('supports_polish', True)
                return FontInfo(supports_polish=True)
            is_mkv = ctx.video_path.suffix.lower() == '.mkv'
            if is_mkv:
                names = get_ass_font_names(ctx.english_source)
                result = find_system_font_for_polish(names)
                if result:
                    fp, fam = result
                    fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                    s.detail('supports_polish', False)
                    s.detail('fallback_font', fam)
                    return FontInfo(
                        supports_polish=False,
                        font_attachments=[fp],
                        fallback_font_family=fallback,
                    )
            s.detail('supports_polish', False)
            return FontInfo(supports_polish=False)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Full stage — runs font check + translation in parallel. Backward compat."""
        assert ctx.dialogue_lines is not None
        assert ctx.english_source is not None

        total = len(ctx.dialogue_lines)
        logger.info(f'Translating {total} lines...')

        dialogue_lines = ctx.dialogue_lines
        tracker = self._tracker
        metrics = ctx.metrics

        if ctx.config.model == 'apple':
            from ..translation.apple_backend import _get_apple_backend
            _backend = _get_apple_backend(ctx.config.batch_size)
            if _backend is None:
                raise RuntimeError('Failed to load translation model')
        else:
            with metrics.span('load_model') as s:
                _translator, cached = _get_translator(
                    ctx.config.device, ctx.config.batch_size, ctx.config.model
                )
                s.detail('cached', cached)
            if _translator is None:
                raise RuntimeError('Failed to load translation model')

        def _translate():
            with metrics.span('batch') as s:
                input_texts = [line.text for line in dialogue_lines]
                s.detail('input_lines', len(input_texts))
                s.detail('input_chars', sum(len(t) for t in input_texts))
                s.detail('batch_size', ctx.config.batch_size)

                def _on_progress(lines_done: int, total_lines: int, rate: float) -> None:
                    if tracker:
                        tracker.set_stage_progress(lines_done, total_lines, rate)

                translated = translate_dialogue_lines(
                    dialogue_lines,
                    ctx.config.device,
                    ctx.config.batch_size,
                    ctx.config.model,
                    progress_callback=_on_progress,
                )
                if translated:
                    s.detail('output_lines', len(translated))
                    s.detail('output_chars', sum(len(line.text) for line in translated))
                return translated

        with ThreadPoolExecutor(max_workers=2) as pool:
            ctx_fonts = copy_context()
            ctx_translate = copy_context()
            font_future = pool.submit(ctx_fonts.run, self.check_fonts, ctx)
            translate_future = pool.submit(ctx_translate.run, _translate)

            font_info: FontInfo = font_future.result()  # ty: ignore[invalid-assignment]
            ctx.font_info = font_info
            translated: list = translate_future.result()  # ty: ignore[invalid-assignment]

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
```

- [ ] **Step 2: Run existing translate tests**

Run: `pytest movie_translator/stages/tests/test_translate.py -v`
Expected: All PASS (run() behavior unchanged)

- [ ] **Step 3: Commit**

```bash
git add movie_translator/stages/translate.py
git commit -m "refactor(parallel): extract check_fonts as public method on TranslateStage"
```

---

### Task 8: Progress Tracker — Multi-file Support

**Files:**
- Modify: `movie_translator/progress.py`
- Create: `movie_translator/tests/test_progress.py`

- [ ] **Step 1: Write tests for multi-file progress tracker**

Create `movie_translator/tests/test_progress.py`:

```python
"""Tests for multi-file progress tracker."""

from unittest.mock import MagicMock

from movie_translator.progress import FileState, ProgressTracker, STAGES


class TestFileState:
    def test_initial_state(self):
        state = FileState(name='ep01', start_time=0.0)
        assert state.current_stage == ''
        assert state.stages_done == []
        assert state.gpu_status == 'none'
        assert state.stage_progress is None


class TestProgressTrackerMultiFile:
    def test_start_file_adds_to_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        assert 'ep01' in tracker._active_files

    def test_complete_file_removes_from_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.complete_file('ep01', 'success')
        assert 'ep01' not in tracker._active_files
        assert tracker._completed == 1

    def test_multiple_files_active(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.start_file('ep02')
        tracker.start_file('ep03')
        assert len(tracker._active_files) == 3

    def test_set_stage_updates_file(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('ep01', 'identify')
        tracker.set_stage('ep01', 'fetch')
        state = tracker._active_files['ep01']
        assert state.current_stage == 'fetch'
        assert 'identify' in state.stages_done

    def test_set_gpu_status(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_gpu_status('ep01', 'queued')
        assert tracker._active_files['ep01'].gpu_status == 'queued'
        tracker.set_gpu_status('ep01', 'running')
        assert tracker._active_files['ep01'].gpu_status == 'running'

    def test_set_stage_progress(self):
        tracker = ProgressTracker(total_files=5)
        tracker.start_file('ep01')
        tracker.set_stage('ep01', 'translate')
        tracker.set_stage_progress('ep01', 50, 100, 3.5)
        assert tracker._active_files['ep01'].stage_progress == (50, 100, 3.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest movie_translator/tests/test_progress.py -v`
Expected: FAIL — current ProgressTracker doesn't accept file_name params

- [ ] **Step 3: Rewrite progress.py for multi-file support**

Replace `movie_translator/progress.py` with the multi-file version. Key changes:

1. Add `FileState` dataclass
2. Change `_active_files: dict[str, FileState]` with `threading.Lock`
3. Update all public methods to accept `file_name` parameter
4. Keep backward-compat: methods without file_name use `_current_file` (for `--workers 1`)
5. Redesign `_render_current_file` → `_render_active_files` (multi-row)

The full implementation:

```python
"""Rich TUI progress display for batch video processing.

Supports multiple concurrent files with per-file stage tracking,
GPU queue status indicators, and log message tagging.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .logging import current_file_tag

STAGES = ['identify', 'fetch', 'extract', 'translate', 'create', 'mux']

STAGE_LABELS = {
    'identify': 'Identify',
    'fetch': 'Fetch',
    'extract': 'Extract',
    'translate': 'Translate',
    'create': 'Subtitles',
    'mux': 'Mux Video',
}

LEVEL_STYLES = {
    'DEBUG': 'dim white',
    'INFO': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'bold red',
}


@dataclass
class FileState:
    """Tracks progress of a single active file."""

    name: str
    start_time: float
    current_stage: str = ''
    stages_done: list[str] = field(default_factory=list)
    gpu_status: str = 'none'  # 'none' | 'queued' | 'running'
    stage_progress: tuple[int, int, float] | None = None  # (done, total, rate)


class _LogCapture(logging.Handler):
    """Captures log records and feeds them to the progress tracker."""

    def __init__(self, tracker: ProgressTracker):
        super().__init__()
        self._tracker = tracker

    def emit(self, record):
        try:
            msg = record.getMessage()
            tag = current_file_tag.get('')
            if tag:
                msg = f'[dim]\\[{tag}][/dim] {msg}'
            self._tracker._add_log(msg, record.levelname)
        except Exception:
            pass


class ProgressTracker:
    """Live TUI progress display for batch processing."""

    def __init__(self, total_files: int, console: Console | None = None):
        self._console = console or Console()
        self._total = total_files
        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._active_files: dict[str, FileState] = {}
        self._lock = threading.Lock()
        self._log_lines: deque[tuple[str, str]] = deque(maxlen=24)
        self._batch_start_time = 0.0
        self._live: Live | None = None
        self._log_handler: _LogCapture | None = None
        # Backward compat: track "current file" for single-worker mode
        self._current_file: str = ''

    def __enter__(self):
        self._batch_start_time = time.monotonic()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()

        self._log_handler = _LogCapture(self)
        logger = logging.getLogger('movie_translator')
        self._original_handlers = logger.handlers[:]
        self._original_propagate = logger.propagate
        logger.handlers = [self._log_handler]
        logger.propagate = False

        return self

    def __exit__(self, *args):
        logger = logging.getLogger('movie_translator')
        logger.handlers = self._original_handlers
        logger.propagate = self._original_propagate

        if self._live:
            self._live.__exit__(*args)
            self._live = None

        self._print_summary()

    def start_file(self, name: str):
        with self._lock:
            self._active_files[name] = FileState(
                name=name, start_time=time.monotonic()
            )
            self._current_file = name
        self._update()

    def set_stage(self, name_or_stage: str, stage: str = ''):
        """Set stage for a file. Supports both old (stage) and new (name, stage) API."""
        with self._lock:
            if stage:
                file_name = name_or_stage
            else:
                # Backward compat: single arg = stage name for current file
                stage = name_or_stage
                file_name = self._current_file

            state = self._active_files.get(file_name)
            if state:
                if state.current_stage and state.current_stage not in state.stages_done:
                    state.stages_done.append(state.current_stage)
                state.current_stage = stage
                state.stage_progress = None
                state.gpu_status = 'none'
        self._update()

    def set_gpu_status(self, file_name: str, status: str):
        """Update GPU queue status for a file: 'none', 'queued', 'running'."""
        with self._lock:
            state = self._active_files.get(file_name)
            if state:
                state.gpu_status = status
        self._update()

    def set_stage_progress(self, name_or_done: str | int, done_or_total: int = 0,
                           total_or_rate: int | float = 0, rate: float = 0.0):
        """Update sub-progress. Supports old (done, total, rate) and new (name, done, total, rate)."""
        with self._lock:
            if isinstance(name_or_done, str):
                file_name = name_or_done
                done = done_or_total
                total = int(total_or_rate)
                r = rate
            else:
                # Backward compat
                file_name = self._current_file
                done = name_or_done
                total = done_or_total
                r = float(total_or_rate)

            state = self._active_files.get(file_name)
            if state:
                state.stage_progress = (done, total, r)
        self._update()

    def complete_file(self, name_or_status: str, status: str = ''):
        """Mark file as done. Supports old (status) and new (name, status) API."""
        with self._lock:
            if status:
                file_name = name_or_status
            else:
                status = name_or_status
                file_name = self._current_file

            state = self._active_files.pop(file_name, None)
            elapsed = time.monotonic() - (state.start_time if state else self._batch_start_time)

            if status == 'success':
                self._completed += 1
                self._add_log(
                    f'[green]✓[/green] {self._short_name(file_name)} '
                    f'[dim]({elapsed:.0f}s)[/dim]',
                    'DONE',
                )
            elif status == 'failed':
                self._failed += 1
                self._add_log(
                    f'[red]✗[/red] {self._short_name(file_name)} '
                    f'[dim]({elapsed:.0f}s)[/dim]',
                    'DONE',
                )
            elif status == 'skipped':
                self._skipped += 1
                self._add_log(
                    f'[yellow]→[/yellow] {self._short_name(file_name)} '
                    f'[dim](skipped)[/dim]',
                    'DONE',
                )
        self._update()

    def _add_log(self, message: str, level: str):
        clean = message.strip()
        if not clean:
            return
        self._log_lines.append((clean, level))
        self._update()

    def _update(self):
        if self._live:
            self._live.update(self._render())

    def _render(self):
        parts = []
        parts.append(self._render_overall())
        with self._lock:
            active = dict(self._active_files)
        if active:
            parts.append(self._render_active_files(active))
        if self._log_lines:
            parts.append(self._render_logs())
        return Group(*parts)

    def _render_overall(self):
        done = self._completed + self._failed + self._skipped
        elapsed = time.monotonic() - self._batch_start_time

        table = Table.grid(padding=(0, 2))
        table.add_column(ratio=1)
        table.add_column(justify='right')

        progress = Progress(
            TextColumn('[bold blue]Movie Translator'),
            BarColumn(),
            MofNCompleteColumn(),
            expand=True,
        )
        progress.add_task('', total=self._total, completed=done)

        status_parts = []
        if self._completed:
            status_parts.append(f'[green]{self._completed} done[/green]')
        if self._failed:
            status_parts.append(f'[red]{self._failed} failed[/red]')
        if self._skipped:
            status_parts.append(f'[yellow]{self._skipped} skipped[/yellow]')

        status = '  '.join(status_parts) if status_parts else '[dim]starting...[/dim]'
        elapsed_str = f'[dim]{elapsed:.0f}s[/dim]'

        table.add_row(progress, Text.from_markup(f'{status}  {elapsed_str}'))
        return Panel(table, style='blue', padding=(0, 1))

    def _render_active_files(self, active: dict[str, FileState]):
        text = Text()
        for state in active.values():
            name = self._short_name(state.name, max_len=18)
            elapsed = time.monotonic() - state.start_time

            # Stage indicators
            stage_parts = []
            for stage in STAGES:
                if stage in state.stages_done:
                    stage_parts.append('[green]✓[/green]')
                elif stage == state.current_stage:
                    label = STAGE_LABELS.get(stage, stage)
                    if state.gpu_status == 'running':
                        stage_parts.append(f'[bold yellow]▸ {label} [cyan]\\[gpu][/cyan][/bold yellow]')
                    elif state.gpu_status == 'queued':
                        stage_parts.append(f'[dim yellow]⏳ {label} [dim]\\[queue][/dim][/dim yellow]')
                    else:
                        stage_parts.append(f'[bold yellow]▸ {label}[/bold yellow]')
                else:
                    stage_parts.append('[dim]○[/dim]')

            line = f'  {name:<18s} {" ".join(stage_parts)}'

            # Sub-progress for current stage
            if state.stage_progress:
                done, total, rate = state.stage_progress
                pct = done * 100 // total if total else 0
                line += f'  [dim]{done}/{total} {rate:.1f}/s[/dim]'

            line += f'  [dim]{elapsed:.0f}s[/dim]\n'
            try:
                text.append_text(Text.from_markup(line))
            except Exception:
                text.append(line)

        return Panel(text, title='[cyan]Active', padding=(0, 1))

    def _render_logs(self):
        text = Text()
        max_width = (self._console.width or 120) - 6
        for msg, level in self._log_lines:
            display_msg = msg[:max_width] + '...' if len(msg) > max_width else msg
            base_style = LEVEL_STYLES.get(level, 'white')
            if level == 'DONE':
                base_style = 'white'
            try:
                line = Text.from_markup(display_msg + '\n', style=base_style)
            except Exception:
                line = Text(display_msg + '\n', style=base_style)
            text.append_text(line)
        return Panel(text, title='[dim]Log', border_style='dim', padding=(0, 1))

    def _print_summary(self):
        self._console.print()
        parts = []
        if self._completed:
            parts.append(f'[green]✓ {self._completed} translated[/green]')
        if self._failed:
            parts.append(f'[red]✗ {self._failed} failed[/red]')
        if self._skipped:
            parts.append(f'[yellow]→ {self._skipped} skipped[/yellow]')

        elapsed = time.monotonic() - self._batch_start_time
        self._console.print(' | '.join(parts) + f'  [dim]({elapsed:.0f}s total)[/dim]')

    def _short_name(self, name: str, max_len: int = 0) -> str:
        if not max_len:
            max_len = (self._console.width or 80) - 20
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + '...'
```

- [ ] **Step 4: Run tests**

Run: `pytest movie_translator/tests/test_progress.py -v`
Expected: All PASS

- [ ] **Step 5: Run all existing tests to verify backward compat**

Run: `pytest --tb=short -q`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/progress.py movie_translator/tests/test_progress.py
git commit -m "feat(parallel): redesign ProgressTracker for multi-file concurrent display"
```

---

### Task 9: Async Pipeline Orchestration

**Files:**
- Create: `movie_translator/async_pipeline.py`
- Create: `movie_translator/tests/test_async_pipeline.py`

- [ ] **Step 1: Write integration test with mock stages**

Create `movie_translator/tests/test_async_pipeline.py`:

```python
"""Integration tests for async pipeline orchestration."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from movie_translator.async_pipeline import process_file, run_all
from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.gpu_queue import GpuQueue, GpuTask
from movie_translator.progress import ProgressTracker
from movie_translator.types import DialogueLine


class FakeTranslateTask(GpuTask):
    model_type = 'translate'

    def __init__(self, lines, file_tag='test'):
        self.dialogue_lines = lines
        self.file_tag = file_tag

    def execute(self, model_cache, last_model_type):
        return [DialogueLine(l.start_ms, l.end_ms, f'PL:{l.text}') for l in self.dialogue_lines]


class TestRunAll:
    @pytest.mark.asyncio
    async def test_processes_multiple_files(self, tmp_path):
        """Verify multiple files are processed and results collected."""
        files = []
        for i in range(3):
            f = tmp_path / f'ep{i:02d}.mkv'
            f.touch()
            files.append(f)

        config = PipelineConfig(workers=2)
        results = []

        async def fake_process(video_path, work_dir, config, stages, gpu_queue, tracker):
            results.append(video_path.name)

        with patch('movie_translator.async_pipeline.process_file', side_effect=fake_process):
            tracker = ProgressTracker(total_files=3)
            gpu_queue = GpuQueue()
            worker = asyncio.create_task(gpu_queue.run_worker())
            await run_all(files, tmp_path, config, None, gpu_queue, tracker)
            await gpu_queue.shutdown()
            await worker

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_worker_limit_respected(self, tmp_path):
        """With workers=1, files should process sequentially."""
        files = []
        for i in range(3):
            f = tmp_path / f'ep{i:02d}.mkv'
            f.touch()
            files.append(f)

        config = PipelineConfig(workers=1)
        concurrent_count = []
        active = 0

        async def fake_process(video_path, work_dir, config, stages, gpu_queue, tracker):
            nonlocal active
            active += 1
            concurrent_count.append(active)
            await asyncio.sleep(0.05)
            active -= 1

        with patch('movie_translator.async_pipeline.process_file', side_effect=fake_process):
            tracker = ProgressTracker(total_files=3)
            gpu_queue = GpuQueue()
            worker = asyncio.create_task(gpu_queue.run_worker())
            await run_all(files, tmp_path, config, None, gpu_queue, tracker)
            await gpu_queue.shutdown()
            await worker

        assert max(concurrent_count) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest movie_translator/tests/test_async_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement async_pipeline.py**

Create `movie_translator/async_pipeline.py`:

```python
"""Async pipeline orchestration — runs files concurrently with GPU queue.

Pipeline workers are coroutines (one per file). IO-bound stages run via
asyncio.to_thread(). GPU-bound work is submitted to a shared GpuQueue
and awaited.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .context import FontInfo, PipelineConfig, PipelineContext
from .discovery import create_work_dir
from .gpu_queue import GpuQueue, InpaintTask, OcrTask, TranslateTask
from .logging import current_file_tag, logger
from .metrics.collector import MetricsCollector, NullCollector
from .progress import ProgressTracker
from .stages import (
    CreateTracksStage,
    ExtractEnglishStage,
    ExtractReferenceStage,
    FetchSubtitlesStage,
    IdentifyStage,
    MuxStage,
    TranslateStage,
)


async def process_file(
    video_path: Path,
    work_dir: Path,
    config: PipelineConfig,
    stages: dict,
    gpu_queue: GpuQueue,
    tracker: ProgressTracker,
    metrics: MetricsCollector | NullCollector | None = None,
) -> bool:
    """Process a single video file through the pipeline.

    IO stages run in worker threads via to_thread.
    GPU stages submit to the gpu_queue and await results.
    """
    ctx = PipelineContext(
        video_path=video_path,
        work_dir=work_dir,
        config=config,
        metrics=metrics or NullCollector(),
    )
    file_tag = _make_file_tag(video_path)
    current_file_tag.set(file_tag)

    try:
        # 1. Identify (IO)
        tracker.set_stage(file_tag, 'identify')
        with ctx.metrics.span('identify'):
            ctx = await asyncio.to_thread(stages['identify'].run, ctx)

        # 2. Extract Reference (IO, with potential GPU OCR)
        tracker.set_stage(file_tag, 'extract_reference')
        with ctx.metrics.span('extract_reference'):
            ctx = await asyncio.to_thread(stages['extract_ref'].run_io, ctx)
        if ctx.pending_ocr:
            tracker.set_gpu_status(file_tag, 'queued')
            ocr_result = await gpu_queue.submit(OcrTask(
                video_path=ctx.video_path,
                track_id=ctx.pending_ocr.get('track_id'),
                output_dir=Path(ctx.pending_ocr['output_dir']),
                ocr_type=ctx.pending_ocr['type'],
                file_tag=file_tag,
            ))
            tracker.set_gpu_status(file_tag, 'none')
            if ocr_result.get('srt_path'):
                ctx.reference_path = ocr_result['srt_path']
            if ocr_result.get('ocr_results'):
                ctx.ocr_results = ocr_result['ocr_results']
            ctx.pending_ocr = None

        # 3. Fetch Subtitles (IO)
        tracker.set_stage(file_tag, 'fetch')
        if config.enable_fetch:
            with ctx.metrics.span('fetch'):
                ctx = await asyncio.to_thread(stages['fetch'].run, ctx)

        # 4. Extract English (IO, with potential GPU OCR)
        tracker.set_stage(file_tag, 'extract')
        with ctx.metrics.span('extract'):
            ctx = await asyncio.to_thread(stages['extract_english'].run_io, ctx)
        if ctx.pending_ocr:
            tracker.set_gpu_status(file_tag, 'queued')
            ocr_result = await gpu_queue.submit(OcrTask(
                video_path=ctx.video_path,
                track_id=ctx.pending_ocr.get('track_id'),
                output_dir=Path(ctx.pending_ocr['output_dir']),
                ocr_type=ctx.pending_ocr['type'],
                file_tag=file_tag,
            ))
            tracker.set_gpu_status(file_tag, 'none')
            if ocr_result.get('srt_path'):
                ctx.english_source = ocr_result['srt_path']
                if ocr_result.get('ocr_results'):
                    ctx.ocr_results = ocr_result['ocr_results']
                from .subtitles import SubtitleProcessor
                ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
                if not ctx.dialogue_lines:
                    raise RuntimeError(f'No dialogue lines found after OCR for {video_path.name}')
            ctx.pending_ocr = None

        if ctx.dialogue_lines is None:
            raise RuntimeError(f'No English subtitle source found for {video_path.name}')

        # 5. Translate (font check IO + translation GPU, concurrent)
        tracker.set_stage(file_tag, 'translate')
        def _on_progress(done: int, total: int, rate: float):
            tracker.set_stage_progress(file_tag, done, total, rate)

        font_task = asyncio.to_thread(stages['translate'].check_fonts, ctx)
        tracker.set_gpu_status(file_tag, 'queued')
        translate_task = gpu_queue.submit(TranslateTask(
            dialogue_lines=ctx.dialogue_lines,
            device=config.device,
            batch_size=config.batch_size,
            model=config.model,
            file_tag=file_tag,
            progress_callback=_on_progress,
        ))
        font_info, translated_lines = await asyncio.gather(font_task, translate_task)
        tracker.set_gpu_status(file_tag, 'none')

        ctx.font_info = font_info
        if not translated_lines:
            raise RuntimeError('Translation failed — empty result')
        ctx.translated_lines = translated_lines

        # 6. Create Tracks (IO)
        tracker.set_stage(file_tag, 'create')
        with ctx.metrics.span('create'):
            ctx = await asyncio.to_thread(stages['create_tracks'].run, ctx)

        # 7. Inpaint if needed (GPU), then Mux (IO)
        tracker.set_stage(file_tag, 'mux')
        if ctx.config.enable_inpaint and ctx.ocr_results:
            inpainted = ctx.work_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
            tracker.set_gpu_status(file_tag, 'queued')
            result_path = await gpu_queue.submit(InpaintTask(
                video_path=ctx.video_path,
                output_path=inpainted,
                ocr_results=ctx.ocr_results,
                device=config.device,
                file_tag=file_tag,
            ))
            tracker.set_gpu_status(file_tag, 'none')
            ctx.inpainted_video = result_path

        with ctx.metrics.span('mux'):
            ctx = await asyncio.to_thread(stages['mux'].run, ctx)

        return ctx

    except Exception as e:
        logger.error(f'Failed: {video_path.name} - {e}')
        raise


async def run_all(
    video_files: list[Path],
    root_dir: Path,
    config: PipelineConfig,
    metrics: MetricsCollector | NullCollector | None,
    gpu_queue: GpuQueue,
    tracker: ProgressTracker,
) -> list[tuple[Path, str]]:
    """Run all files through the pipeline with bounded concurrency."""
    workers = config.workers or min(len(video_files), 4)
    sem = asyncio.Semaphore(workers)

    stages = {
        'identify': IdentifyStage(),
        'extract_ref': ExtractReferenceStage(),
        'fetch': FetchSubtitlesStage(),
        'extract_english': ExtractEnglishStage(),
        'translate': TranslateStage(),
        'create_tracks': CreateTracksStage(),
        'mux': MuxStage(),
    }

    results: list[tuple[Path, str]] = []

    async def run_one(video_path: Path):
        file_tag = _make_file_tag(video_path)
        work_dir = create_work_dir(video_path, root_dir)
        tracker.start_file(file_tag)

        async with sem:
            try:
                from .subtitles import SubtitleExtractor
                has_polish = await asyncio.to_thread(
                    SubtitleExtractor().has_polish_subtitles, video_path
                )
                if has_polish:
                    tracker.complete_file(file_tag, 'skipped')
                    results.append((video_path, 'skipped'))
                    return

                await process_file(
                    video_path, work_dir, config, stages, gpu_queue, tracker,
                    metrics=metrics,
                )
                tracker.complete_file(file_tag, 'success')
                results.append((video_path, 'success'))
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                tracker.complete_file(file_tag, 'failed')
                results.append((video_path, 'failed'))

    tasks = [asyncio.create_task(run_one(vf)) for vf in video_files]
    await asyncio.gather(*tasks)
    return results


def _make_file_tag(video_path: Path) -> str:
    """Create a short display tag from a video path."""
    stem = video_path.stem
    if len(stem) > 20:
        stem = stem[:17] + '...'
    return stem
```

- [ ] **Step 4: Run tests**

Run: `pytest movie_translator/tests/test_async_pipeline.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/async_pipeline.py movie_translator/tests/test_async_pipeline.py
git commit -m "feat(parallel): add async pipeline orchestration with GPU queue integration"
```

---

### Task 10: Wire Into main.py

**Files:**
- Modify: `movie_translator/main.py`

- [ ] **Step 1: Add async entry point**

Replace the main processing loop in `main.py`. Keep the sync path for `--workers 1`:

```python
import asyncio

# In main(), after video_files discovery and config setup:

    workers = args.workers if args.workers > 0 else min(len(video_files), 4)
    config = PipelineConfig(
        device=args.device,
        batch_size=args.batch_size,
        model=args.model,
        enable_fetch=not args.no_fetch,
        enable_inpaint=args.inpaint,
        workers=workers,
    )

    if workers > 1:
        asyncio.run(_async_main(video_files, root_dir, config, args, collector, report_builder))
    else:
        _sync_main(video_files, root_dir, config, args, collector, report_builder)
```

Add `_async_main`:

```python
async def _async_main(video_files, root_dir, config, args, collector, report_builder):
    from .async_pipeline import run_all
    from .gpu_queue import GpuQueue

    gpu_queue = GpuQueue()
    gpu_worker = asyncio.create_task(gpu_queue.run_worker())

    with ProgressTracker(len(video_files), console=console) as tracker:
        results = await run_all(video_files, root_dir, config, collector, gpu_queue, tracker)

    await gpu_queue.shutdown()
    await gpu_worker

    # Handle report builder if metrics enabled
    if report_builder is not None:
        from .metrics.report import build_report, save_report
        report = build_report(
            videos=report_builder.videos,
            config={
                'device': config.device,
                'batch_size': config.batch_size,
                'model': config.model,
                'enable_fetch': config.enable_fetch,
                'enable_inpaint': config.enable_inpaint,
                'workers': config.workers,
            },
        )
        report_path = root_dir / '.translate_temp' / 'metrics.json'
        save_report(report, report_path)
        console.print(f'[dim]Metrics saved to {report_path}[/dim]')
```

Move existing sync loop into `_sync_main`:

```python
def _sync_main(video_files, root_dir, config, args, collector, report_builder):
    """Sequential pipeline — original behavior for --workers 1 or single files."""
    from .pipeline import TranslationPipeline
    from .subtitles import SubtitleExtractor

    extractor = SubtitleExtractor()

    with ProgressTracker(len(video_files), console=console) as tracker:
        pipeline = TranslationPipeline(
            device=config.device,
            batch_size=config.batch_size,
            model=config.model,
            enable_fetch=config.enable_fetch,
            enable_inpaint=config.enable_inpaint,
            tracker=tracker,
            metrics=collector,
        )

        for video_path in video_files:
            # ... existing loop body unchanged ...
```

- [ ] **Step 2: Run all tests**

Run: `pytest --tb=short -q`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add movie_translator/main.py
git commit -m "feat(parallel): wire async pipeline into main entry point with --workers flag"
```

---

### Task 11: Full Integration Verification

- [ ] **Step 1: Run all tests**

```bash
pytest --tb=short -q
```

- [ ] **Step 2: Run linter and formatter**

```bash
ruff check . && ruff format --check .
```

- [ ] **Step 3: Run type checker (if configured)**

```bash
# Check if type checking is part of pre-commit
git stash -u && git stash pop
```

- [ ] **Step 4: Verify single-file mode works (backward compat)**

```bash
# --workers 1 should use the sync path
python -m movie_translator --workers 1 --dry-run /path/to/single/file.mkv
```

- [ ] **Step 5: Verify multi-file mode works**

```bash
# Default workers should process concurrently
python -m movie_translator --dry-run /path/to/anime/season/
```

- [ ] **Step 6: Commit everything and push**

```bash
git push origin feat/metrics-observability
```

---

### Task 12: Real-World Integration Test

- [ ] **Step 1: Copy test files from torrents**

Pick 2-3 episodes from different anime series in the downloads directory. Copy to a temp directory on disk (not tmpfs/RAM).

```bash
# Find available anime
ls ~/Downloads/torrents/completed/ | head -20

# Create test directory on disk
mkdir -p /tmp/movie_translator_test

# Copy a few episodes from different series
cp "path/to/anime1/S01E01.mkv" /tmp/movie_translator_test/
cp "path/to/anime1/S01E02.mkv" /tmp/movie_translator_test/
cp "path/to/anime2/S01E01.mkv" /tmp/movie_translator_test/
```

- [ ] **Step 2: Run with default workers (parallel mode)**

```bash
python -m movie_translator --dry-run --metrics --verbose /tmp/movie_translator_test/
```

Verify:
- TUI shows multiple active files
- GPU tasks serialize (only one translate at a time)
- All files complete successfully
- No errors in log

- [ ] **Step 3: Compare with single-worker baseline**

```bash
python -m movie_translator --dry-run --metrics --workers 1 /tmp/movie_translator_test/
```

Compare wall clock times. Parallel should be faster when there are multiple files.

- [ ] **Step 4: Clean up test files**

```bash
rm -rf /tmp/movie_translator_test
```
