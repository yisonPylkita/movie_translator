# Parallel Pipeline with GPU Work Queue

**Date:** 2026-03-30
**Status:** Draft

## Problem

The pipeline processes video files sequentially. When given a full anime season (12-24 episodes), each file goes through 7 stages one after another. Most stages are IO-bound (network requests, ffmpeg, file parsing), but the pipeline is bottlenecked by three compute-heavy operations that require dedicated hardware (GPU/NPU): AI translation, OCR, and inpainting.

While one file is translating (30-90s), other files could be fetching subtitles, identifying media, extracting tracks — all IO work that doesn't compete for the GPU. Today, that time is wasted.

## Goals

1. Process multiple files concurrently, with IO-bound stages running in parallel across files.
2. Serialize all GPU/NPU-bound work through a single dedicated worker that avoids model thrashing.
3. Redesign the TUI progress display to show multiple active files simultaneously.
4. Maintain existing stage code with minimal changes — async orchestration wraps the sync stages.

## Non-goals

- Converting stage internals to async (stages stay synchronous).
- Prioritizing or reordering GPU tasks (strict FIFO).
- Distributed processing or multi-GPU support.
- Changing the pipeline stage order or merging/splitting stages.

## Architecture

### Async orchestration with sync stage bridge

The top-level runtime switches from a synchronous `for` loop to `asyncio.run()`. Pipeline workers are lightweight coroutines — one per file, throttled by a semaphore. Existing stage code runs unmodified inside `asyncio.to_thread()` calls.

```
┌─ asyncio event loop ──────────────────────────────────────────┐
│                                                                │
│  Throttle: asyncio.Semaphore(N)                               │
│                                                                │
│  Pipeline Workers (coroutines)                                 │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐             │
│  │ File A │  │ File B │  │ File C │  │ File D │   ...        │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘             │
│      │           │           │           │                    │
│  IO stages: await asyncio.to_thread(stage.run, ctx)           │
│  GPU work:  await gpu_queue.submit(task)                      │
│      │           │           │           │                    │
│      └───────────┴─────┬─────┴───────────┘                    │
│                        │                                       │
│                  ┌─────▼──────┐                               │
│                  │ GPU Queue  │  asyncio.Queue (FIFO)         │
│                  └─────┬──────┘                               │
│                        │                                       │
│                  ┌─────▼────────────────┐                     │
│                  │ GPU Worker (1 coro)  │                     │
│                  │ pulls task           │                     │
│                  │ to_thread(execute)   │                     │
│                  │ sets Future result   │                     │
│                  └──────────────────────┘                     │
└────────────────────────────────────────────────────────────────┘
```

### Why asyncio over raw threading

The codebase already uses `ThreadPoolExecutor` internally (fetch stage, translate stage). Adding another layer of raw threads for inter-file parallelism creates nested thread pools that are hard to reason about.

Asyncio provides structured concurrency: the event loop is the single coordinator, coroutines suspend cleanly at `await` points, and `asyncio.to_thread()` bridges into the existing sync code. This maps directly to how the architecture would look in Rust with tokio — spawn lightweight tasks, await on IO or channel results, send compute to a dedicated worker.

Each pipeline worker coroutine controls its own file's progression through stages. When it needs GPU work, it submits a task and suspends. The event loop is free to run other workers while the coroutine waits. No manual thread coordination, no deadlock risk from nested locks.

## GPU Work Queue

### Interface

```python
class GpuQueue:
    """Single-consumer async queue for GPU/NPU-bound tasks."""

    def __init__(self):
        self._queue: asyncio.Queue[tuple[GpuTask, asyncio.Future]] = asyncio.Queue()

    async def submit(self, task: GpuTask) -> Any:
        """Submit a task and await its result.

        The calling coroutine suspends until the GPU worker completes the task.
        If the task fails, the exception propagates to the caller.
        """
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((task, future))
        return await future

    async def run_worker(self):
        """GPU worker loop. Runs as a single long-lived coroutine.

        Pulls tasks FIFO, executes each to completion via to_thread,
        and delivers the result (or exception) through the Future.
        Keeps the last model loaded to avoid unnecessary reloads.
        Shuts down when it receives a None sentinel.
        """
        last_model_type: str | None = None
        model_cache: dict[str, Any] = {}

        while True:
            item = await self._queue.get()
            if item is None:
                break
            task, future = item
            try:
                # Set file context for logging from GPU thread
                _current_file.set(task.file_tag)
                result = await asyncio.to_thread(
                    task.execute, model_cache, last_model_type
                )
                last_model_type = task.model_type
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
```

### Task types

Each GPU task is a complete, self-contained operation. The GPU worker finishes one task entirely before starting the next. No interleaving, no partial work.

| Task | model_type | Input | Output | Description |
|------|-----------|-------|--------|-------------|
| `TranslateTask` | `"translate"` | Dialogue lines, config (device, batch_size, model) | List of translated `DialogueLine` objects | Translates all dialogue lines for one episode. Loads Helsinki-NLP/Opus-MT model if not cached. |
| `OcrTask` | `"ocr"` | Video path, track info or frame data, work dir | OCR results + extracted subtitle path | OCR of all PGS bitmap subtitles or all burned-in subtitle frames for one episode. Uses Apple Vision. |
| `InpaintTask` | `"inpaint"` | Video path, OCR bounding boxes, work dir, device | Path to inpainted video | Inpaints all frames with burned-in subtitles for one episode. Loads LaMa model if not cached. |

### Model caching strategy

The GPU worker keeps the last-used model loaded. When the next task has the same `model_type`, the cached model is reused. When the type changes, the old model is unloaded (memory freed) and the new one loaded.

In practice, tasks will naturally cluster by type. If files A, B, C all reach the translate stage around the same time, the queue sees `[translate(A), translate(B), translate(C)]` — three translations back-to-back with no model swap.

No reordering is done to achieve this. The FIFO order is respected. The caching just avoids gratuitous reloads when same-type tasks happen to be adjacent.

### Error isolation

A failed GPU task sets the exception on that task's Future. The awaiting pipeline worker catches it and marks its file as failed. The GPU worker continues to the next task in the queue. One file's failure never poisons the queue or affects other workers.

The GPU worker itself only crashes on truly unrecoverable errors (e.g., device lost). In that case, all pending Futures receive the exception and all workers fail gracefully.

## Pipeline Worker Coroutine

Each file gets one coroutine that orchestrates its progression through stages. The coroutine decides whether each stage runs in the worker (IO) or goes to the GPU queue (compute).

```python
async def process_file(
    video_path: Path,
    work_dir: Path,
    config: PipelineConfig,
    stages: StageRegistry,  # simple container holding one instance of each stage
    gpu_queue: GpuQueue,
    tracker: ProgressTracker,
):
    ctx = PipelineContext(video_path=video_path, work_dir=work_dir, config=config)

    # 1. Identify (IO)
    ctx = await asyncio.to_thread(stages.identify.run, ctx)

    # 2. Extract Reference (IO, with potential GPU submission for OCR)
    ctx = await asyncio.to_thread(stages.extract_ref.run_io, ctx)
    if ctx.pending_ocr:
        ocr_result = await gpu_queue.submit(OcrTask.from_context(ctx))
        ctx = ctx.with_ocr_result(ocr_result)

    # 3. Fetch Subtitles (IO — internal ThreadPool for providers)
    ctx = await asyncio.to_thread(stages.fetch.run, ctx)

    # 4. Extract English (IO — may submit OCR if fallback needed)
    ctx = await asyncio.to_thread(stages.extract_english.run_io, ctx)
    if ctx.pending_ocr:
        ocr_result = await gpu_queue.submit(OcrTask.from_context(ctx))
        ctx = ctx.with_ocr_result(ocr_result)

    # 5. Translate (font check IO + translation GPU, concurrent)
    font_task = asyncio.to_thread(stages.translate.check_fonts, ctx)
    translate_task = gpu_queue.submit(TranslateTask.from_context(ctx))
    font_info, translated_lines = await asyncio.gather(font_task, translate_task)
    ctx = ctx.with_translation(translated_lines, font_info)

    # 6. Create Tracks (IO)
    ctx = await asyncio.to_thread(stages.create_tracks.run, ctx)

    # 7. Inpaint if needed (GPU), then Mux (IO)
    if ctx.config.enable_inpaint and ctx.ocr_results:
        inpaint_result = await gpu_queue.submit(InpaintTask.from_context(ctx))
        ctx = ctx.with_inpaint_result(inpaint_result)
    ctx = await asyncio.to_thread(stages.mux.run, ctx)
```

### Stage refactoring for mixed IO/compute stages

Two stages currently mix IO and compute work internally: `extract_ref` and `extract_english`. Both can trigger OCR (compute) as a fallback when image-based subtitles are encountered.

These stages gain a split interface:

- **`run_io(ctx)`** — performs the IO-only work (track discovery, text extraction, subtitle parsing). If OCR is needed, sets `ctx.pending_ocr` with the necessary parameters but does not execute it.
- **`run(ctx)`** — preserved for backward compatibility and non-parallel execution. Performs everything including OCR inline.

The `translate` stage similarly splits: `check_fonts(ctx)` (IO) is separated from the translation work that goes through the GPU queue.

The `mux` stage loses its internal inpainting call. Inpainting is submitted to the GPU queue by the worker coroutine before mux runs. The mux stage receives `ctx.inpainted_video` (already set) and uses it as the source for ffmpeg muxing.

### Worker throttling

A `asyncio.Semaphore(N)` limits how many pipeline workers run concurrently, where `N = min(file_count, --workers)`. This caps concurrent ffmpeg processes, open file handles, and memory usage.

All files are queued as coroutines at startup, but only N are actively running at any time. When one completes, the next file's coroutine starts.

```python
async def run_all(video_files, config, gpu_queue, tracker):
    sem = asyncio.Semaphore(config.workers)

    async def throttled(video_path, work_dir):
        async with sem:
            await process_file(video_path, work_dir, config, ..., gpu_queue, tracker)

    tasks = [
        asyncio.create_task(throttled(vf, create_work_dir(vf, root)))
        for vf in video_files
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
```

## Progress Tracking (TUI Redesign)

### Multi-file display

The current `ProgressTracker` tracks one file at a time. The new design tracks N active files simultaneously.

```
┌─ Movie Translator ──────────────────────────────────────────────┐
│ ████████████░░░░░░  12/24    8 done  0 fail  4 skip      142s  │
├─ Active ────────────────────────────────────────────────────────┤
│ KonoSuba.05   ✓ ✓ ✓ ✓ ▸ Translate  ████░░ 312/508  4.2/s [gpu]│
│ KonoSuba.06   ✓ ✓ ✓ ○ ⏳ Translate                     [queue] │
│ OPM.03        ✓ ▸ Fetch                                        │
│ OPM.04        ▸ Identify                                       │
├─ Log ───────────────────────────────────────────────────────────┤
│ ✓ KonoSuba.04 (67s)                                            │
│ ✓ OPM.02 (52s)                                                 │
│ [OPM.03] Fetched 3 PL subs from napiprojekt                    │
│ [Kono.06] Extracted 412 dialogue lines                          │
└─────────────────────────────────────────────────────────────────┘
```

### Per-file state

```python
@dataclass
class FileState:
    name: str                    # Short display name
    current_stage: str           # Current stage name
    stages_done: list[str]       # Completed stages
    gpu_status: str              # 'none' | 'running' | 'queued'
    stage_progress: tuple[int, int, float] | None  # (done, total, rate)
    start_time: float
```

The tracker holds `_active_files: dict[str, FileState]` protected by a `threading.Lock`. All public methods (`start_file`, `set_stage`, `set_gpu_status`, `set_stage_progress`, `complete_file`) acquire the lock before mutating state.

Rich's `Live.update()` is thread-safe. The render method reads the locked state and builds the display.

### Stage indicators per file

Each active file row shows compact stage indicators:

- `✓` — stage completed
- `▸ StageName` — currently running in worker
- `⏳ StageName` — submitted to GPU queue, waiting
- `▸ StageName [gpu]` — running on GPU worker
- `○` — pending, not started

### Log message tagging

A `ContextVar[str]` stores the current file's short name per worker. The `_LogCapture` handler prepends it automatically:

```python
_current_file: ContextVar[str] = ContextVar('current_file', default='')

class _LogCapture(logging.Handler):
    def emit(self, record):
        tag = _current_file.get()
        prefix = f'[{tag}] ' if tag else ''
        self._tracker._add_log(f'{prefix}{record.getMessage()}', record.levelname)
```

Existing `logger.info(...)` calls throughout stage code gain file context without any changes to those call sites. The worker coroutine sets the ContextVar before running each stage via `to_thread`.

For GPU tasks, the ContextVar doesn't automatically propagate into the GPU worker's `to_thread` call (different thread context). Each `GpuTask` carries a `file_tag: str` field. The GPU worker sets `_current_file` before calling `task.execute`, so log messages from GPU-bound work (translation progress, OCR frame counts) are correctly tagged.

### GPU queue status in TUI

The tracker exposes `set_gpu_status(file_name, status)` where status is `'queued'`, `'running'`, or `'none'`. The GPU queue calls this when a task is submitted (queued), when the worker picks it up (running), and when it completes (none). This drives the `[gpu]`/`[queue]` indicators.

## Configuration Changes

### New CLI flag

```
--workers N    Number of concurrent pipeline workers (default: auto)
```

Auto default: `min(file_count, 4)`.

### PipelineConfig addition

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    workers: int = 4
```

## Entry Point Changes (main.py)

The synchronous `main()` function delegates to an async `async_main()`:

```python
def main():
    args = parse_args()
    # ... validation, dependency checks ...
    asyncio.run(async_main(args, video_files, root_dir))

async def async_main(args, video_files, root_dir):
    gpu_queue = GpuQueue()
    gpu_worker_task = asyncio.create_task(gpu_queue.run_worker())

    with ProgressTracker(len(video_files), console=console) as tracker:
        await run_all(video_files, config, gpu_queue, tracker)

    await gpu_queue.shutdown()  # sends sentinel, awaits worker exit
    await gpu_worker_task
```

## Files Changed

| File | Change |
|------|--------|
| `main.py` | Replace sync loop with `asyncio.run()` + `async_main()`. Add `--workers` flag. |
| `pipeline.py` | Extract `process_file` coroutine. Remove `TranslationPipeline` class or keep as config holder. |
| `gpu_queue.py` | **New.** `GpuQueue`, `GpuTask` base, `TranslateTask`, `OcrTask`, `InpaintTask`. |
| `progress.py` | Multi-file `FileState` dict, `threading.Lock`, redesigned `_render()` with Active panel. Add `set_gpu_status()`. |
| `logging.py` | Add `_current_file` ContextVar. |
| `context.py` | Add `workers` to `PipelineConfig`. Add `pending_ocr` field to `PipelineContext`. |
| `stages/extract_ref.py` | Add `run_io()` method that defers OCR. Keep `run()` for compat. |
| `stages/extract_english.py` | Add `run_io()` method that defers OCR. Keep `run()` for compat. |
| `stages/translate.py` | Extract `check_fonts()` as standalone. Translation logic moves to `TranslateTask.execute()`. |
| `stages/mux.py` | Remove internal inpainting call. Expect `ctx.inpainted_video` to be pre-set. |

## Testing Strategy

1. **Unit tests for GpuQueue** — submit tasks, verify FIFO order, verify error isolation, verify model cache reuse.
2. **Integration test with mock GPU tasks** — run 3-4 files through the async pipeline with fake stages that sleep to simulate IO/compute timing. Verify concurrent execution of IO stages and serial execution of GPU tasks.
3. **Regression test** — run against existing benchmark files (Code Geass + OPM S01E01) and compare metrics output. Total wall time should decrease; per-file metrics should be similar.
4. **TUI visual test** — manual verification that the multi-file display renders correctly with 1, 2, 4 concurrent files.

## Migration Path

The refactoring is backward-compatible in the sense that the pipeline produces the same output for each file. The stage order, stage logic, and output formats are unchanged. The only behavioral change is that multiple files process concurrently.

Single-file invocation (`movie-translator file.mkv`) behaves identically to today — one worker, no contention for the GPU queue.

The `--workers 1` flag restores fully sequential behavior for debugging.
