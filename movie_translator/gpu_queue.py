"""GPU work queue: serialises GPU-bound tasks through a single async worker."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from movie_translator.logging import current_file_tag
from movie_translator.types import BurnedInResult, DialogueLine, OCRResult, ProgressCallback

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


@dataclass
class GpuTask(ABC):
    """A unit of GPU work submitted to the queue."""

    model_type: str = field(init=False)
    file_tag: str = ''

    @abstractmethod
    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> Any:
        """Run the GPU work synchronously. Called from a worker thread."""


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------


@dataclass
class TranslateTask(GpuTask):
    """Translate dialogue lines via the Helsinki-NLP model."""

    model_type: str = field(init=False, default='translate')

    dialogue_lines: list[DialogueLine] = field(default_factory=list)
    device: str = 'cpu'
    batch_size: int = 32
    model: str = 'Helsinki-NLP/opus-mt-en-pl'
    progress_callback: ProgressCallback | None = None

    def execute(
        self, model_cache: dict[str, Any], last_model_type: str | None
    ) -> list[DialogueLine]:
        from movie_translator.translation import translate_dialogue_lines

        return translate_dialogue_lines(
            dialogue_lines=self.dialogue_lines,
            device=self.device,
            batch_size=self.batch_size,
            model=self.model,
            progress_callback=self.progress_callback,
        )


@dataclass
class OcrTask(GpuTask):
    """OCR a subtitle track (PGS bitmap or burned-in)."""

    model_type: str = field(init=False, default='ocr')

    ocr_type: str = 'pgs'  # 'pgs' or 'burned_in'
    video_path: Path = field(default_factory=lambda: Path())
    # PGS-specific
    track_index: int = 0
    work_dir: Path = field(default_factory=lambda: Path())
    # Burned-in-specific
    output_dir: Path = field(default_factory=lambda: Path())
    crop_ratio: float = 0.25
    fps: int = 1

    def execute(
        self, model_cache: dict[str, Any], last_model_type: str | None
    ) -> Path | BurnedInResult | None:
        if self.ocr_type == 'pgs':
            from movie_translator.ocr.pgs_extractor import extract_pgs_track

            return extract_pgs_track(
                video_path=self.video_path,
                track_index=self.track_index,
                work_dir=self.work_dir,
            )
        else:
            from movie_translator.ocr import extract_burned_in_subtitles

            return extract_burned_in_subtitles(
                video_path=self.video_path,
                output_dir=self.output_dir,
                crop_ratio=self.crop_ratio,
                fps=self.fps,
            )


@dataclass
class InpaintTask(GpuTask):
    """Remove burned-in subtitles via inpainting."""

    model_type: str = field(init=False, default='inpaint')

    video_path: Path = field(default_factory=lambda: Path())
    output_path: Path = field(default_factory=lambda: Path())
    ocr_results: list[OCRResult] = field(default_factory=list)
    device: str = 'cpu'
    backend: str = 'lama'

    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> None:
        from movie_translator.inpainting import remove_burned_in_subtitles

        remove_burned_in_subtitles(
            video_path=self.video_path,
            output_path=self.output_path,
            ocr_results=self.ocr_results,
            device=self.device,
            backend=self.backend,
        )


# ---------------------------------------------------------------------------
# Queue + worker
# ---------------------------------------------------------------------------

_SENTINEL: None = None


class GpuQueue:
    """Single-worker async queue that serialises GPU-bound tasks."""

    def __init__(self, tracker=None) -> None:
        self._queue: asyncio.Queue[tuple[GpuTask, asyncio.Future[Any]] | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._last_model_type: str | None = None
        self._model_cache: dict[str, Any] = {}
        self._tracker = tracker  # ProgressTracker for GPU panel updates

    @property
    def pending(self) -> int:
        """Number of tasks waiting in the queue (not including currently executing)."""
        return self._queue.qsize()

    def start(self) -> None:
        """Start the background worker."""
        self._worker_task = asyncio.get_event_loop().create_task(self.run_worker())

    async def submit(self, task: GpuTask) -> Any:
        """Submit a task and wait for its result."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((task, future))
        if self._tracker:
            self._tracker.gpu_queue_size(self._queue.qsize())
        return await future

    async def run_worker(self) -> None:
        """Pull tasks FIFO and execute them one at a time."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break

            task, future = item
            # Set context var for log tagging
            token = current_file_tag.set(task.file_tag)

            if self._tracker:
                self._tracker.gpu_task_started(task.model_type, task.file_tag)

            try:
                result = await asyncio.to_thread(
                    task.execute, self._model_cache, self._last_model_type
                )
                self._last_model_type = task.model_type
                if not future.cancelled():
                    future.set_result(result)
                if self._tracker:
                    self._tracker.gpu_task_completed(task.model_type, task.file_tag)
            except Exception as exc:
                if not future.cancelled():
                    future.set_exception(exc)
                if self._tracker:
                    self._tracker.gpu_task_failed(task.model_type, task.file_tag)
            finally:
                current_file_tag.reset(token)
                self._queue.task_done()

    async def shutdown(self) -> None:
        """Send sentinel and wait for worker to finish."""
        await self._queue.put(_SENTINEL)
        if self._worker_task is not None:
            await self._worker_task
