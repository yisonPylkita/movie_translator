"""Progress tracker for batch video processing.

Manages state for files, stages, GPU tasks, and log capture.
TUI rendering is delegated to TuiRenderer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live

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
    name: str
    start_time: float
    current_stage: str = ''
    stages_done: list[str] = field(default_factory=list)
    gpu_status: str = 'none'  # 'none' | 'queued' | 'running'
    stage_progress: tuple[int, int, float] | None = None


@dataclass
class GpuWorkerState:
    """Tracks the GPU worker's current activity."""

    current_task_type: str = ''  # 'translate' | 'ocr' | 'inpaint' | ''
    current_file: str = ''
    progress: tuple[int, int, float] | None = None  # (done, total, rate)
    queue_depth: int = 0
    start_time: float = 0.0
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=6))


class _LogCapture(logging.Handler):
    """Captures log records and feeds them to the progress tracker."""

    def __init__(self, tracker: ProgressTracker):
        super().__init__()
        self._tracker = tracker

    def emit(self, record):
        try:
            from .logging import current_file_tag

            msg = record.getMessage()
            tag = current_file_tag.get('')
            if tag:
                msg = f'[dim]\\[{tag}][/dim] {msg}'
            self._tracker._add_log(msg, record.levelname)
        except Exception:
            pass


class ProgressTracker:
    """Tracks progress state for batch video processing.

    TUI rendering is delegated to TuiRenderer.
    """

    def __init__(self, total_files: int, console: Console | None = None):
        self._console = console or Console()
        self._total = total_files
        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._active_files: dict[str, FileState] = {}
        self._gpu: GpuWorkerState = GpuWorkerState()
        self._lock = threading.Lock()
        self._current_file = ''
        self._log_lines: deque[tuple[str, str]] = deque(maxlen=24)
        self._batch_start_time = 0.0
        self._live: Live | None = None
        self._log_handler: _LogCapture | None = None
        self._renderer = None

    def __enter__(self):
        from .tui_renderer import TuiRenderer

        self._batch_start_time = time.monotonic()
        self._renderer = TuiRenderer(self._console, self._total, self._batch_start_time)

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

        if self._renderer:
            self._renderer.render_summary(self._completed, self._failed, self._skipped)

    # ------------------------------------------------------------------
    # Public API — all methods accept an optional file_name as first arg.
    # When omitted (old single-arg API), _current_file is used instead.
    # ------------------------------------------------------------------

    def start_file(self, name: str):
        """Register a new file as active."""
        state = FileState(name=name, start_time=time.monotonic())
        with self._lock:
            self._active_files[name] = state
            self._current_file = name
        self._update()

    def set_stage(self, name_or_stage: str, stage: str = ''):
        """Update the current stage for a file.

        Two-arg form (new API):  set_stage('ep01', 'identify')
        One-arg form (old API):  set_stage('identify')   — uses _current_file
        """
        if stage:
            file_name = name_or_stage
            new_stage = stage
        else:
            file_name = self._current_file
            new_stage = name_or_stage

        with self._lock:
            state = self._active_files.get(file_name)
            if state is None:
                return
            if state.current_stage and state.current_stage not in state.stages_done:
                state.stages_done.append(state.current_stage)
            state.current_stage = new_stage
            state.stage_progress = None

        self._update()

    def set_gpu_status(self, file_name: str, status: str):
        """Set the GPU queue status for a file: 'none' | 'queued' | 'running'."""
        with self._lock:
            state = self._active_files.get(file_name)
            if state is None:
                return
            state.gpu_status = status
        self._update()

    def set_stage_progress(
        self, name_or_done, done_or_total=None, total_or_rate=None, rate: float = 0.0
    ):
        """Update sub-progress for a stage (e.g. translation lines).

        Two-call forms:
          New API: set_stage_progress('ep01', done, total, rate)
          Old API: set_stage_progress(done, total, rate)
        """
        if done_or_total is None:
            return

        if total_or_rate is None:
            file_name = self._current_file
            done = int(name_or_done)
            total = int(done_or_total)
            r = rate
        elif isinstance(name_or_done, str):
            file_name = name_or_done
            done = int(done_or_total)
            total = int(total_or_rate)
            r = rate
        else:
            file_name = self._current_file
            done = int(name_or_done)
            total = int(done_or_total)
            r = float(total_or_rate)

        with self._lock:
            state = self._active_files.get(file_name)
            if state is None:
                return
            state.stage_progress = (done, total, r)

        self._update()

    def complete_file(self, name_or_status: str, status: str = ''):
        """Mark a file as done. Status: 'success' | 'failed' | 'skipped'.

        Two-arg form (new API):  complete_file('ep01', 'success')
        One-arg form (old API):  complete_file('success')   — uses _current_file
        """
        if status:
            file_name = name_or_status
            final_status = status
        else:
            file_name = self._current_file
            final_status = name_or_status

        with self._lock:
            state = self._active_files.pop(file_name, None)

        if state is None:
            return

        elapsed = time.monotonic() - state.start_time
        short = self._short_name(file_name)

        if final_status == 'success':
            with self._lock:
                self._completed += 1
            self._add_log(f'[green]✓[/green] {short} [dim]({elapsed:.0f}s)[/dim]', 'DONE')
        elif final_status == 'failed':
            with self._lock:
                self._failed += 1
            self._add_log(f'[red]✗[/red] {short} [dim]({elapsed:.0f}s)[/dim]', 'DONE')
        elif final_status == 'skipped':
            with self._lock:
                self._skipped += 1
            self._add_log(f'[yellow]→[/yellow] {short} [dim](skipped)[/dim]', 'DONE')

        self._update()

    # ------------------------------------------------------------------
    # GPU worker API
    # ------------------------------------------------------------------

    def gpu_task_started(self, task_type: str, file_tag: str):
        """Called by the GPU worker when it picks up a task."""
        with self._lock:
            self._gpu.current_task_type = task_type
            self._gpu.current_file = file_tag
            self._gpu.progress = None
            self._gpu.start_time = time.monotonic()
            self._gpu.queue_depth = max(0, self._gpu.queue_depth - 1)
        self._update()

    def gpu_task_progress(self, done: int, total: int, rate: float = 0.0):
        """Update sub-progress for the running GPU task."""
        with self._lock:
            self._gpu.progress = (done, total, rate)
        self._update()

    def gpu_task_completed(self, task_type: str, file_tag: str):
        """Called by the GPU worker when a task finishes."""
        with self._lock:
            elapsed = time.monotonic() - self._gpu.start_time
            label = {'translate': 'Translated', 'ocr': 'OCR', 'inpaint': 'Inpainted'}.get(
                task_type, task_type
            )
            if self._gpu.progress:
                _done, total, _rate = self._gpu.progress
                detail = f' ({total} lines, {elapsed:.1f}s)'
            else:
                detail = f' ({elapsed:.1f}s)'
            self._gpu.recent.append(f'[green]✓[/green] {label} {file_tag}{detail}')
            self._gpu.current_task_type = ''
            self._gpu.current_file = ''
            self._gpu.progress = None
        self._update()

    def gpu_task_failed(self, task_type: str, file_tag: str):
        """Called by the GPU worker when a task fails."""
        with self._lock:
            elapsed = time.monotonic() - self._gpu.start_time
            label = {'translate': 'Translate', 'ocr': 'OCR', 'inpaint': 'Inpaint'}.get(
                task_type, task_type
            )
            self._gpu.recent.append(f'[red]✗[/red] {label} {file_tag} ({elapsed:.1f}s)')
            self._gpu.current_task_type = ''
            self._gpu.current_file = ''
            self._gpu.progress = None
        self._update()

    def gpu_queue_size(self, size: int):
        """Update the GPU queue depth display."""
        with self._lock:
            self._gpu.queue_depth = size
        self._update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_log(self, message: str, level: str):
        clean = message.strip()
        if not clean:
            return
        with self._lock:
            self._log_lines.append((clean, level))
        self._update()

    def _update(self):
        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Snapshot state and delegate to TuiRenderer."""
        if self._renderer is None:
            from rich.console import Group

            return Group()

        with self._lock:
            active = dict(self._active_files)
            gpu_snapshot = (
                self._gpu.current_task_type,
                self._gpu.current_file,
                self._gpu.progress,
                self._gpu.queue_depth,
                list(self._gpu.recent),
            )
            log_lines = list(self._log_lines)

        return self._renderer.render(
            completed=self._completed,
            failed=self._failed,
            skipped=self._skipped,
            active_files=active,
            gpu=gpu_snapshot,
            log_lines=log_lines,
        )

    def _short_name(self, name: str) -> str:
        max_len = (self._console.width or 80) - 20
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + '...'
