"""Rich TUI progress display for batch video processing.

Provides a live-updating display with:
- Overall progress bar (files processed / total)
- Per-file stage tracker (identify → fetch → extract → translate → mux)
- GPU queue status per file
- Scrolling log panel with color-coded messages
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
    """Live TUI progress display for batch processing."""

    def __init__(self, total_files: int, console: Console | None = None):
        self._console = console or Console()
        self._total = total_files
        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._active_files: dict[str, FileState] = {}
        self._lock = threading.Lock()
        # _current_file tracks the most recently started file for backward compat
        self._current_file = ''
        self._log_lines: deque[tuple[str, str]] = deque(maxlen=24)
        self._batch_start_time = 0.0
        self._live: Live | None = None
        self._log_handler: _LogCapture | None = None

    def __enter__(self):
        self._batch_start_time = time.monotonic()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()

        # Install log handler to capture messages into the TUI panel.
        # Disable propagation to prevent the root logger's RichHandler
        # from also printing to the console (which fights with Live).
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

        # Print final summary
        self._print_summary()

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
            # Called as set_stage_progress(done, total, rate=...) — shouldn't happen
            # but guard against it
            return

        if total_or_rate is None:
            # set_stage_progress(done, total) — old API without rate
            file_name = self._current_file
            done = int(name_or_done)
            total = int(done_or_total)
            r = rate
        elif isinstance(name_or_done, str):
            # set_stage_progress('ep01', done, total, rate)
            file_name = name_or_done
            done = int(done_or_total)
            total = int(total_or_rate)
            r = rate
        else:
            # set_stage_progress(done, total, rate) — old API
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

        if final_status == 'success':
            with self._lock:
                self._completed += 1
            self._add_log(
                f'[green]✓[/green] {self._short_name(file_name)} [dim]({elapsed:.0f}s)[/dim]',
                'DONE',
            )
        elif final_status == 'failed':
            with self._lock:
                self._failed += 1
            self._add_log(
                f'[red]✗[/red] {self._short_name(file_name)} [dim]({elapsed:.0f}s)[/dim]',
                'DONE',
            )
        elif final_status == 'skipped':
            with self._lock:
                self._skipped += 1
            self._add_log(
                f'[yellow]→[/yellow] {self._short_name(file_name)} [dim](skipped)[/dim]',
                'DONE',
            )

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
        """Build the complete TUI display."""
        parts = []

        # Overall progress
        parts.append(self._render_overall())

        # Copy active files under lock, then render without lock
        with self._lock:
            active = dict(self._active_files)

        if active:
            parts.append(self._render_active_files(active))

        # Log panel
        with self._lock:
            log_lines = list(self._log_lines)

        if log_lines:
            parts.append(self._render_logs(log_lines))

        return Group(*parts)

    def _render_overall(self):
        with self._lock:
            done = self._completed + self._failed + self._skipped
            completed = self._completed
            failed = self._failed
            skipped = self._skipped

        elapsed = time.monotonic() - self._batch_start_time

        table = Table.grid(padding=(0, 2))
        table.add_column(ratio=1)
        table.add_column(justify='right')

        # Progress bar
        progress = Progress(
            TextColumn('[bold blue]Movie Translator'),
            BarColumn(),
            MofNCompleteColumn(),
            expand=True,
        )
        progress.add_task('', total=self._total, completed=done)

        # Status counts
        status_parts = []
        if completed:
            status_parts.append(f'[green]{completed} done[/green]')
        if failed:
            status_parts.append(f'[red]{failed} failed[/red]')
        if skipped:
            status_parts.append(f'[yellow]{skipped} skipped[/yellow]')

        status = '  '.join(status_parts) if status_parts else '[dim]starting...[/dim]'
        elapsed_str = f'[dim]{elapsed:.0f}s[/dim]'

        table.add_row(progress, Text.from_markup(f'{status}  {elapsed_str}'))
        return Panel(table, style='blue', padding=(0, 1))

    def _render_active_files(self, active: dict[str, FileState]):
        """Render all active files, each with stage indicators and GPU status."""
        renderables = []

        for file_name, state in active.items():
            name = self._short_name(file_name)
            elapsed = time.monotonic() - state.start_time

            stage_parts = []
            for stage in STAGES:
                label = STAGE_LABELS[stage]

                if stage in state.stages_done:
                    stage_parts.append(f'[green]✓ {label}[/green]')
                elif stage == state.current_stage:
                    if state.gpu_status == 'running':
                        stage_parts.append(f'[bold yellow]▸ {label} [gpu][/bold yellow]')
                    elif state.gpu_status == 'queued':
                        stage_parts.append(f'[yellow]⏳ {label} [queue][/yellow]')
                    else:
                        stage_parts.append(f'[bold yellow]▸ {label}[/bold yellow]')
                else:
                    stage_parts.append(f'[dim]○ {label}[/dim]')

            stages_line = '  '.join(stage_parts)
            file_renderables: list = [
                Text.from_markup(f'[bold]{name}[/bold] [dim]{elapsed:.0f}s[/dim]\n{stages_line}')
            ]

            # Sub-progress bar for current stage (e.g. translation)
            if state.stage_progress is not None:
                done, total, rate = state.stage_progress
                sub = Progress(
                    BarColumn(bar_width=40),
                    MofNCompleteColumn(),
                    TextColumn('•'),
                    TextColumn(f'{rate:.1f} lines/s'),
                    expand=False,
                )
                sub.add_task('', total=total, completed=done)
                file_renderables.append(sub)

            renderables.append(Group(*file_renderables))

        title = f'[cyan]Active Files ({len(active)})'
        return Panel(Group(*renderables), title=title, padding=(0, 1))

    def _render_logs(self, log_lines: list[tuple[str, str]]):
        text = Text()
        max_width = (self._console.width or 120) - 6
        for msg, level in log_lines:
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

    def _short_name(self, name: str) -> str:
        max_len = (self._console.width or 80) - 20
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + '...'
