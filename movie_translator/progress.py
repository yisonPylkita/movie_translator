"""Rich TUI progress display for batch video processing.

Provides a live-updating display with:
- Overall progress bar (files processed / total)
- Current file stage tracker (identify → fetch → extract → translate → mux)
- Scrolling log panel with color-coded messages
"""

from __future__ import annotations

import logging
import time
from collections import deque

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


class _LogCapture(logging.Handler):
    """Captures log records and feeds them to the progress tracker."""

    def __init__(self, tracker: ProgressTracker):
        super().__init__()
        self._tracker = tracker

    def emit(self, record):
        try:
            msg = record.getMessage()
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
        self._current_file = ''
        self._current_stage = ''
        self._stages_done: list[str] = []
        self._stage_info: dict[str, str] = {}  # Extra info per stage
        self._log_lines: deque[tuple[str, str]] = deque(maxlen=24)
        self._file_start_time = 0.0
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

    def start_file(self, name: str):
        self._current_file = name
        self._current_stage = ''
        self._stages_done = []
        self._stage_info = {}
        self._file_start_time = time.monotonic()
        self._update()

    def set_stage(self, stage: str, info: str = ''):
        if self._current_stage and self._current_stage not in self._stages_done:
            self._stages_done.append(self._current_stage)
        self._current_stage = stage
        if info:
            self._stage_info[stage] = info
        self._update()

    def complete_file(self, status: str):
        """Mark current file as done. Status: 'success', 'failed', 'skipped'."""
        if self._current_stage and self._current_stage not in self._stages_done:
            self._stages_done.append(self._current_stage)

        elapsed = time.monotonic() - self._file_start_time
        if status == 'success':
            self._completed += 1
            self._add_log(
                f'[green]✓[/green] {self._short_name(self._current_file)} '
                f'[dim]({elapsed:.0f}s)[/dim]',
                'DONE',
            )
        elif status == 'failed':
            self._failed += 1
            self._add_log(
                f'[red]✗[/red] {self._short_name(self._current_file)} [dim]({elapsed:.0f}s)[/dim]',
                'DONE',
            )
        elif status == 'skipped':
            self._skipped += 1
            self._add_log(
                f'[yellow]→[/yellow] {self._short_name(self._current_file)} [dim](skipped)[/dim]',
                'DONE',
            )

        self._current_file = ''
        self._current_stage = ''
        self._update()

    def _add_log(self, message: str, level: str):
        # Clean up rich markup that logger might not produce
        clean = message.strip()
        if not clean:
            return
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

        # Current file (if processing)
        if self._current_file:
            parts.append(self._render_current_file())

        # Log panel
        if self._log_lines:
            parts.append(self._render_logs())

        return Group(*parts)

    def _render_overall(self):
        done = self._completed + self._failed + self._skipped
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

    def _render_current_file(self):
        name = self._short_name(self._current_file)
        elapsed = time.monotonic() - self._file_start_time

        # Stage indicators
        stage_parts = []
        for stage in STAGES:
            label = STAGE_LABELS[stage]
            info = self._stage_info.get(stage, '')
            info_str = f' ({info})' if info else ''

            if stage in self._stages_done:
                stage_parts.append(f'[green]✓ {label}{info_str}[/green]')
            elif stage == self._current_stage:
                stage_parts.append(f'[bold yellow]▸ {label}{info_str}[/bold yellow]')
            else:
                stage_parts.append(f'[dim]○ {label}[/dim]')

        stages_line = '  '.join(stage_parts)

        content = Text.from_markup(f'[bold]{name}[/bold] [dim]{elapsed:.0f}s[/dim]\n{stages_line}')
        return Panel(content, title='[cyan]Current File', padding=(0, 1))

    def _render_logs(self):
        text = Text()
        max_width = (self._console.width or 120) - 6
        for msg, level in self._log_lines:
            display_msg = msg[:max_width] + '...' if len(msg) > max_width else msg
            base_style = LEVEL_STYLES.get(level, 'white')
            if level == 'DONE':
                base_style = 'white'
            # Parse Rich markup in the message (e.g. [green]✓[/green])
            # so existing colors are preserved; the level style is the default.
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
        max_len = (self._console.width or 80) - 20  # leave room for elapsed time + padding
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + '...'
