"""Rich TUI rendering for the progress display.

Receives snapshots of progress state and builds Rich renderables.
Decoupled from the ProgressTracker's state management and threading.
"""

from __future__ import annotations

import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .progress import LEVEL_STYLES, STAGE_LABELS, STAGES, FileState

# Type alias for GPU worker snapshot
GpuSnapshot = tuple[str, str, tuple[int, int, float] | None, int, list[str]]


class TuiRenderer:
    """Renders the TUI display from state snapshots."""

    def __init__(self, console: Console, total_files: int, batch_start_time: float):
        self._console = console
        self._total = total_files
        self._batch_start_time = batch_start_time

    def render(
        self,
        *,
        completed: int,
        failed: int,
        skipped: int,
        active_files: dict[str, FileState],
        gpu: GpuSnapshot,
        log_lines: list[tuple[str, str]],
    ) -> Group:
        """Build the complete TUI display from state snapshots."""
        parts = []

        parts.append(self._render_overall(completed, failed, skipped))

        if active_files:
            parts.append(self._render_active_files(active_files))

        gpu_task_type, gpu_file, gpu_progress, gpu_queue_depth, gpu_recent = gpu
        if gpu_task_type or gpu_recent or gpu_queue_depth:
            parts.append(
                self._render_gpu_panel(
                    gpu_task_type, gpu_file, gpu_progress, gpu_queue_depth, gpu_recent
                )
            )

        if log_lines:
            parts.append(self._render_logs(log_lines))

        return Group(*parts)

    def _render_overall(self, completed: int, failed: int, skipped: int):
        done = completed + failed + skipped
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
        """Render all active files with stage indicators and GPU status."""
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

    def _render_gpu_panel(
        self,
        task_type: str,
        file_tag: str,
        progress: tuple[int, int, float] | None,
        queue_depth: int,
        recent: list[str],
    ):
        """Render the dedicated GPU worker panel."""
        renderables: list = []

        if task_type:
            label = {'translate': 'Translating', 'ocr': 'Running OCR', 'inpaint': 'Inpainting'}.get(
                task_type, task_type
            )
            line = f'[bold magenta]▸[/bold magenta] {label} [bold]{file_tag}[/bold]'
            renderables.append(Text.from_markup(line))

            if progress:
                done, total, rate = progress
                sub = Progress(
                    BarColumn(bar_width=40),
                    MofNCompleteColumn(),
                    TextColumn('•'),
                    TextColumn(f'{rate:.1f}/s'),
                    expand=False,
                )
                sub.add_task('', total=total, completed=done)
                renderables.append(sub)
        else:
            renderables.append(Text.from_markup('[dim]idle[/dim]'))

        if queue_depth:
            renderables.append(Text.from_markup(f'[dim]Queue: {queue_depth} pending[/dim]'))

        if recent:
            renderables.append(Text.from_markup(''))
            for entry in recent:
                try:
                    renderables.append(Text.from_markup(entry))
                except Exception:
                    renderables.append(Text(entry))

        return Panel(
            Group(*renderables), title='[magenta]GPU Worker', border_style='magenta', padding=(0, 1)
        )

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

    def render_summary(self, completed: int, failed: int, skipped: int):
        """Print the final summary line to the console."""
        self._console.print()
        parts = []
        if completed:
            parts.append(f'[green]✓ {completed} translated[/green]')
        if failed:
            parts.append(f'[red]✗ {failed} failed[/red]')
        if skipped:
            parts.append(f'[yellow]→ {skipped} skipped[/yellow]')

        elapsed = time.monotonic() - self._batch_start_time
        self._console.print(' | '.join(parts) + f'  [dim]({elapsed:.0f}s total)[/dim]')

    def _short_name(self, name: str) -> str:
        max_len = (self._console.width or 80) - 20
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + '...'
