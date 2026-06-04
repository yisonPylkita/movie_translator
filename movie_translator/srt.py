"""Shared SRT serialization for `DialogueLine` lists.

Single home for the block format + timestamp formatting previously duplicated
across burned-in OCR, PGS OCR, and ASR transcription output.
"""

from __future__ import annotations

from pathlib import Path

from .types import DialogueLine


def format_timestamp(ms: int) -> str:
    """Milliseconds -> SRT `HH:MM:SS,mmm`."""
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'


def write_srt(lines: list[DialogueLine], output_path: Path) -> None:
    """Write `lines` to `output_path` in SRT block format."""
    blocks = [
        f'{i}\n{format_timestamp(line.start_ms)} --> {format_timestamp(line.end_ms)}\n{line.text}\n'
        for i, line in enumerate(lines, 1)
    ]
    output_path.write_text('\n'.join(blocks), encoding='utf-8')
