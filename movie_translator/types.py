from pathlib import Path
from typing import NamedTuple


class SubtitleFile(NamedTuple):
    """Subtitle file with metadata for muxing."""

    path: Path
    language: str
    title: str
    is_default: bool


class DialogueLine(NamedTuple):
    """A single dialogue line with timing."""

    start: int
    end: int
    text: str
