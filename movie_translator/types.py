from pathlib import Path
from typing import NamedTuple


class SubtitleFile(NamedTuple):
    path: Path
    language: str
    title: str
    is_default: bool


class DialogueLine(NamedTuple):
    start_ms: int
    end_ms: int
    text: str
