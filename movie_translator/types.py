from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

# Callback receives (lines_done, total_lines, lines_per_second)
ProgressCallback = Callable[[int, int, float], None]

# Styles that indicate non-dialogue content (signs, songs, etc.)
# Covers common fansub naming: OP/ED/IN (insert song) layers with
# romaji (OPRO/INRO) and English (OPEN/INEN) suffixes.
NON_DIALOGUE_STYLES = ('sign', 'song', 'title', 'op', 'ed', 'insert', 'inro', 'inen')

# Polish diacritical characters
POLISH_CHARS = 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'

# Mapping to replace Polish chars with ASCII equivalents
POLISH_CHAR_MAP = str.maketrans(
    'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ',
    'acelnoszzACELNOSZZ',
)


def replace_polish_chars(text: str) -> str:
    """Replace Polish diacritical characters with ASCII equivalents."""
    return text.translate(POLISH_CHAR_MAP)


class SubtitleFile(NamedTuple):
    path: Path
    language: str
    title: str
    is_default: bool


class DialogueLine(NamedTuple):
    start_ms: int
    end_ms: int
    text: str


class BoundingBox(NamedTuple):
    x: float  # normalized 0-1, left edge
    y: float  # normalized 0-1, top edge (top-left origin)
    width: float  # normalized 0-1
    height: float  # normalized 0-1


class OCRResult(NamedTuple):
    timestamp_ms: int
    text: str
    boxes: list[BoundingBox]


class BurnedInResult(NamedTuple):
    srt_path: Path
    ocr_results: list[OCRResult]
