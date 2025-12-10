from pathlib import Path
from typing import NamedTuple

# Styles that indicate non-dialogue content (signs, songs, etc.)
NON_DIALOGUE_STYLES = ('sign', 'song', 'title', 'op', 'ed')

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
