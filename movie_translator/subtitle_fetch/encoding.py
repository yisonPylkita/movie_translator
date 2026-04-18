"""Encoding detection and normalization for downloaded subtitle files.

Polish subtitles frequently arrive in CP1250 or ISO-8859-2 encoding.
If not converted to UTF-8, pysubs2 may fail to parse or produce garbled text.
"""

from pathlib import Path

from ..logging import logger

# CP1250 and ISO-8859-2 both decode most bytes successfully but produce
# different characters for Polish letters ą/Ą, ś/Ś, ź/Ź. We try both
# and pick the one that produces more recognizable Polish characters.
_AMBIGUOUS_ENCODINGS = ['cp1250', 'iso-8859-2']
_FALLBACK_ENCODINGS = ['utf-8-sig', 'iso-8859-1']

_POLISH_CHARS = frozenset('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')


def _count_polish(text: str) -> int:
    return sum(1 for c in text if c in _POLISH_CHARS)


def normalize_encoding(path: Path) -> None:
    """Detect encoding of a subtitle file and re-save as UTF-8 if needed.

    Uses Polish character frequency to disambiguate CP1250 vs ISO-8859-2.
    If already UTF-8, does nothing. On failure, leaves file unchanged.
    """
    raw = path.read_bytes()

    # BOM check: UTF-8 BOM → already fine
    if raw.startswith(b'\xef\xbb\xbf'):
        return

    # Try UTF-8 first — if it decodes cleanly, no conversion needed
    try:
        raw.decode('utf-8')
        return
    except UnicodeDecodeError, ValueError:
        pass

    # Try CP1250 and ISO-8859-2, pick whichever produces more Polish characters
    best_text = None
    best_enc = None
    best_score = -1

    for enc in _AMBIGUOUS_ENCODINGS:
        try:
            text = raw.decode(enc)
            score = _count_polish(text)
            if score > best_score:
                best_score = score
                best_text = text
                best_enc = enc
        except UnicodeDecodeError, ValueError:
            continue

    if best_text is not None:
        path.write_text(best_text, encoding='utf-8')
        logger.debug(f'Converted {path.name} from {best_enc} to UTF-8')
        return

    # Last resort: try other encodings
    for enc in _FALLBACK_ENCODINGS:
        try:
            text = raw.decode(enc)
            path.write_text(text, encoding='utf-8')
            logger.debug(f'Converted {path.name} from {enc} to UTF-8')
            return
        except UnicodeDecodeError, ValueError:
            continue

    logger.debug(f'Could not detect encoding for {path.name}, leaving as-is')
