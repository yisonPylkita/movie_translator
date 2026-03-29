"""Subtitle alignment using ilass (improved alass).

Uses the ilass CLI tool for subtitle-to-subtitle alignment via dynamic
programming with split penalties. Handles OP removal, ad breaks, and
other structural differences automatically without heuristic gap detection.

ilass is built from source in vendor/ilass and must be compiled before use.
See: https://github.com/SandroHc/ilass
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..logging import logger

# Path to the ilass binary, resolved relative to project root.
_ILASS_BINARY = (
    Path(__file__).resolve().parent.parent.parent
    / 'vendor'
    / 'ilass'
    / 'target'
    / 'release'
    / 'ilass'
)


def is_available() -> bool:
    """Check if the ilass binary is built and available."""
    return _ILASS_BINARY.is_file()


def align_to_reference(
    subtitle_path: Path,
    reference_path: Path,
    split_penalty: float = 7.0,
) -> bool:
    """Align a subtitle file to a reference using ilass.

    Uses ilass's DP algorithm with split penalties to find per-line
    offsets. This handles OP removal, ad breaks, and other structural
    differences automatically.

    The subtitle file is modified in place.

    Args:
        subtitle_path: Path to the subtitle file to realign.
        reference_path: Path to the reference subtitle file.
        split_penalty: Controls eagerness to avoid splits (0-1000).
            Lower values allow more split points. Default 7 works
            well for anime with OP/ED removal.

    Returns:
        True if alignment succeeded, False otherwise.
    """
    if not is_available():
        logger.warning('ilass binary not found at %s', _ILASS_BINARY)
        return False

    # ilass writes to a new file — we'll use a temp file then replace
    output_path = subtitle_path.with_suffix('.ilass_tmp' + subtitle_path.suffix)

    try:
        result = subprocess.run(
            [
                str(_ILASS_BINARY),
                str(reference_path),
                str(subtitle_path),
                str(output_path),
                '--split-penalty',
                str(split_penalty),
                '--disable-fps-guessing',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning('ilass failed (exit %d): %s', result.returncode, result.stderr)
            return False

        # Log the alignment summary from stderr
        for line in result.stderr.splitlines():
            if line.startswith('shifted block'):
                logger.info('ilass: %s', line)

        # Replace original with aligned output
        shutil.move(str(output_path), str(subtitle_path))
        return True

    except subprocess.TimeoutExpired:
        logger.warning('ilass timed out after 120s')
        return False
    except Exception as e:
        logger.warning('ilass error: %s', e)
        return False
    finally:
        output_path.unlink(missing_ok=True)
