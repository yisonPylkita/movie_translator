"""Align the OCR'd Polish .srt to the local video's timeline.

The OCR'd subtitles are timed to the ogladajanime encode, which can differ
from the user's local video (different start offset, OP/ED present or not).
To be usable on the local file they must be aligned to a reference that IS on
the local timeline — its English subtitle track.

We reuse the exact same engine the main pipeline uses: the vendored **ilass**
binary (DP alignment with split penalties — handles OP removal / ad breaks /
piecewise drift, not just a global offset). Same binary, same argument order
as `crates/mt-fetch/src/align_ilass.rs`, so behaviour matches the Rust path.

ilass is language-agnostic (it aligns on subtitle *timing/activity*, not
text), so aligning Polish OCR against an English reference works fine.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .contracts import HardsubError

logger = logging.getLogger(__name__)

# Same default as mt-fetch (align_ilass.rs).
_SPLIT_PENALTY = '7.0'


def _ilass_binary() -> Path:
    """Locate the vendored ilass binary (built by `just build`)."""
    # scripts/hardsub_poc/align.py -> repo root is two parents up.
    root = Path(__file__).resolve().parents[2]
    return root / 'vendor' / 'ilass' / 'target' / 'release' / 'ilass'


def build_ilass_argv(
    binary: Path, reference_srt: Path, subtitle_srt: Path, output_srt: Path
) -> list[str]:
    """Build the ilass argv (mirrors mt-fetch's build_ilass_argv, same order)."""
    return [
        str(binary),
        str(reference_srt),  # reference (on the target timeline)
        str(subtitle_srt),  # the subtitle to be aligned
        str(output_srt),  # output
        '--split-penalty',
        _SPLIT_PENALTY,
        '--disable-fps-guessing',
    ]


def align_to_reference(subtitle_srt: Path, reference_srt: Path, output_srt: Path) -> Path:
    """Align `subtitle_srt` to `reference_srt`'s timeline via ilass -> `output_srt`.

    Returns the written `output_srt`. Raises `HardsubError` if the ilass binary
    is missing (run `just build`) or the alignment fails — unlike the main
    pipeline we do NOT silently fall back to a global-offset xcorr here; the
    PoC surfaces the failure so it's visible.
    """
    binary = _ilass_binary()
    if not binary.exists():
        raise HardsubError(
            f'ilass binary not found at {binary} — run `just submodules && just build` first.'
        )
    output_srt = Path(output_srt)
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    argv = build_ilass_argv(binary, Path(reference_srt), Path(subtitle_srt), output_srt)
    logger.info('Aligning to reference via ilass: %s', ' '.join(argv))
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HardsubError(f'ilass failed to run: {exc}') from exc
    if proc.returncode != 0:
        raise HardsubError(
            f'ilass exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}'
        )
    if not output_srt.exists() or output_srt.stat().st_size == 0:
        raise HardsubError(f'ilass produced no output at {output_srt}')
    return output_srt


def align_in_place(subtitle_srt: Path, reference_srt: Path) -> Path:
    """Align `subtitle_srt` to `reference_srt`, overwriting `subtitle_srt`."""
    subtitle_srt = Path(subtitle_srt)
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
        tmp_out = Path(tmp.name)
    try:
        align_to_reference(subtitle_srt, reference_srt, tmp_out)
        shutil.move(str(tmp_out), str(subtitle_srt))
    finally:
        tmp_out.unlink(missing_ok=True)
    return subtitle_srt
