"""Compile-on-first-use helper for Swift bridge binaries.

Both Apple backends (Translation, SpeechAnalyzer transcription) ship a Swift
source next to the package and compile it lazily; this is the single home for
that mechanism so toolchain fixes (flags, SDK changes, staleness rules) land
once instead of drifting per-feature.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .logging import logger


def macos_at_least(major: int) -> bool:
    """True when running on macOS with at least the given major version."""
    if platform.system() != 'Darwin':
        return False
    mac_ver = platform.mac_ver()[0]
    try:
        return int(mac_ver.split('.')[0]) >= major
    except ValueError, IndexError:
        return False


def ensure_compiled(
    source: Path,
    binary: Path,
    extra_args: Sequence[str] = (),
    timeout: int = 120,
) -> Path:
    """Compile `source` to `binary` with swiftc if missing or stale (by mtime)."""
    if not source.exists():
        raise FileNotFoundError(f'Swift bridge source not found: {source}')
    fresh = binary.exists() and source.stat().st_mtime <= binary.stat().st_mtime
    if fresh:
        return binary

    swiftc = shutil.which('swiftc')
    if swiftc is None:
        raise FileNotFoundError(
            'Swift compiler (swiftc) not found. Install Command Line Tools: xcode-select --install'
        )
    logger.info(f'Compiling Swift bridge: {source.name}')
    result = subprocess.run(
        [swiftc, '-O', *extra_args, str(source), '-o', str(binary)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'Swift bridge compilation failed ({source.name}):\n{result.stderr[-1000:]}'
        )
    logger.info(f'Compiled: {binary}')
    return binary
