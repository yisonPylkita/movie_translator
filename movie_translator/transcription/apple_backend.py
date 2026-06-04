"""Apple SpeechAnalyzer backend (macOS 26+) via a compiled Swift bridge.

On-device ANE transcription: fastest engine in the bake-off (RTF ~0.01 on a
full episode), no model files, native endpointing (no Whisper-style ED
hallucination). Emits coarse multi-sentence utterances — callers run
`splitter.split_segments` for subtitle-sized lines. Bridge is compiled from
`swift/transcribe_bridge.swift` on first use, mirroring the Apple Translation
bridge.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine

_SWIFT_DIR = Path(__file__).parent / 'swift'
_SWIFT_SOURCE = _SWIFT_DIR / 'transcribe_bridge.swift'
_SWIFT_BINARY = _SWIFT_DIR / 'transcribe_bridge'

_LOCALES = {'en': 'en-US', 'ja': 'ja-JP'}


def is_available() -> bool:
    """macOS 26+ with the bridge source present (compiler checked at build)."""
    if platform.system() != 'Darwin':
        return False
    mac_ver = platform.mac_ver()[0]
    try:
        if int(mac_ver.split('.')[0]) < 26:
            return False
    except ValueError, IndexError:
        return False
    return _SWIFT_SOURCE.exists()


def _ensure_binary() -> Path:
    """Compile the Swift bridge if missing or the source is newer."""
    if not _SWIFT_SOURCE.exists():
        raise FileNotFoundError(f'Swift bridge source not found: {_SWIFT_SOURCE}')
    needs_compile = (
        not _SWIFT_BINARY.exists() or _SWIFT_SOURCE.stat().st_mtime > _SWIFT_BINARY.stat().st_mtime
    )
    if needs_compile:
        logger.info('Compiling SpeechAnalyzer transcription bridge...')
        swiftc = shutil.which('swiftc')
        if swiftc is None:
            raise FileNotFoundError(
                'Swift compiler (swiftc) not found. Install Command Line Tools: xcode-select --install'
            )
        result = subprocess.run(
            [swiftc, '-O', str(_SWIFT_SOURCE), '-o', str(_SWIFT_BINARY)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f'Swift bridge compilation failed:\n{result.stderr[-1000:]}')
    return _SWIFT_BINARY


def transcribe(wav: Path, language: str) -> list[DialogueLine]:
    """Transcribe `wav` -> coarse utterances. Raises on bridge failure."""
    locale = _LOCALES.get(language)
    if locale is None:
        raise ValueError(f'apple backend: unsupported language {language!r}')
    binary = _ensure_binary()
    logger.info(f'SpeechAnalyzer transcribing {wav.name} ({locale})')
    proc = subprocess.run([str(binary), str(wav), locale], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'transcribe_bridge failed: {proc.stderr.strip()[-500:]}')
    segments = json.loads(proc.stdout)['segments']
    return [
        DialogueLine(int(s['start_ms']), int(s['end_ms']), s['text'].strip())
        for s in segments
        if s['text'].strip()
    ]
