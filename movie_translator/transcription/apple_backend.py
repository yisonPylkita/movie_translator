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
import subprocess
from pathlib import Path

from ..logging import logger
from ..swift_bridge import ensure_compiled, macos_at_least
from ..types import DialogueLine

_SWIFT_DIR = Path(__file__).parent / 'swift'
_SWIFT_SOURCE = _SWIFT_DIR / 'transcribe_bridge.swift'
_SWIFT_BINARY = _SWIFT_DIR / 'transcribe_bridge'

_LOCALES = {'en': 'en-US', 'ja': 'ja-JP'}


def is_available() -> bool:
    """macOS 26+ with the bridge source present (compiler checked at build)."""
    return macos_at_least(26) and _SWIFT_SOURCE.exists()


def _ensure_binary() -> Path:
    return ensure_compiled(_SWIFT_SOURCE, _SWIFT_BINARY, timeout=120)


def _parse_segments(payload: str) -> list[DialogueLine]:
    """Bridge JSON -> DialogueLines. Drops empties and non-positive durations
    (a result whose runs carry no audioTimeRange comes through as 0/0)."""
    out: list[DialogueLine] = []
    for s in json.loads(payload)['segments']:
        text = s['text'].strip()
        start_ms, end_ms = int(s['start_ms']), int(s['end_ms'])
        if text and end_ms > start_ms:
            out.append(DialogueLine(start_ms, end_ms, text))
    return out


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
    return _parse_segments(proc.stdout)
