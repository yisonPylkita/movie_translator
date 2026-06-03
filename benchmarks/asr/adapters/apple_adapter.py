"""Apple SpeechAnalyzer (macOS 26) via the compiled Swift bridge `envs/apple_speech`.

On-device (ANE), no model download/management. `model` argv is ignored (the OS
owns the model). Runs in any python — only stdlib + the swift binary.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import _common

BIN = Path(__file__).parent.parent / 'envs' / 'apple_speech'
LOCALE = {'en': 'en-US', 'ja': 'ja-JP'}


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    proc = subprocess.run([str(BIN), wav, LOCALE[lang]], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'apple_speech failed: {proc.stderr.strip()[:300]}')
    return json.loads(proc.stdout)['segments']


if __name__ == '__main__':
    raise SystemExit(_common.run('apple-speech', transcribe))
