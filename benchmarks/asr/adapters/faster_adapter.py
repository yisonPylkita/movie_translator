"""faster-whisper (CTranslate2). No MPS on macOS -> CPU int8."""

from __future__ import annotations

import _common


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device='cpu', compute_type='int8')
    segments, _info = m.transcribe(wav, language=lang)
    return [
        {'start_ms': int(s.start * 1000), 'end_ms': int(s.end * 1000), 'text': s.text.strip()}
        for s in segments
    ]


if __name__ == '__main__':
    raise SystemExit(_common.run('faster-whisper', transcribe))
