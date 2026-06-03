"""whisper.cpp via pywhispercpp (Metal on Apple Silicon). ggml models."""

from __future__ import annotations

import _common


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    from pywhispercpp.model import Model

    m = Model(model, language=lang, print_realtime=False, print_progress=False)
    segments = m.transcribe(wav)
    out = []
    for s in segments:
        # pywhispercpp timestamps are centiseconds (10 ms units).
        out.append({'start_ms': int(s.t0 * 10), 'end_ms': int(s.t1 * 10), 'text': s.text.strip()})
    return out


if __name__ == '__main__':
    raise SystemExit(_common.run('whisper.cpp', transcribe))
