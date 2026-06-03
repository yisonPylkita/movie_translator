"""openai-whisper (PyTorch). Tries MPS, falls back to CPU if MPS ops fail."""

from __future__ import annotations

import _common


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    import torch
    import whisper

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = whisper.load_model(model, device=device)
    try:
        r = m.transcribe(wav, language=lang, fp16=False, verbose=False)
    except Exception:
        # Some whisper ops are unimplemented on MPS — retry on CPU.
        m = whisper.load_model(model, device='cpu')
        r = m.transcribe(wav, language=lang, fp16=False, verbose=False)
    return [
        {
            'start_ms': int(s['start'] * 1000),
            'end_ms': int(s['end'] * 1000),
            'text': s['text'].strip(),
        }
        for s in r['segments']
    ]


if __name__ == '__main__':
    raise SystemExit(_common.run('openai-whisper', transcribe))
