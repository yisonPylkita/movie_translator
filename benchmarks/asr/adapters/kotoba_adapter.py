"""kotoba-whisper v2.0 (Japanese-specialized distil-Whisper) via transformers.

JP only. Tries MPS, falls back to CPU. `model` argv is ignored (fixed model).
"""

from __future__ import annotations

import _common

MODEL_ID = 'kotoba-tech/kotoba-whisper-v2.0'


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    import torch
    from transformers import pipeline

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32

    def build(dev, dt):
        return pipeline(
            'automatic-speech-recognition',
            model=MODEL_ID,
            torch_dtype=dt,
            device=dev,
            chunk_length_s=15,
            batch_size=8,
        )

    gen = {'language': 'ja', 'task': 'transcribe'}
    try:
        pipe = build(device, dtype)
        r = pipe(wav, return_timestamps=True, generate_kwargs=gen)
    except Exception:
        pipe = build('cpu', torch.float32)
        r = pipe(wav, return_timestamps=True, generate_kwargs=gen)

    out = []
    for c in r['chunks']:
        ts = c['timestamp']
        if ts[0] is None:
            continue
        end = ts[1] if ts[1] is not None else ts[0]
        out.append(
            {'start_ms': int(ts[0] * 1000), 'end_ms': int(end * 1000), 'text': c['text'].strip()}
        )
    return out


if __name__ == '__main__':
    raise SystemExit(_common.run('kotoba-whisper', transcribe))
