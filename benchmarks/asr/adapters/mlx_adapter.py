"""mlx-whisper (Apple MLX / Metal). Apple-Silicon-native."""

from __future__ import annotations

import _common

REPO = {
    'small': 'mlx-community/whisper-small-mlx',
    'large-v3': 'mlx-community/whisper-large-v3-mlx',
}


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    import mlx_whisper

    r = mlx_whisper.transcribe(wav, path_or_hf_repo=REPO[model], language=lang)
    return [
        {
            'start_ms': int(s['start'] * 1000),
            'end_ms': int(s['end'] * 1000),
            'text': s['text'].strip(),
        }
        for s in r['segments']
    ]


if __name__ == '__main__':
    raise SystemExit(_common.run('mlx-whisper', transcribe))
