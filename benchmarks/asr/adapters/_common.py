"""Shared adapter scaffolding. Each engine adapter runs in its OWN venv and
imports this (same dir, so it's on sys.path[0]). It standardizes timing, peak
RAM, the result-JSON contract, and crash-to-`ok:false` handling so one broken
engine never aborts the bake-off.

Adapter contract (argv): <wav> <lang> <model> <out_json>
A transcribe fn returns list[{'start_ms','end_ms','text'}].
"""

from __future__ import annotations

import json
import resource
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path


def wav_seconds(path: str) -> float:
    with wave.open(path, 'rb') as w:
        return w.getnframes() / float(w.getframerate())


def peak_ram_mb() -> int:
    # macOS ru_maxrss is bytes; Linux is KiB. This PoC is macOS.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))


def run(engine: str, transcribe: Callable[[str, str, str], list[dict]]) -> int:
    """Drive one transcription: argv -> timed run -> result JSON."""
    wav, lang, model, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    audio_s = wav_seconds(wav)
    result = {
        'engine': engine,
        'model': model,
        'lang': lang,
        'wav': Path(wav).name,
        'segments': [],
        'meta': {'audio_s': round(audio_s, 1), 'ok': False, 'error': None},
    }
    t0 = time.perf_counter()
    try:
        segments = transcribe(wav, lang, model)
        infer_s = time.perf_counter() - t0
        result['segments'] = segments
        result['meta'].update(
            ok=True,
            infer_s=round(infer_s, 1),
            rtf=round(infer_s / audio_s, 3) if audio_s else None,
            peak_ram_mb=peak_ram_mb(),
            n_segments=len(segments),
        )
    except Exception as exc:  # noqa: BLE001 — capture any engine failure
        import traceback

        result['meta'].update(
            ok=False,
            error=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc()[-2000:],
            infer_s=round(time.perf_counter() - t0, 1),
        )
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f'{engine}/{model}/{lang}: ok={result["meta"]["ok"]} '
        f'segs={len(result["segments"])} rtf={result["meta"].get("rtf")}'
    )
    return 0 if result['meta']['ok'] else 2
