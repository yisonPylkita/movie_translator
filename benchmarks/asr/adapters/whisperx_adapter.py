"""WhisperX: faster-whisper backend (CPU int8) + wav2vec2 forced alignment."""

from __future__ import annotations

import _common


def transcribe(wav: str, lang: str, model: str) -> list[dict]:
    import whisperx

    device = 'cpu'
    m = whisperx.load_model(model, device, compute_type='int8', language=lang)
    audio = whisperx.load_audio(wav)
    res = m.transcribe(audio, language=lang)
    segs = res['segments']
    # Forced alignment sharpens word/segment timestamps. Best-effort: if no
    # align model exists for the language, keep whisper's own timing.
    try:
        amodel, meta = whisperx.load_align_model(language_code=lang, device=device)
        aligned = whisperx.align(segs, amodel, meta, audio, device, return_char_alignments=False)
        segs = aligned['segments']
    except Exception as exc:
        print(f'whisperx align skipped ({lang}): {exc}')
    out = []
    for s in segs:
        if s.get('start') is None or s.get('end') is None:
            continue
        out.append(
            {
                'start_ms': int(s['start'] * 1000),
                'end_ms': int(s['end'] * 1000),
                'text': s['text'].strip(),
            }
        )
    return out


if __name__ == '__main__':
    raise SystemExit(_common.run('whisperx', transcribe))
