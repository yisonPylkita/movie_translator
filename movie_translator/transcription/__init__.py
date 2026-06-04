"""Audio→subtitle transcription (ASR) — sources English dialogue from the
audio track when no subtitle text exists.

Two engines (see benchmarks/asr/REPORT.md for the bake-off that picked them):

- ``apple``  — Apple SpeechAnalyzer (macOS 26+, on-device, fastest). Coarse
  utterances are sentence-split into subtitle-sized lines.
- ``whisper`` — mlx-whisper large-v3 (Metal). Output is post-filtered against
  ED/music hallucination.

`transcribe_to_srt` returns None (skip, not fail) when the requested audio
track doesn't exist, the engine is unavailable, or no usable lines came out.
"""

from __future__ import annotations

from pathlib import Path

from ..logging import logger
from ..srt import write_srt
from . import apple_backend, audio, postfilter, splitter, whisper_backend

ENGINES = ('apple', 'whisper')


def is_available(engine: str) -> bool:
    if engine == 'apple':
        return apple_backend.is_available()
    if engine == 'whisper':
        return whisper_backend.is_available()
    return False


def transcribe_to_srt(
    video_path: Path,
    output_dir: Path,
    language: str = 'en',
    engine: str = 'apple',
) -> Path | None:
    """Transcribe `video_path`'s `language` audio track to an SRT, or None."""
    if engine not in ENGINES:
        raise ValueError(f'unknown transcription engine {engine!r} (use {ENGINES})')
    if not is_available(engine):
        logger.warning(f'transcription engine {engine!r} unavailable on this system')
        return None

    stream = audio.find_audio_stream(video_path, language)
    if stream is None:
        logger.info(f'no {language!r} audio track in {video_path.name}; skipping transcription')
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    wav = output_dir / f'transcribe_{language}.wav'
    audio.extract_wav(video_path, stream, wav)

    try:
        if engine == 'apple':
            raw = apple_backend.transcribe(wav, language)
            lines = splitter.split_segments(raw)
        else:
            raw = whisper_backend.transcribe(wav, language)
            # Clamp against the real audio length (Whisper hallucinates past it).
            lines = postfilter.clean_segments(raw, audio.wav_duration_ms(wav))
    finally:
        wav.unlink(missing_ok=True)

    if not lines:
        logger.warning(f'{engine} transcription produced no usable lines')
        return None

    srt_path = output_dir / f'transcribed_{language}.srt'
    write_srt(lines, srt_path)
    logger.info(f'{engine} transcription: {len(lines)} lines -> {srt_path.name}')
    return srt_path


__all__ = ['ENGINES', 'is_available', 'transcribe_to_srt']
