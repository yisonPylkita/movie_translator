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

from collections.abc import Callable
from pathlib import Path

from ..logging import logger
from ..srt import write_srt
from . import apple_backend, audio, postfilter, splitter, vad, whisper_backend

ENGINES = ('apple', 'whisper')


def is_available(engine: str) -> bool:
    if engine == 'apple':
        return apple_backend.is_available()
    if engine == 'whisper':
        return whisper_backend.is_available()
    return False


def _suggest_engine_fallback(engine: str) -> str | None:
    """Suggest an available fallback engine when `engine` is unavailable."""
    for alt in ENGINES:
        if alt != engine and is_available(alt):
            return alt
    return None


def transcribe_to_srt(
    video_path: Path,
    output_dir: Path,
    language: str = 'en',
    engine: str = 'apple',
    progress_callback: Callable[[int], None] | None = None,
) -> Path | None:
    """Transcribe `video_path`'s `language` audio track to an SRT, or None.

    Args:
        progress_callback: Called with ``percent`` (0-100) during transcription
            if the engine supports progress reporting. Not yet wired through
            the Rust PyO3 bridge — added as a hook for future integration.
    """
    if engine not in ENGINES:
        raise ValueError(f'unknown transcription engine {engine!r} (use {ENGINES})')

    if not is_available(engine):
        fallback = _suggest_engine_fallback(engine)
        msg = f'transcription engine {engine!r} unavailable on this system'
        if fallback:
            msg += f'; try --transcribe-engine {fallback}'
            logger.warning(msg)
        else:
            logger.warning(f'{msg} and no fallback engine is available')
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
            if progress_callback:
                progress_callback(50)

            # Use VAD pause boundaries to improve segmentation when available.
            if vad.available():
                pause_boundaries = vad.find_pause_boundaries(wav, min_pause_ms=300)
                if pause_boundaries:
                    lines = splitter.split_segments(raw, pause_boundaries)
                else:
                    lines = splitter.split_segments(raw)
            else:
                lines = splitter.split_segments(raw)

            if progress_callback:
                progress_callback(100)
        else:
            # Trim leading/trailing silence before Whisper to reduce
            # ED/music hallucination, then transcribe.
            if vad.available():
                trimmed = vad.trim_silence(
                    wav,
                    out_path=output_dir / f'transcribe_{language}_trimmed.wav',
                )
                if progress_callback:
                    progress_callback(15)
            else:
                trimmed = wav

            raw = whisper_backend.transcribe(trimmed, language)
            if progress_callback:
                progress_callback(80)
            duration_ms = audio.wav_duration_ms(trimmed)
            lines = postfilter.clean_segments(raw, duration_ms)
            if progress_callback:
                progress_callback(100)

            # Clean up trimmed copy if VAD ran.
            if trimmed != wav:
                trimmed.unlink(missing_ok=True)
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
