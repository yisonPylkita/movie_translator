"""Voice Activity Detection for ASR pre-processing.

Wraps ``webrtcvad`` (lightweight C extension, zero ML deps) to provide:

- ``trim_silence(wav_path, margin_ms)`` — strip leading/trailing silence from a
  WAV file, writing a trimmed copy alongside the original. Reduces Whisper's
  tendency to hallucinate on trailing music/silence.
- ``find_pause_boundaries(wav_path, min_pause_ms)`` — return timestamps (ms) of
  silence gaps ≥ ``min_pause_ms``. Useful for splitting Apple SpeechAnalyzer's
  coarse utterances into subtitle-sized chunks at natural pause boundaries.

webrtcvad operates on 10/20/30 ms frames at 16 kHz mono s16le — our exact
ASR extraction format. Aggressiveness mode 2 (moderately aggressive) is the
default: it catches trailing silence in credit rolls without falsely clipping
quiet dialogue.
"""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

from ..logging import logger

_HAS_WEBRTCVAD = importlib.util.find_spec('webrtcvad') is not None

# ── Constants ──────────────────────────────────────────────────────────────

# webrtcvad frame size (ms). 30 ms = 480 samples @ 16 kHz — good balance of
# granularity vs throughput.
_FRAME_MS = 30
_FRAME_BYTES = 480 * 2  # 16-bit samples @ 16 kHz * 30 ms
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # s16le

# Defaults exposed for the pipeline but tweakable via keyword args.
_DEFAULT_AGGRESSIVENESS = 2  # 0=loosest … 3=tightest
_DEFAULT_MIN_PAUSE_MS = 500  # Pause ≥ 500 ms → utterance boundary
_DEFAULT_MARGIN_MS = 50  # Padding around speech regions

__all__ = ['available', 'trim_silence', 'find_pause_boundaries']

# ── Public API ─────────────────────────────────────────────────────────────


def available() -> bool:
    """True when webrtcvad is installed and importable."""
    return _HAS_WEBRTCVAD


def trim_silence(
    wav_path: Path,
    out_path: Path | None = None,
    aggressiveness: int = _DEFAULT_AGGRESSIVENESS,
    margin_ms: int = _DEFAULT_MARGIN_MS,
) -> Path:
    """Strip leading/trailing silence from ``wav_path``.

    Writes the trimmed audio to ``out_path`` (or ``wav_path`` with
    ``_trimmed`` suffix when ``None``). Returns the output path.

    Raises ``RuntimeError`` when webrtcvad is not installed.
    """
    _require_vad()
    audio, frames, total_ms = _read_frames(wav_path)
    speech_mask = _compute_speech_mask(audio, aggressiveness)

    # Find first and last speech frame indexes.
    try:
        first = next(i for i, v in enumerate(speech_mask) if v)
        last = len(speech_mask) - 1 - next(i for i, v in enumerate(reversed(speech_mask)) if v)
    except StopIteration:
        logger.warning(f'VAD: no speech detected in {wav_path.name} — returning original')
        out = out_path or wav_path.with_name(wav_path.stem + '_trimmed' + wav_path.suffix)
        if out != wav_path:
            _copy_wav(wav_path, out)
        return out

    # Convert frame indexes to byte offsets with margin.
    margin_bytes = margin_ms * _SAMPLE_RATE * _SAMPLE_WIDTH // 1000
    start_byte = max(0, first * _FRAME_BYTES - margin_bytes)
    end_byte = min(len(audio), (last + 1) * _FRAME_BYTES + margin_bytes)

    trimmed = audio[start_byte:end_byte]
    out = out_path or wav_path.with_name(wav_path.stem + '_trimmed' + wav_path.suffix)
    _write_wav(trimmed, out)
    trimmed_ms = len(trimmed) * 1000 // _SAMPLE_RATE
    logger.info(
        f'VAD trim: {wav_path.name} {total_ms}ms → {trimmed_ms}ms'
        f' ({100 * trimmed_ms // total_ms}% kept)'
    )
    return out


def find_pause_boundaries(
    wav_path: Path,
    min_pause_ms: int = _DEFAULT_MIN_PAUSE_MS,
    aggressiveness: int = _DEFAULT_AGGRESSIVENESS,
) -> list[int]:
    """Return millisecond timestamps of silence gaps ≥ ``min_pause_ms``.

    Each boundary marks the **start** of a silence gap (the end of the
    preceding speech region). Callers can split at these positions for
    utterance-aligned segmentation.

    Returns an empty list when webrtcvad is unavailable or no long enough
    pause is found.
    """
    if not _HAS_WEBRTCVAD:
        return []
    audio, frames, _total_ms = _read_frames(wav_path)
    speech_mask = _compute_speech_mask(audio, aggressiveness)

    # Find runs of consecutive non-speech frames.
    boundaries: list[int] = []
    silence_start: int | None = None
    pause_frames = min_pause_ms // _FRAME_MS

    for i, is_speech in enumerate(speech_mask):
        if not is_speech and silence_start is None:
            silence_start = i
        elif is_speech and silence_start is not None:
            if i - silence_start >= pause_frames:
                # Boundary at the start of the silence gap.
                boundaries.append(silence_start * _FRAME_MS)
            silence_start = None

    return boundaries


# ── Internal helpers ───────────────────────────────────────────────────────


def _require_vad() -> None:
    if not _HAS_WEBRTCVAD:
        raise RuntimeError(
            'webrtcvad is required for VAD-based audio processing. '
            'Install it with: uv pip install webrtcvad'
        )


def _read_frames(wav_path: Path) -> tuple[bytes, int, int]:
    """Read a 16 kHz mono s16le WAV, return (raw_audio, frame_count, duration_ms)."""
    with wave.open(str(wav_path), 'rb') as w:
        assert w.getnchannels() == 1, 'VAD requires mono audio'
        assert w.getsampwidth() == _SAMPLE_WIDTH, 'VAD requires s16le'
        assert w.getframerate() == _SAMPLE_RATE, 'VAD requires 16 kHz'
        audio = w.readframes(w.getnframes())
    total_ms = len(audio) * 1000 // (_SAMPLE_RATE * _SAMPLE_WIDTH)
    frames = len(audio) // _FRAME_BYTES
    return audio, frames, total_ms


def _compute_speech_mask(audio: bytes, aggressiveness: int) -> list[bool]:
    """Run webrtcvad on every frame, return list of ``True`` (speech) per frame."""
    _require_vad()
    import webrtcvad  # noqa: F811 — guarded by _require_vad above

    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)
    mask: list[bool] = []
    for offset in range(0, len(audio) - _FRAME_BYTES + 1, _FRAME_BYTES):
        frame = audio[offset : offset + _FRAME_BYTES]
        mask.append(vad.is_speech(frame, _SAMPLE_RATE))
    return mask


def _write_wav(samples: bytes, out_path: Path) -> None:
    """Write raw s16le mono PCM data to a WAV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(_SAMPLE_WIDTH)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(samples)


def _copy_wav(src: Path, dst: Path) -> None:
    """Copy a WAV file (plain file copy)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
