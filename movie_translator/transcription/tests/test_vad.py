"""Tests for VAD-based silence trimming and pause detection.

We generate synthetic WAVs with known silence/speech patterns and verify the
VAD wrapper behaves correctly.
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path

from movie_translator.transcription import vad

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_wav(
    chunks: list[tuple[str, int]],  # ('silence'|'tone', duration_ms)
    sample_rate: int = 16000,
    global_phase: float = 0.0,
) -> bytes:
    """Build a WAV from labelled chunks. 'tone' = 200 Hz sine at -3 dBFS.

    Uses non-zero global phase so the first sample is not at a zero crossing,
    and applies a slight DC offset to ensure webrtcvad can detect it.
    """
    all_samples: list[int] = []
    for kind, dur_ms in chunks:
        n = sample_rate * dur_ms // 1000
        for i in range(n):
            t = i / sample_rate
            if kind == 'tone':
                # Full-amplitude tone with non-zero phase for reliable VAD.
                val = int(0.9 * 32767 * math.sin(2 * math.pi * 200 * t + global_phase))
            else:
                val = 0
            all_samples.append(val)

    import struct

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack('<' + 'h' * len(all_samples), *all_samples))
    return buf.getvalue()


# ── available ───────────────────────────────────────────────────────────────


def test_available_returns_true_when_installed():
    """webrtcvad is installed in the test venv."""
    assert vad.available() is True


# ── trim_silence ────────────────────────────────────────────────────────────


def test_trim_silence_strips_leading_and_trailing(tmp_path: Path):
    """1s silence + 1s tone + 1s silence → ~1s output."""
    wav_bytes = _make_wav(
        [
            ('silence', 1000),
            ('tone', 1000),
            ('silence', 1000),
        ]
    )
    src = tmp_path / 'input.wav'
    src.write_bytes(wav_bytes)
    out = vad.trim_silence(src, out_path=tmp_path / 'trimmed.wav')
    assert out.is_file()
    trimmed_ms = vad._read_frames(out)[2]  # total_ms
    # Should be close to the tone duration (with margin). VAD may detect
    # slightly less due to per-frame granularity at the boundaries.
    assert 700 <= trimmed_ms <= 1300, f'unexpected trimmed length: {trimmed_ms}ms'


def test_trim_silence_all_silence_returns_original(tmp_path: Path):
    """No speech at all → returns original."""
    wav_bytes = _make_wav([('silence', 2000)])
    src = tmp_path / 'input.wav'
    src.write_bytes(wav_bytes)
    out = vad.trim_silence(src)
    assert out.name == 'input_trimmed.wav'
    total_ms = vad._read_frames(out)[2]
    assert total_ms == 2000  # untouched


def test_trim_silence_default_outpath(tmp_path: Path):
    """Default out_path adds '_trimmed' suffix."""
    wav_bytes = _make_wav([('tone', 500)])
    src = tmp_path / 'speech.wav'
    src.write_bytes(wav_bytes)
    out = vad.trim_silence(src)
    assert out == src.with_name('speech_trimmed.wav')


# ── find_pause_boundaries ───────────────────────────────────────────────────


def test_find_pause_boundaries_detects_gap(tmp_path: Path):
    """1s tone + 800ms silence + 1s tone → a boundary at ~1000ms."""
    wav_bytes = _make_wav(
        [
            ('tone', 1000),
            ('silence', 800),
            ('tone', 1000),
        ]
    )
    src = tmp_path / 'pause.wav'
    src.write_bytes(wav_bytes)
    boundaries = vad.find_pause_boundaries(src, min_pause_ms=500)
    assert len(boundaries) > 0
    # Boundary should be near or after 1000ms (end of first tone segment).
    assert any(b >= 900 for b in boundaries), f'no boundary after 900ms: {boundaries}'


def test_find_pause_boundaries_short_gap_ignored(tmp_path: Path):
    """200ms silence is below the 500ms threshold → no boundary."""
    wav_bytes = _make_wav(
        [
            ('tone', 500),
            ('silence', 200),
            ('tone', 500),
        ]
    )
    src = tmp_path / 'short_pause.wav'
    src.write_bytes(wav_bytes)
    boundaries = vad.find_pause_boundaries(src, min_pause_ms=500)
    assert boundaries == []


def test_find_pause_boundaries_multiple_gaps(tmp_path: Path):
    """Detects both 600ms gaps."""
    wav_bytes = _make_wav(
        [
            ('tone', 500),
            ('silence', 700),
            ('tone', 500),
            ('silence', 700),
            ('tone', 500),
        ]
    )
    src = tmp_path / 'multi_pause.wav'
    src.write_bytes(wav_bytes)
    boundaries = vad.find_pause_boundaries(src, min_pause_ms=500)
    assert len(boundaries) >= 2


def test_find_pause_boundaries_no_speech_returns_empty(tmp_path: Path):
    """Pure silence → no boundaries."""
    wav_bytes = _make_wav([('silence', 3000)])
    src = tmp_path / 'nospeech.wav'
    src.write_bytes(wav_bytes)
    boundaries = vad.find_pause_boundaries(src)
    assert boundaries == []
