"""Tests for audio stream discovery, extraction, and duration helpers.

Uses ``conftest.py`` fixtures for FFmpeg availability and creates synthetic
WAV files for offline tests where possible. Extraction tests require FFmpeg
and a real video — these are marked ``integration``.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from movie_translator.ffmpeg import get_ffmpeg
from movie_translator.transcription import audio

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_wav_bytes(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    """Generate a silent 16 kHz mono s16le WAV."""
    nframes = sample_rate * duration_ms // 1000
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b'\x00\x00' * nframes)
    return buf.getvalue()


def _make_synthetic_video_info(
    streams: list[dict],
) -> str:
    """Serialise a fake ffprobe JSON blob."""
    return json.dumps({'streams': streams})


# ── find_audio_stream ───────────────────────────────────────────────────────


def test_find_by_language_tag():
    info = _make_synthetic_video_info(
        [
            {'index': 0, 'codec_type': 'audio', 'tags': {'language': 'eng'}},
            {'index': 1, 'codec_type': 'audio', 'tags': {'language': 'jpn'}},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'en') == 0


def test_find_by_iso_alias():
    info = _make_synthetic_video_info(
        [
            {'index': 2, 'codec_type': 'audio', 'tags': {'language': 'en'}},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'en') == 2


def test_fallback_single_unlabeled_stream():
    """Single unlabeled audio track counts as any requested language."""
    info = _make_synthetic_video_info(
        [
            {'index': 0, 'codec_type': 'video'},
            {'index': 1, 'codec_type': 'audio'},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'en') == 1


def test_no_fallback_multiple_unlabeled_streams():
    """Multiple unlabeled streams → don't guess."""
    info = _make_synthetic_video_info(
        [
            {'index': 0, 'codec_type': 'audio'},
            {'index': 1, 'codec_type': 'audio'},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'en') is None


def test_no_audio_stream_returns_none():
    info = _make_synthetic_video_info(
        [
            {'index': 0, 'codec_type': 'video'},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'en') is None


def test_find_ja_language():
    info = _make_synthetic_video_info(
        [
            {'index': 0, 'codec_type': 'audio', 'tags': {'language': 'jpn'}},
        ]
    )
    with patch(
        'movie_translator.transcription.audio.get_video_info', return_value=json.loads(info)
    ):
        assert audio.find_audio_stream(Path('/fake.mkv'), 'ja') == 0


# ── wav_duration_ms ─────────────────────────────────────────────────────────


def test_wav_duration_ms(tmp_path: Path):
    wav = tmp_path / 'test.wav'
    wav.write_bytes(_make_wav_bytes(duration_ms=1234, sample_rate=16000))
    assert audio.wav_duration_ms(wav) == 1234


def test_wav_duration_ms_different_rate(tmp_path: Path):
    wav = tmp_path / 'test.wav'
    wav.write_bytes(_make_wav_bytes(duration_ms=2000, sample_rate=48000))
    assert audio.wav_duration_ms(wav) == 2000


# ── extract_wav (integration, needs ffmpeg + a real video) ──────────────────


@pytest.mark.integration
def test_extract_wav(tmp_path: Path):
    """Requires FFmpeg and a real video with an audio track."""
    ffmpeg = get_ffmpeg()
    # Create a minimal valid video with an audio stream using FFmpeg.
    video = tmp_path / 'source.mkv'
    subprocess.run(
        [
            ffmpeg,
            '-f',
            'lavfi',
            '-i',
            'sine=frequency=440:duration=1',  # 1s tone
            '-f',
            'lavfi',
            '-i',
            'color=c=black:s=64x64:d=1',  # 1s black video
            '-shortest',
            '-c:a',
            'aac',  # encode as AAC for MKV
            '-y',
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    out_wav = tmp_path / 'out.wav'
    result = audio.extract_wav(video, 1, out_wav)  # stream 1 = audio (0 = video)
    assert result == out_wav
    assert out_wav.is_file()
    assert out_wav.stat().st_size > 0
    # Should be ~1 second of 16 kHz mono s16le.
    assert 15000 < out_wav.stat().st_size < 50000  # vary by ffmpeg version
