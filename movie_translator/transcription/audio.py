"""Extract an audio track from a video as ASR-ready wav (16 kHz mono s16)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..ffmpeg import get_ffmpeg, get_video_info
from ..logging import logger

# ISO codes that count as a given two-letter request.
_LANG_ALIASES = {'en': {'en', 'eng'}, 'ja': {'ja', 'jpn', 'jp'}}


def find_audio_stream(video_path: Path, language: str) -> int | None:
    """Index (absolute stream index) of the first audio track tagged `language`.

    Falls back to the first audio track when the file has exactly one audio
    stream and no language tags (common for web rips).
    """
    wanted = _LANG_ALIASES.get(language, {language})
    info = get_video_info(video_path)
    audio = [s for s in info.get('streams', []) if s.get('codec_type') == 'audio']
    for s in audio:
        lang = (s.get('tags') or {}).get('language', '').lower()
        if lang in wanted:
            return int(s['index'])
    if len(audio) == 1 and not (audio[0].get('tags') or {}).get('language'):
        return int(audio[0]['index'])
    return None


def extract_wav(video_path: Path, stream_index: int, out_wav: Path) -> Path:
    """Extract `stream_index` to 16 kHz mono pcm_s16le wav at `out_wav`."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg(),
        '-y',
        '-i',
        str(video_path),
        '-map',
        f'0:{stream_index}',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        str(out_wav),
    ]
    logger.info(f'Extracting audio stream {stream_index} -> {out_wav.name}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size == 0:
        raise RuntimeError(f'ffmpeg audio extraction failed: {result.stderr[-500:]}')
    return out_wav


def wav_duration_ms(wav: Path) -> int:
    import wave

    with wave.open(str(wav), 'rb') as w:
        return int(w.getnframes() * 1000 / w.getframerate())
