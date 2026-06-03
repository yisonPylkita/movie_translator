"""Extract ASR inputs + reference subs from the Isekai Ojisan test episode.

Pulls the Japanese (track 1) and English-dub (track 2) audio to 16 kHz mono
wav (what every Whisper-family engine wants), plus a short iteration segment,
and the English + Polish reference subtitle tracks. System ffmpeg only — no
project venv needed.

Usage: python3 prep_audio.py [/path/to/episode.mkv]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
AUDIO = HERE / 'audio'
REFS = HERE / 'refs'

DEFAULT_EP = Path(
    '/Users/w/Downloads/Torrents/completed/'
    '[Judas] Isekai Ojisan (Uncle from Another World) (Season 01) '
    '[BD 1080p][HEVC x265 10bit][Dual-Audio][Eng-Subs]/'
    '[Judas] Isekai Ojisan - S01E01.mkv'
)

# Iteration segment: 180 s starting at 2:00 (past the OP, into dialogue).
SEG_START = '00:02:00'
SEG_DUR = '180'

# (output stem, ffmpeg -map spec) for the two audio tracks.
AUDIO_TRACKS = [('ja_full', '0:1'), ('en_full', '0:2')]
# (output name, -map spec, codec) for reference subtitle tracks.
SUB_TRACKS = [
    ('en_ref.ass', '0:3', 'copy'),
    ('pl_allegro.ass', '0:5', 'copy'),
]


def _run(args: list[str]) -> None:
    print('+ ' + ' '.join(args))
    subprocess.run(args, check=True, capture_output=True, text=True)


def extract_audio(ep: Path) -> None:
    for stem, mapspec in AUDIO_TRACKS:
        full = AUDIO / f'{stem}.wav'
        _run(
            [
                'ffmpeg',
                '-y',
                '-i',
                str(ep),
                '-map',
                mapspec,
                '-ac',
                '1',
                '-ar',
                '16000',
                '-c:a',
                'pcm_s16le',
                str(full),
            ]
        )
        seg = AUDIO / f'{stem.replace("_full", "_seg")}.wav'
        _run(
            [
                'ffmpeg',
                '-y',
                '-ss',
                SEG_START,
                '-t',
                SEG_DUR,
                '-i',
                str(ep),
                '-map',
                mapspec,
                '-ac',
                '1',
                '-ar',
                '16000',
                '-c:a',
                'pcm_s16le',
                str(seg),
            ]
        )


def extract_subs(ep: Path) -> None:
    for name, mapspec, codec in SUB_TRACKS:
        _run(['ffmpeg', '-y', '-i', str(ep), '-map', mapspec, '-c', codec, str(REFS / name)])


def main() -> int:
    ep = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EP
    if not ep.is_file():
        print(f'episode not found: {ep}', file=sys.stderr)
        return 1
    AUDIO.mkdir(exist_ok=True)
    REFS.mkdir(exist_ok=True)
    extract_audio(ep)
    extract_subs(ep)
    print('\nprepared:')
    for p in sorted(AUDIO.glob('*.wav')) + sorted(REFS.glob('*')):
        print(f'  {p.relative_to(HERE)}  ({p.stat().st_size // 1024} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
