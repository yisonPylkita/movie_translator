#!/usr/bin/env python3
"""Gather subtitle data for alignment analysis.

Recursively scans directories for MKV files, extracts English subtitle
tracks, downloads Polish subtitle candidates, and produces a single JSON
artifact containing per-event structural metadata for every subtitle file
encountered. The output is designed to be consumed by analysis scripts
without re-downloading or re-extracting anything.

Usage:
    python scripts/gather_subtitle_data.py /path/to/anime [/more/paths ...] -o output.json

The script is idempotent — re-running with the same output file skips
episodes that have already been processed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to path so we can import movie_translator modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pysubs2  # noqa: E402

from movie_translator.identifier.identify import identify_media  # noqa: E402
from movie_translator.subtitle_fetch import SubtitleFetcher  # noqa: E402
from movie_translator.subtitle_fetch.providers.animesub import AnimeSubProvider  # noqa: E402
from movie_translator.subtitle_fetch.providers.napiprojekt import NapiProjektProvider  # noqa: E402
from movie_translator.subtitle_fetch.providers.podnapisi import PodnapisiProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Subtitle extraction from MKV
# ---------------------------------------------------------------------------


def get_subtitle_streams(video_path: Path) -> list[dict]:
    """Get metadata for all subtitle streams in a video file."""
    result = subprocess.run(
        [
            'ffprobe',
            '-v',
            'quiet',
            '-print_format',
            'json',
            '-show_streams',
            '-select_streams',
            's',
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return data.get('streams', [])


def extract_subtitle_track(video_path: Path, stream_index: int, output_path: Path) -> bool:
    """Extract a single subtitle track from a video file."""
    result = subprocess.run(
        [
            'ffmpeg',
            '-v',
            'quiet',
            '-y',
            '-i',
            str(video_path),
            '-map',
            f'0:{stream_index}',
            str(output_path),
        ],
        capture_output=True,
    )
    return result.returncode == 0


def find_english_subtitle_stream(streams: list[dict]) -> dict | None:
    """Find the best English subtitle stream (text-based, dialogue-focused)."""
    eng_streams = []
    for s in streams:
        lang = s.get('tags', {}).get('language', '').lower()
        codec = s.get('codec_name', '')
        if lang in ('eng', 'en', 'english') and codec in ('ass', 'subrip', 'srt'):
            eng_streams.append(s)

    if not eng_streams:
        return None

    # Prefer the first one (usually the main dialogue track)
    return eng_streams[0]


# ---------------------------------------------------------------------------
# ASS event analysis
# ---------------------------------------------------------------------------


def analyze_ass_event(event: pysubs2.SSAEvent) -> dict:
    """Extract structural features from a single ASS event."""
    text = event.text or ''
    plaintext = event.plaintext.strip() if hasattr(event, 'plaintext') else text

    return {
        'start_ms': event.start,
        'end_ms': event.end,
        'duration_ms': event.end - event.start,
        'style': event.style,
        'layer': event.layer,
        'effect': event.effect or '',
        'name': getattr(event, 'name', '') or '',
        'text_length': len(plaintext),
        'plaintext_preview': plaintext[:80],
        'is_comment': event.type == 'Comment',
        'is_drawing': event.is_drawing if hasattr(event, 'is_drawing') else False,
        'has_pos': '\\pos(' in text,
        'has_move': '\\move(' in text,
        'has_clip': '\\clip(' in text or '\\iclip(' in text,
        'has_karaoke': any(t in text for t in ('\\k', '\\K', '\\kf', '\\ko')),
        'has_an_override': '\\an' in text,
        'margin_l': event.marginl,
        'margin_r': event.marginr,
        'margin_v': event.marginv,
    }


def analyze_subtitle_file(path: Path, include_events: bool = True) -> dict:
    """Analyze a subtitle file and return per-event and per-style metadata.

    Args:
        path: Path to the subtitle file.
        include_events: If False, omit per-event data to save space.
    """
    try:
        subs = pysubs2.load(str(path))
    except Exception as e:
        return {'error': str(e), 'events': [], 'styles': {}}

    events = []
    for event in subs:
        if not event.text or not event.text.strip():
            continue
        events.append(analyze_ass_event(event))

    # Aggregate per-style statistics
    style_stats = {}
    for evt in events:
        style = evt['style']
        if style not in style_stats:
            style_stats[style] = {
                'count': 0,
                'total_duration_ms': 0,
                'total_text_length': 0,
                'positioned_count': 0,
                'karaoke_count': 0,
                'drawing_count': 0,
                'comment_count': 0,
                'has_clip_count': 0,
            }
        s = style_stats[style]
        s['count'] += 1
        s['total_duration_ms'] += evt['duration_ms']
        s['total_text_length'] += evt['text_length']
        s['positioned_count'] += int(evt['has_pos'] or evt['has_move'])
        s['karaoke_count'] += int(evt['has_karaoke'])
        s['drawing_count'] += int(evt['is_drawing'])
        s['comment_count'] += int(evt['is_comment'])
        s['has_clip_count'] += int(evt['has_clip'])

    # Compute derived stats
    for _style, s in style_stats.items():
        n = s['count']
        s['avg_duration_ms'] = s['total_duration_ms'] / n if n else 0
        s['avg_text_length'] = s['total_text_length'] / n if n else 0
        s['positioned_ratio'] = s['positioned_count'] / n if n else 0
        s['karaoke_ratio'] = s['karaoke_count'] / n if n else 0

    # Extract style definitions from the ASS header
    style_defs = {}
    if hasattr(subs, 'styles'):
        for name, style_obj in subs.styles.items():
            style_defs[name] = {
                'fontname': style_obj.fontname,
                'fontsize': style_obj.fontsize,
                'alignment': style_obj.alignment,
                'bold': style_obj.bold,
                'italic': style_obj.italic,
                'margin_l': style_obj.marginl,
                'margin_r': style_obj.marginr,
                'margin_v': style_obj.marginv,
            }

    result = {
        'path': str(path),
        'format': path.suffix.lstrip('.'),
        'event_count': len(events),
        'style_count': len(style_stats),
        'styles': style_stats,
        'style_definitions': style_defs,
    }
    if include_events:
        result['events'] = events
    return result


# ---------------------------------------------------------------------------
# Polish subtitle downloading
# ---------------------------------------------------------------------------


def download_polish_candidates(
    video_path: Path,
    work_dir: Path,
    include_events: bool = True,
) -> list[dict]:
    """Search and download Polish subtitle candidates for a video."""
    try:
        identity = identify_media(video_path)
    except Exception as e:
        return [{'error': f'identify failed: {e}'}]

    providers = [AnimeSubProvider(), PodnapisiProvider()]
    napi = NapiProjektProvider()
    napi.set_video_path(video_path)
    providers.append(napi)

    fetcher = SubtitleFetcher(providers)

    try:
        matches = fetcher.search_all(identity, ['pol'])
    except Exception as e:
        return [{'error': f'search failed: {e}'}]

    candidates = []
    for i, match in enumerate(matches):
        filename = f'{match.source}_{match.language}_{i}.{match.format}'
        output_path = work_dir / filename
        candidate_info = {
            'source': match.source,
            'subtitle_id': match.subtitle_id,
            'release_name': match.release_name,
            'format': match.format,
            'score': match.score,
            'hash_match': match.hash_match,
        }

        try:
            fetcher.download_candidate(match, output_path)
            analysis = analyze_subtitle_file(output_path, include_events=include_events)
            candidate_info['analysis'] = analysis
        except Exception as e:
            candidate_info['error'] = str(e)

        candidates.append(candidate_info)

    return candidates


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_video(video_path: Path, work_dir: Path, include_events: bool = True) -> dict:
    """Process a single video file: extract English subs, download Polish candidates."""
    print(f'  Processing: {video_path.name}')

    result = {
        'video_path': str(video_path),
        'video_name': video_path.name,
        'video_size_bytes': video_path.stat().st_size,
    }

    # Identify media
    try:
        identity = identify_media(video_path)
        result['identity'] = {
            'title': identity.title,
            'season': identity.season,
            'episode': identity.episode,
            'year': identity.year,
        }
    except Exception as e:
        result['identity'] = {'error': str(e)}

    # Get subtitle streams
    streams = get_subtitle_streams(video_path)
    result['subtitle_streams'] = [
        {
            'index': s['index'],
            'codec': s.get('codec_name', ''),
            'language': s.get('tags', {}).get('language', ''),
            'title': s.get('tags', {}).get('title', ''),
        }
        for s in streams
    ]

    # Extract and analyze English subtitle
    eng_stream = find_english_subtitle_stream(streams)
    if eng_stream:
        codec = eng_stream.get('codec_name', 'ass')
        ext = '.ass' if codec == 'ass' else '.srt'
        eng_path = work_dir / f'eng_{video_path.stem}{ext}'

        if extract_subtitle_track(video_path, eng_stream['index'], eng_path):
            result['english_subtitle'] = analyze_subtitle_file(
                eng_path, include_events=include_events
            )
        else:
            result['english_subtitle'] = {'error': 'extraction failed'}
    else:
        result['english_subtitle'] = {'error': 'no English text subtitle found'}

    # Download and analyze Polish candidates
    pol_dir = work_dir / 'polish'
    pol_dir.mkdir(exist_ok=True)
    result['polish_candidates'] = download_polish_candidates(
        video_path, pol_dir, include_events=include_events
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Gather subtitle data for alignment analysis',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='Directories to scan recursively for MKV files, or individual MKV files',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('subtitle_data.json'),
        help='Output JSON file (default: subtitle_data.json)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Max episodes to process (0 = unlimited)',
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Omit per-event data (style-level aggregates only). '
        'Reduces output from ~9MB/episode to ~15KB/episode.',
    )
    args = parser.parse_args()

    # Collect all MKV files
    mkv_files: list[Path] = []
    for p in args.paths:
        if p.is_file() and p.suffix.lower() == '.mkv':
            mkv_files.append(p)
        elif p.is_dir():
            mkv_files.extend(sorted(p.rglob('*.mkv')))
        else:
            print(f'Warning: skipping {p} (not a file or directory)')

    if not mkv_files:
        print('No MKV files found.')
        sys.exit(1)

    if args.limit > 0:
        mkv_files = mkv_files[: args.limit]

    # Load existing output for idempotency
    existing: dict = {}
    if args.output.exists():
        with open(args.output) as f:
            data = json.load(f)
            existing = {ep['video_name']: ep for ep in data.get('episodes', [])}
        print(f'Loaded {len(existing)} existing episodes from {args.output}')

    print(f'Found {len(mkv_files)} MKV files to process')

    episodes = list(existing.values())
    processed_names = set(existing.keys())

    with tempfile.TemporaryDirectory(prefix='subtitle_gather_') as tmp:
        work_base = Path(tmp)

        for i, mkv in enumerate(mkv_files):
            if mkv.name in processed_names:
                print(f'  [{i + 1}/{len(mkv_files)}] Skipping (already processed): {mkv.name}')
                continue

            print(f'  [{i + 1}/{len(mkv_files)}]', end='')
            work_dir = work_base / f'ep_{i}'
            work_dir.mkdir(parents=True)

            try:
                episode = process_video(mkv, work_dir, include_events=not args.compact)
                episodes.append(episode)
            except Exception as e:
                print(f'    ERROR: {e}')
                episodes.append(
                    {
                        'video_path': str(mkv),
                        'video_name': mkv.name,
                        'error': str(e),
                    }
                )

            # Write after each episode for crash resilience
            output = {
                'version': 1,
                'episode_count': len(episodes),
                'episodes': episodes,
            }
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nDone. {len(episodes)} episodes written to {args.output}')


if __name__ == '__main__':
    main()
