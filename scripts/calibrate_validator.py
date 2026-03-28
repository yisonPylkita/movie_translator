#!/usr/bin/env python3
"""Calibrate subtitle validator against real anime test data.

For each video file with an embedded English subtitle track:
1. Extract the embedded subtitle as reference
2. Search providers for all candidates
3. Score each candidate against the reference
4. Cross-test: score candidates against OTHER episodes' references

Produces a score matrix to determine optimal threshold and bin size.

Usage:
    python scripts/calibrate_validator.py ~/Downloads/translated
    python scripts/calibrate_validator.py ~/Downloads/translated --bin-sizes 1000,2000,5000
    python scripts/calibrate_validator.py ~/Downloads/translated --max-videos 6 -v
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from movie_translator.identifier import identify_media
from movie_translator.logging import set_verbose
from movie_translator.subtitle_fetch import SubtitleFetcher
from movie_translator.subtitle_fetch.providers.animesub import AnimeSubProvider
from movie_translator.subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from movie_translator.subtitle_fetch.validator import (
    build_density_vector,
    compute_density_correlation,
    extract_timestamps,
)
from movie_translator.subtitles import SubtitleExtractor


def find_videos(input_dir: Path) -> list[Path]:
    videos = []
    for ext in ('*.mkv', '*.mp4'):
        videos.extend(input_dir.rglob(ext))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description='Calibrate subtitle validator')
    parser.add_argument('input_dir', help='Directory with anime video files')
    parser.add_argument(
        '--bin-sizes',
        default='1000,2000,5000,10000',
        help='Bin sizes to test (ms, comma-separated)',
    )
    parser.add_argument('--max-videos', type=int, default=10, help='Max videos to process')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    set_verbose(args.verbose)
    input_dir = Path(args.input_dir)
    windows = [int(x) for x in args.bin_sizes.split(',')]

    videos = find_videos(input_dir)[: args.max_videos]
    print(f'Found {len(videos)} videos (processing up to {args.max_videos})')

    extractor = SubtitleExtractor()
    providers = [AnimeSubProvider()]
    api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
    if api_key:
        providers.append(OpenSubtitlesProvider(api_key=api_key))
    else:
        print('WARNING: OPENSUBTITLES_API_KEY not set, only AnimeSub results available')
    fetcher = SubtitleFetcher(providers)

    # Phase 1: Extract references and collect candidates
    references: dict[str, Path] = {}
    ref_timestamps: dict[str, tuple[list[tuple[int, int]], int]] = {}
    candidates: dict[str, list[tuple[str, Path]]] = {}

    with tempfile.TemporaryDirectory(prefix='calibrate_') as tmpdir:
        tmp = Path(tmpdir)

        for video in videos:
            name = video.stem
            print(f'\n--- {name} ---')

            track_info = extractor.get_track_info(video)
            if not track_info:
                print('  SKIP: no track info')
                continue

            eng_track = extractor.find_english_track(track_info)
            if not eng_track:
                print('  SKIP: no English track')
                continue

            ref_dir = tmp / name
            ref_dir.mkdir(parents=True, exist_ok=True)

            ref_path = ref_dir / f'ref{extractor.get_subtitle_extension(eng_track)}'
            subtitle_index = eng_track.get('subtitle_index', 0)
            try:
                extractor.extract_subtitle(video, eng_track['id'], ref_path, subtitle_index)
            except Exception as e:
                print(f'  SKIP: extraction failed: {e}')
                continue

            ts, dur = extract_timestamps(ref_path)
            if not ts:
                print('  SKIP: no dialogue in reference')
                continue

            references[name] = ref_path
            ref_timestamps[name] = (ts, dur)
            print(f'  Reference: {len(ts)} dialogue events, {dur / 1000:.1f}s')

            # Search for candidates
            try:
                identity = identify_media(video)
                matches = fetcher.search_all(identity, ['eng', 'pol'])
                print(f'  Candidates: {len(matches)} found')

                video_candidates = []
                for i, match in enumerate(matches[:10]):
                    cand_path = ref_dir / f'cand_{i}_{match.source}_{match.language}.{match.format}'
                    try:
                        fetcher.download_candidate(match, cand_path)
                        label = f'{match.release_name} ({match.source}/{match.language})'
                        video_candidates.append((label, cand_path))
                    except Exception as e:
                        print(f'  Download failed: {match.release_name}: {e}')

                candidates[name] = video_candidates
            except Exception as e:
                print(f'  Search failed: {e}')
                candidates[name] = []

        # Phase 2: Density correlation score matrix
        print('\n\n' + '=' * 60)
        print('DENSITY CORRELATION SCORE MATRIX')
        print('=' * 60)

        ref_names = list(references.keys())

        for window in windows:
            print(f'\n--- Window: {window}ms ---')

            own_scores = []
            cross_scores = []

            for ref_name in ref_names:
                ref_ts, ref_dur = ref_timestamps[ref_name]

                print(f'\n  [{ref_name}]')

                # Score own candidates
                for cand_label, cand_path in candidates.get(ref_name, []):
                    cand_ts, cand_dur = extract_timestamps(cand_path)
                    if not cand_ts:
                        print(f'    OWN   | ---   | {cand_label} (no dialogue)')
                        continue

                    duration = max(ref_dur, cand_dur)
                    ref_density = build_density_vector(ref_ts, duration, window)
                    cand_density = build_density_vector(cand_ts, duration, window)
                    score = compute_density_correlation(ref_density, cand_density)
                    own_scores.append(score)
                    print(f'    OWN   | {score:.3f} | {cand_label}')

                # Cross-test against other episodes
                for other_name in ref_names[:8]:
                    if other_name == ref_name:
                        continue
                    for cand_label, cand_path in candidates.get(other_name, [])[:2]:
                        cand_ts, cand_dur = extract_timestamps(cand_path)
                        if not cand_ts:
                            continue
                        duration = max(ref_dur, cand_dur)
                        ref_density = build_density_vector(ref_ts, duration, window)
                        cand_density = build_density_vector(cand_ts, duration, window)
                        score = compute_density_correlation(ref_density, cand_density)
                        cross_scores.append(score)
                        print(f'    CROSS | {score:.3f} | from {other_name}: {cand_label}')

            # Summary
            if own_scores and cross_scores:
                print(f'\n  SUMMARY (window={window}ms):')
                print(
                    f'    OWN   scores: min={min(own_scores):.3f} '
                    f'avg={sum(own_scores) / len(own_scores):.3f} '
                    f'max={max(own_scores):.3f} (n={len(own_scores)})'
                )
                print(
                    f'    CROSS scores: min={min(cross_scores):.3f} '
                    f'avg={sum(cross_scores) / len(cross_scores):.3f} '
                    f'max={max(cross_scores):.3f} (n={len(cross_scores)})'
                )
                gap = min(own_scores) - max(cross_scores)
                print(f'    Gap (min_own - max_cross): {gap:.3f}')
                if gap > 0:
                    suggested = (min(own_scores) + max(cross_scores)) / 2
                    print(f'    Suggested threshold: {suggested:.3f}')
                else:
                    print('    WARNING: OWN and CROSS scores overlap!')


if __name__ == '__main__':
    main()
