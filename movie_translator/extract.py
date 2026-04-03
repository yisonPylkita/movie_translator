"""Subtitle extraction pipeline — extracts text and OCR subtitles from video files."""

import json
from pathlib import Path

from .discovery import find_videos
from .identifier import identify_media
from .identifier.types import MediaIdentity
from .logging import console, logger
from .ocr.burned_in_extractor import extract_burned_in_subtitles
from .subtitles import SubtitleExtractor


def _build_output_stem(identity: MediaIdentity) -> str:
    """Build a normalized filename stem from media identity."""
    title = identity.parsed_title or identity.title or 'Unknown'
    # Sanitize title for filesystem
    title = ''.join(c if c.isalnum() or c in ' -_' else '' for c in title).strip()

    if (
        identity.media_type == 'episode'
        and identity.season is not None
        and identity.episode is not None
    ):
        return f'{title} - S{identity.season:02d}E{identity.episode:02d}'
    elif identity.episode is not None:
        return f'{title} - E{identity.episode:02d}'
    else:
        return title


def _identity_to_dict(identity: MediaIdentity) -> dict:
    """Convert MediaIdentity to a JSON-serializable dict."""
    return {
        'title': identity.title,
        'parsed_title': identity.parsed_title,
        'season': identity.season,
        'episode': identity.episode,
        'media_type': identity.media_type,
        'is_anime': identity.is_anime,
    }


def _extract_text_tracks(
    video_path: Path,
    output_dir: Path,
    extractor: SubtitleExtractor,
    output_stem: str,
) -> list[dict]:
    """Extract embedded text subtitle tracks for English and Polish."""
    track_info = extractor.get_track_info(video_path)
    tracks = track_info.get('tracks', [])
    results = []

    # Languages we care about
    lang_map = {
        'eng': 'en',
        'en': 'en',
        'und': 'en',
        'pol': 'pl',
        'pl': 'pl',
    }

    for track in tracks:
        props = track.get('properties', {})
        lang = (props.get('language') or '').lower()
        if lang not in lang_map:
            continue

        codec = track.get('codec', '').lower()
        # Skip image-based tracks — those need OCR, handled separately
        if any(codec == c or codec.startswith(c) for c in SubtitleExtractor.IMAGE_CODECS):
            continue

        # Skip signs/songs
        track_name = (props.get('track_name') or '').lower()
        if track_name and any(kw in track_name for kw in ('sign', 'song', 'op', 'ed')):
            continue

        out_lang = lang_map[lang]
        ext = extractor.get_subtitle_extension(track)
        out_file = f'{output_stem}.{out_lang}{ext}'
        out_path = output_dir / out_file

        try:
            extractor.extract_subtitle(
                video_path,
                track['id'],
                out_path,
                subtitle_index=track.get('subtitle_index'),
            )
            line_count = _count_subtitle_lines(out_path)
            results.append(
                {
                    'file': out_file,
                    'language': out_lang,
                    'method': 'embedded_text',
                    'line_count': line_count,
                }
            )
            logger.info(f'Extracted {out_lang} text track: {out_file} ({line_count} lines)')
        except Exception as e:
            logger.warning(f'Failed to extract track {track["id"]}: {e}')

    return results


def _count_subtitle_lines(path: Path) -> int:
    """Count dialogue lines in a subtitle file."""
    try:
        import pysubs2

        subs = pysubs2.load(str(path))
        return len([e for e in subs.events if e.type == 'Dialogue'])
    except Exception:
        # Fallback: count SRT blocks
        text = path.read_text(encoding='utf-8', errors='replace')
        return text.count(' --> ')


def _extract_ocr(
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    language: str,
) -> list[dict]:
    """Extract burned-in subtitles via OCR."""
    work_dir = output_dir / '_ocr_work'
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = extract_burned_in_subtitles(
            video_path,
            work_dir,
            language=language,
        )
        if result is None:
            logger.warning(f'No burned-in subtitles found in {video_path.name}')
            return []

        # Copy SRT to output with normalized name
        out_file = f'{output_stem}.{language}.ocr.srt'
        out_path = output_dir / out_file
        import shutil

        shutil.copy2(result.srt_path, out_path)

        line_count = _count_subtitle_lines(out_path)
        logger.info(f'Extracted {language} OCR subtitles: {out_file} ({line_count} lines)')

        return [
            {
                'file': out_file,
                'language': language,
                'method': 'ocr_burned_in',
                'line_count': line_count,
            }
        ]
    finally:
        import shutil

        if work_dir.exists():
            shutil.rmtree(work_dir)


def run_extract(
    input_path: Path,
    output_dir: Path,
    ocr_language: str = 'pl',
) -> Path:
    """Run the extraction pipeline on one or more video files.

    Returns the path to the manifest JSON.
    """
    video_files = find_videos(input_path)
    if not video_files:
        console.print(f'[red]No video files found in {input_path}[/red]')
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SubtitleExtractor()

    manifest = {
        'version': 1,
        'source_dir': str(input_path.resolve()),
        'entries': [],
    }

    console.print(f'Extracting subtitles from {len(video_files)} file(s)...')

    for video_path in video_files:
        console.print(f'\n[bold]{video_path.name}[/bold]')

        # Step 1: Identify
        logger.info(f'Identifying: {video_path.name}')
        identity = identify_media(video_path)
        output_stem = _build_output_stem(identity)
        console.print(f'  Identified: {identity.title} (S{identity.season}E{identity.episode})')

        entry = {
            'source_file': video_path.name,
            'identity': _identity_to_dict(identity),
            'subtitles': [],
        }

        # Step 2: Extract embedded text tracks
        text_subs = _extract_text_tracks(video_path, output_dir, extractor, output_stem)
        entry['subtitles'].extend(text_subs)

        # Step 3: OCR burned-in subtitles
        ocr_subs = _extract_ocr(video_path, output_dir, output_stem, ocr_language)
        entry['subtitles'].extend(ocr_subs)

        if not entry['subtitles']:
            console.print('  [yellow]No subtitles extracted[/yellow]')
        else:
            for sub in entry['subtitles']:
                console.print(
                    f'  [green]{sub["file"]}[/green] '
                    f'({sub["language"]}, {sub["method"]}, {sub["line_count"]} lines)'
                )

        manifest['entries'].append(entry)

    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    console.print(f'\n[bold green]Manifest written to {manifest_path}[/bold green]')

    return manifest_path
