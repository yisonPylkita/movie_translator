"""Hardsub-OCR PoC orchestrator: players JSON -> download -> OCR -> .srt.

Usage (from repo root, with the venv python):

    .venv/bin/python -m scripts.hardsub_poc <players.json> [options]

The players JSON is produced by one of the two resolver paths (the site's
anti-debug rules out a headless/CDP scraper — see the design doc):

  * `ogladajanime_resolver.user.js` — a Tampermonkey userscript that runs
    in the real browser, iterates every player, and downloads a JSON.
  * `flow_extract.py` — reads the same JSON out of a mitmproxy capture of
    the user clicking players.

Both emit `{episode_url, resolved: [{host, sub, quality, embed_url, ...}]}`.
This orchestrator picks the best resolved player (PL sub, host preference),
downloads the lowest OCR-legible copy via yt-dlp, OCRs the burned-in Polish
subtitles, and writes a `.srt`.

Standalone PoC — NOT wired into the Rust pipeline. Design:
docs/superpowers/specs/2026-06-03-hardsub-ocr-poc-design.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from .contracts import (
    DEFAULT_MIN_HEIGHT,
    HOST_PREFERENCE,
    HardsubError,
    VideoSource,
)
from .download import download_lowest_legible

logger = logging.getLogger('hardsub_poc')

# A plausible desktop Chrome UA; cda 403s a bot-looking one.
_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'
)


def select_source(
    resolved: list[dict],
    *,
    sub: str | None = 'pl',
    host: str | None = None,
    host_preference: tuple[str, ...] = HOST_PREFERENCE,
) -> dict | None:
    """Pick the best resolved player to download (pure; unit-testable).

    Filters to `sub` language (default 'pl' for the hardsub target) and, if
    given, an exact `host`. Among the survivors, prefers hosts earliest in
    `host_preference` (cda first), then higher resolution. Returns the chosen
    entry dict, or None if nothing matches.
    """

    def height(entry: dict) -> int:
        q = str(entry.get('quality') or '')
        digits = ''.join(ch for ch in q if ch.isdigit())
        return int(digits) if digits else 0

    def host_rank(entry: dict) -> int:
        h = entry.get('host')
        return host_preference.index(h) if h in host_preference else len(host_preference)

    candidates = [e for e in resolved if e.get('embed_url')]
    if sub is not None:
        candidates = [e for e in candidates if (e.get('sub') or '') == sub]
    if host is not None:
        candidates = [e for e in candidates if e.get('host') == host]
    if not candidates:
        return None
    candidates.sort(key=lambda e: (host_rank(e), -height(e)))
    return candidates[0]


def load_episodes(data: dict) -> list[dict]:
    """Normalize any resolver JSON shape into a list of episode dicts.

    Handles all three producers (pure; unit-testable):
      * multi-episode userscript job: ``{episodes: [{episode, episode_url,
        resolved}, ...]}``
      * single-episode userscript: same, list of one
      * flow_extract.py flat shape: ``{episode_id, resolved, ...}`` -> wrapped
        into a one-element list.
    Each returned dict has at least ``resolved`` (list) and ``episode_url``.
    """
    if isinstance(data.get('episodes'), list):
        return [
            {
                'episode': ep.get('episode'),
                'episode_url': ep.get('episode_url'),
                'resolved': ep.get('resolved') or [],
            }
            for ep in data['episodes']
        ]
    # flow_extract flat shape.
    return [
        {
            'episode': data.get('episode_id'),
            'episode_url': data.get('episode_url'),
            'resolved': data.get('resolved') or [],
        }
    ]


def _video_source(entry: dict, episode_url: str | None) -> VideoSource:
    headers = {'User-Agent': _USER_AGENT}
    if episode_url:
        headers['Referer'] = episode_url
    return VideoSource(
        url=entry['embed_url'],
        headers=headers,
        host=entry.get('host'),
        page_url=episode_url,
    )


def run(args: argparse.Namespace) -> int:
    players_path = Path(args.players_json).expanduser()
    if not players_path.is_file():
        logger.error('Players JSON not found: %s', players_path)
        return 2
    try:
        data = json.loads(players_path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        logger.error('Could not read players JSON %s: %s', players_path, exc)
        return 2

    episodes = load_episodes(data)
    if args.episode is not None:
        episodes = [e for e in episodes if e.get('episode') == args.episode]
        if not episodes:
            logger.error('Episode %s not in JSON.', args.episode)
            return 1

    out_dir = Path(args.out).expanduser() if args.out else players_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = data.get('anime_slug') or players_path.stem.replace('.players', '') or 'anime'

    logger.info('Processing %d episode(s) from %s', len(episodes), players_path.name)
    ok, failed = 0, 0
    for ep in episodes:
        try:
            if _process_episode(ep, slug, out_dir, args):
                ok += 1
        except HardsubError as exc:
            failed += 1
            logger.error('Episode %s FAILED: %s', ep.get('episode'), exc)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            failed += 1
            logger.error('Episode %s crashed: %s', ep.get('episode'), exc)

    print(f'\n=== done: {ok} ok, {failed} failed, of {len(episodes)} episode(s) -> {out_dir}')
    return 0 if failed == 0 else 1


def _episode_label(ep: dict, slug: str) -> str:
    """`<slug>-ep<NN>` filename stem for an episode (zero-padded when numeric)."""
    n = ep.get('episode')
    if isinstance(n, int):
        return f'{slug}-ep{n:02d}'
    return f'{slug}-ep{n}' if n else slug


def _process_episode(ep: dict, slug: str, out_dir: Path, args: argparse.Namespace) -> bool:
    """Download + OCR one episode. Returns True on success, raises on failure.

    Skips (returns False) when the episode has no resolved player matching the
    sub/host filter — the batch loop then moves on to the next episode.
    """
    label = _episode_label(ep, slug)
    resolved = ep.get('resolved') or []
    if not resolved:
        logger.warning('[%s] no resolved players — skipping', label)
        return False

    chosen = select_source(resolved, sub=args.sub, host=args.host)
    if chosen is None:
        logger.warning(
            '[%s] no player matched sub=%r host=%r (have %s) — skipping',
            label,
            args.sub,
            args.host,
            [(e.get('host'), e.get('sub')) for e in resolved],
        )
        return False
    logger.info(
        '[%s] %s %s (sub=%s) -> %s',
        label,
        chosen.get('host'),
        chosen.get('quality'),
        chosen.get('sub'),
        chosen['embed_url'],
    )

    # Skip work already done (resumable batch).
    srt_path = out_dir / f'{label}.pl.srt'
    if not args.no_ocr and srt_path.exists() and not args.force:
        logger.info('[%s] %s exists — skipping (use --force to redo)', label, srt_path.name)
        return True

    temp_root = Path(tempfile.mkdtemp(prefix=f'hardsub_{label}_'))
    video_path = temp_root / 'stream.mp4'
    source = _video_source(chosen, ep.get('episode_url'))

    download_lowest_legible(source, video_path, min_height=args.min_height)
    logger.info(
        '[%s] downloaded %.1f MB via %s', label, video_path.stat().st_size / 1e6, chosen.get('host')
    )

    if args.no_ocr:
        final = out_dir / f'{label}.{chosen.get("host")}.mp4'
        final.write_bytes(video_path.read_bytes())
        logger.info('[%s] saved video (no OCR): %s', label, final.name)
        return True

    from movie_translator.ocr import extract_burned_in_subtitles, is_vision_ocr_available

    if not is_vision_ocr_available():
        raise HardsubError(
            'Apple Vision OCR unavailable (macOS-only). Use --no-ocr to just download.'
        )
    result = extract_burned_in_subtitles(video_path, temp_root, language='pl')
    if result is None:
        raise HardsubError(
            f'OCR produced no subtitles — check crop/resolution. Frames in {temp_root}.'
        )

    # Clean the raw per-frame OCR into deduped, garbage-filtered lines.
    from .postprocess import merge_ocr_results, to_srt

    frame_texts = [(r.timestamp_ms, r.text) for r in result.ocr_results]
    clean = merge_ocr_results(frame_texts)
    if not clean:
        raise HardsubError(f'No dialogue lines survived cleanup. Frames in {temp_root}.')
    srt_path.write_text(to_srt(clean), encoding='utf-8')
    logger.info(
        '[%s] wrote %s — %d lines (from %d raw OCR frames)',
        label,
        srt_path.name,
        len(clean),
        len(result.ocr_results),
    )

    # Optional: align to the local video's timeline against an English
    # reference (reuses the vendored ilass engine — see align.py).
    if args.reference_srt:
        from .align import align_in_place

        ref = Path(args.reference_srt).expanduser()
        if not ref.is_file():
            raise HardsubError(f'--reference-srt not found: {ref}')
        align_in_place(srt_path, ref)
        logger.info('[%s] aligned %s to reference %s', label, srt_path.name, ref.name)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m scripts.hardsub_poc',
        description='Hardsub-OCR Polish-subtitle PoC: players JSON -> download -> OCR -> .srt.',
    )
    parser.add_argument('players_json', help='Players JSON from the userscript or flow_extract.py')
    parser.add_argument(
        '--episode',
        type=int,
        default=None,
        help='Process only this episode (default: ALL episodes in the JSON)',
    )
    parser.add_argument(
        '--sub', default='pl', help="Sub language to require (default 'pl'; '' = any)"
    )
    parser.add_argument('--host', default=None, help='Force a specific host (e.g. cda, sibnet)')
    parser.add_argument(
        '--reference-srt',
        default=None,
        help="Align the OCR'd subs to this English reference .srt via ilass "
        '(use with --episode; one reference matches one episode)',
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=DEFAULT_MIN_HEIGHT,
        help='Lowest stream height to accept for OCR legibility (default: %(default)s)',
    )
    parser.add_argument('--out', default=None, help='Output dir (default: next to the JSON)')
    parser.add_argument('--no-ocr', action='store_true', help='Just download the video, skip OCR')
    parser.add_argument(
        '--force', action='store_true', help='Re-do episodes whose .srt already exists'
    )
    parser.add_argument('--keep-temp', action='store_true', help='Keep the scratch dir')
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug logging')
    args = parser.parse_args(argv)

    # Empty --sub means "any language".
    if args.sub == '':
        args.sub = None

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
