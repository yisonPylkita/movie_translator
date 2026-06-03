"""Download a low-res copy of a hardsubbed stream for OCR.

The OCR step only needs the burned-in Polish text to be *legible*, not
high quality — so we deliberately grab the smallest format whose height
is still >= the OCR floor (`DEFAULT_MIN_HEIGHT`), not the absolute worst.

Two source shapes are handled:

1. A *direct media URL* — cda's in-iframe `<video>.src`, typically a
   single mp4 with no format ladder. We stream it straight to disk with
   the host-required headers (Referer/Origin/User-Agent — cda 403s
   without them). There is no resolution to choose; we just download it.

2. A *page / embed URL* needing format negotiation. We hand it to
   `yt_dlp`, which enumerates the available formats and lets us pick the
   lowest height >= `min_height` (falling back to the largest available
   when nothing qualifies).

`_is_direct_media_url` is the (pure) routing decision so it can be
tested in isolation. Note HLS (`.m3u8`) is a playlist, not a plain file,
so it routes to yt-dlp despite looking like a direct URL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .contracts import DEFAULT_MIN_HEIGHT, HardsubError, VideoSource

logger = logging.getLogger(__name__)

# Extensions we can stream straight to disk as a single file. `.m3u8`
# (HLS) and `.mpd` (DASH) are intentionally absent — they are manifests
# that reference many segments, so they go through yt-dlp instead.
_DIRECT_MEDIA_EXTENSIONS = frozenset(
    {'.mp4', '.m4v', '.mkv', '.webm', '.mov', '.avi', '.flv', '.ts'}
)

# Every cda host — the watch pages (cda.pl, www.cda.pl) AND the embed
# hosts (ebd.cda.pl, ebdXXX.cda.pl) — is an HTML player page, not a direct
# media file. yt-dlp has a dedicated cda.pl extractor, so all *.cda.pl URLs
# route to yt-dlp for resolution + format selection.
_CDA_HOST_SUFFIX = '.cda.pl'

# Streaming download tuning.
_CHUNK_SIZE = 1 << 16  # 64 KiB
_CONNECT_READ_TIMEOUT = (10, 60)  # (connect, read) seconds


def _is_direct_media_url(url: str) -> bool:
    """Return True if `url` is a single downloadable media file.

    A direct URL is streamed with `requests`; anything else (an embed or
    episode page, an HLS/DASH manifest) is handed to yt-dlp for format
    negotiation. The decision is path-extension first, with a cda
    edge-host fallback for extension-less direct links.
    """
    parts = urlsplit(url)
    path = parts.path.lower()
    suffix = Path(path).suffix

    # HLS/DASH manifests look file-ish but are playlists -> yt-dlp.
    if suffix in {'.m3u8', '.mpd'}:
        return False

    # All cda hosts are player pages -> yt-dlp (it has a cda extractor),
    # even if the URL happens to carry a media-looking extension.
    host = parts.netloc.lower()
    if host == 'cda.pl' or host.endswith(_CDA_HOST_SUFFIX):
        return False

    if suffix in _DIRECT_MEDIA_EXTENSIONS:
        return True

    return False


def _format_selector(min_height: int) -> str:
    """Build the yt-dlp format selector string for `min_height`.

    Intent: pick the *lowest* resolution still >= the OCR floor so the
    download is fast but the burned-in text stays legible. This pairs with
    ``format_sort=['+size','+res']`` (ascending) in the yt-dlp opts, which is
    what actually makes the selection prefer the SMALLEST candidate.

    - `bv*[height>=H]+ba` — a >= floor video track (incl. muxed) plus audio;
      with the ascending sort this resolves to the smallest qualifying combo.
      (Note: `worstvideo[height>=H]` misbehaves on cda's MPD — it picked 1080p
      in testing — so we use `bv*` + size-sort instead.)
    - `b[height>=H]` — progressive (pre-muxed) fallback at/above the floor.
    - `bv*+ba/b` — last resort if nothing meets the floor: download the
      smallest available rather than fail.
    """
    return f'bv*[height>={min_height}]+ba/b[height>={min_height}]/bv*+ba/b'


def _download_direct(source: VideoSource, out_path: Path) -> Path:
    """Stream a single media file to `out_path` with the host headers."""
    logger.info('Direct download: %s -> %s', source.url, out_path)
    try:
        with requests.get(
            source.url,
            headers=source.headers,
            stream=True,
            timeout=_CONNECT_READ_TIMEOUT,
        ) as response:
            response.raise_for_status()

            total = 0
            last_logged_mb = 0
            with out_path.open('wb') as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
                    mb = total // (1 << 20)
                    if mb >= last_logged_mb + 16:
                        last_logged_mb = mb
                        logger.info('  downloaded %d MiB', mb)
    except requests.RequestException as exc:
        raise HardsubError(f'Direct download failed for {source.url}: {exc}') from exc

    if total == 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise HardsubError(f'Direct download produced an empty file for {source.url}')

    logger.info('Direct download complete: %d bytes', out_path.stat().st_size)
    return out_path


def _download_with_ytdlp(source: VideoSource, out_path: Path, min_height: int) -> Path:
    """Negotiate formats and download via yt-dlp to `out_path`."""
    # Imported lazily so unit tests that only exercise the pure helpers
    # don't pay yt-dlp's import cost.
    import yt_dlp

    selector = _format_selector(min_height)
    logger.info('yt-dlp download: %s -> %s (format=%r)', source.url, out_path, selector)

    ydl_opts: dict = {
        'format': selector,
        'outtmpl': str(out_path),
        # Ascending: smallest filesize first, then lowest resolution. This is
        # what makes `bv*[height>=H]` resolve to the SMALLEST qualifying track
        # (e.g. cda 480p ~248MB) instead of the largest.
        'format_sort': ['+size', '+res'],
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'overwrites': True,
    }
    if source.headers:
        ydl_opts['http_headers'] = dict(source.headers)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source.url])
    except yt_dlp.utils.DownloadError as exc:
        raise HardsubError(f'yt-dlp download failed for {source.url}: {exc}') from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise HardsubError(f'yt-dlp produced no output for {source.url}')

    logger.info('yt-dlp download complete: %d bytes', out_path.stat().st_size)
    return out_path


def download_lowest_legible(
    source: VideoSource, out_path: Path, min_height: int = DEFAULT_MIN_HEIGHT
) -> Path:
    """Download the lowest OCR-legible copy of `source` to `out_path`.

    For a direct media URL the file is streamed as-is (no resolution
    choice exists). For a page/embed URL yt-dlp negotiates formats and
    picks the smallest height >= `min_height`, falling back to the
    largest available. Returns the written `out_path`.

    Raises `HardsubError` on any download failure (HTTP error, empty
    file, yt-dlp DownloadError).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_direct_media_url(source.url):
        return _download_direct(source, out_path)
    return _download_with_ytdlp(source, out_path, min_height)
