"""Download a low-res copy of a hardsubbed stream for OCR.

The OCR step only needs the burned-in Polish text to be *legible*, not high
quality — so we deliberately grab the smallest format whose height is still
>= the OCR floor (`min_height`), not the absolute worst. This pairs with
``format_sort=['+size','+res']`` (ascending) in the yt-dlp opts, which is what
actually makes the selection prefer the SMALLEST qualifying candidate.

Graduated from the `scripts/hardsub_poc/download.py` PoC. Only the embed/page
path (yt-dlp format negotiation) is needed in the main pipeline — URL
resolution to a player embed happens upstream, so `download_episode` always
hands the embed URL to yt-dlp.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Lowest resolution we'll accept for OCR. Below ~480p the burned-in Polish text
# smears and Vision OCR produces garbage; above it the download is needlessly
# slow. We pick the smallest format whose height is >= this floor (falling back
# to the largest available if none meet it).
DEFAULT_MIN_HEIGHT = 480

# Desktop Chrome UA — some hosts 403 a non-browser UA on the media request.
_DESKTOP_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
)

# Extensions that name a single downloadable media file. `.m3u8` (HLS) and
# `.mpd` (DASH) are intentionally absent — they are manifests referencing many
# segments, so they go through yt-dlp instead.
_DIRECT_MEDIA_EXTENSIONS = frozenset(
    {'.mp4', '.m4v', '.mkv', '.webm', '.mov', '.avi', '.flv', '.ts'}
)

# Every cda host — watch pages (cda.pl, www.cda.pl) AND embed hosts
# (ebd.cda.pl, ebdXXX.cda.pl) — is an HTML player page, not a direct media
# file. yt-dlp has a dedicated cda.pl extractor, so all *.cda.pl URLs route to
# yt-dlp for resolution + format selection.
_CDA_HOST_SUFFIX = '.cda.pl'


class HardsubError(RuntimeError):
    """A hardsub step failing in a way the user should see + act on."""


def _is_direct_media_url(url: str) -> bool:
    """Return True if `url` is a single downloadable media file.

    Anything else (an embed or episode page, an HLS/DASH manifest) is handed to
    yt-dlp for format negotiation. The decision is path-extension first, with a
    cda edge-host fallback for extension-less direct links.
    """
    parts = urlsplit(url)
    path = parts.path.lower()
    suffix = Path(path).suffix

    # HLS/DASH manifests look file-ish but are playlists -> yt-dlp.
    if suffix in {'.m3u8', '.mpd'}:
        return False

    # All cda hosts are player pages -> yt-dlp (it has a cda extractor), even if
    # the URL happens to carry a media-looking extension.
    host = parts.netloc.lower()
    if host == 'cda.pl' or host.endswith(_CDA_HOST_SUFFIX):
        return False

    return suffix in _DIRECT_MEDIA_EXTENSIONS


def _format_selector(min_height: int) -> str:
    """Build the yt-dlp format selector string for `min_height`.

    Intent: pick the *lowest* resolution still >= the OCR floor so the download
    is fast but the burned-in text stays legible. This pairs with
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


def download_episode(
    embed_url: str,
    out_path: str,
    min_height: int = DEFAULT_MIN_HEIGHT,
    referer: str | None = None,
) -> str:
    """Download the lowest OCR-legible copy of the player embed URL to out_path.

    Hands `embed_url` to yt-dlp, which enumerates available formats and (paired
    with the ascending size sort) picks the smallest height >= `min_height`,
    falling back to the largest available. When `referer` is given the media
    request carries it plus a desktop Chrome User-Agent (some hosts 403
    otherwise).

    Returns the path written (== `out_path`). Raises :class:`HardsubError` on
    any failure (yt-dlp error, empty/missing output).
    """
    # Imported lazily so callers that don't download don't pay yt-dlp's import
    # cost (and so the module imports cleanly without yt-dlp present).
    import yt_dlp

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    selector = _format_selector(min_height)
    logger.info('yt-dlp download: %s -> %s (format=%r)', embed_url, out, selector)

    ydl_opts: dict = {
        'format': selector,
        'outtmpl': str(out),
        # Ascending: smallest filesize first, then lowest resolution. This is
        # what makes `bv*[height>=H]` resolve to the SMALLEST qualifying track
        # (e.g. cda 480p ~248MB) instead of the largest.
        'format_sort': ['+size', '+res'],
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'overwrites': True,
    }
    if referer:
        ydl_opts['http_headers'] = {'Referer': referer, 'User-Agent': _DESKTOP_UA}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([embed_url])
    except yt_dlp.utils.DownloadError as exc:
        raise HardsubError(f'yt-dlp download failed for {embed_url}: {exc}') from exc

    if not out.exists() or out.stat().st_size == 0:
        raise HardsubError(f'yt-dlp produced no output for {embed_url}')

    logger.info('yt-dlp download complete: %d bytes', out.stat().st_size)
    return str(out)
