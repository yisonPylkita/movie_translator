"""Download a low-res copy of a hardsubbed stream for OCR.

The OCR step only needs the burned-in Polish text to be *legible*, not high
quality — so we deliberately grab the smallest format whose height is still
>= the OCR floor (`min_height`), not the absolute worst. This pairs with
``format_sort=['+size','+res']`` (ascending) in the yt-dlp opts, which is what
actually makes the selection prefer the SMALLEST qualifying candidate.

URL resolution to a player embed happens upstream (the browser userscript), so
`download_episode` always hands a ready embed URL to yt-dlp for format
negotiation.
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


def _build_ydl_opts(
    out_path: str,
    min_height: int,
    best: bool,
    referer: str | None,
) -> dict:
    """Build the yt-dlp options dict for either download mode.

    Two modes, selected by `best`:

    - `best=False` (OCR): smallest format >= `min_height`, paired with the
      ascending `format_sort` that makes the >= floor resolve to the SMALLEST
      qualifying track. Writes the exact `out_path` (caller supplies the ext).
    - `best=True` (watch-it download): highest-quality video+audio (`bv*+ba/b`)
      with no ascending sort, so yt-dlp's default prefers the best track. The
      caller's suffix is dropped and yt-dlp picks the real container ext via
      `%(ext)s` (a best-quality merge is typically `.mkv`).

    When `referer` is given the media request carries it plus a desktop Chrome
    User-Agent (some hosts 403 otherwise).
    """
    opts: dict = {
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'overwrites': True,
    }
    if best:
        opts['format'] = 'bv*+ba/b'
        # Strip the caller's ext; yt-dlp fills in the real container.
        stem = str(Path(out_path).with_suffix(''))
        opts['outtmpl'] = f'{stem}.%(ext)s'
        # Land a real, playable container. Merging separate video+audio writes
        # mkv; the remux PP additionally rescues single-stream HLS sources,
        # which yt-dlp otherwise leaves as MPEG-TS bytes under a bogus `.m3u8`
        # extension that most players refuse to open. mkv is a stream-copy
        # (no re-encode) and tolerates whatever codecs the source carries.
        opts['merge_output_format'] = 'mkv'
        opts['postprocessors'] = [{'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mkv'}]
    else:
        opts['format'] = _format_selector(min_height)
        opts['outtmpl'] = out_path
        # Ascending: smallest filesize first, then lowest resolution. This is
        # what makes `bv*[height>=H]` resolve to the SMALLEST qualifying track
        # (e.g. cda 480p ~248MB) instead of the largest.
        opts['format_sort'] = ['+size', '+res']
    if referer:
        opts['http_headers'] = {'Referer': referer, 'User-Agent': _DESKTOP_UA}
    return opts


def download_episode(
    embed_url: str,
    out_path: str,
    min_height: int = DEFAULT_MIN_HEIGHT,
    best: bool = False,
    referer: str | None = None,
) -> str:
    """Download the player embed URL to `out_path`.

    `best=False` (default, OCR) grabs the smallest copy whose height is still
    >= `min_height`; `best=True` grabs the highest-quality video+audio and lets
    yt-dlp choose the container extension. See :func:`_build_ydl_opts`.

    Returns the path actually written. In OCR mode that is `out_path`; in best
    mode the extension may differ from the one passed (yt-dlp picks it), so the
    real written path is resolved from the output template. Raises
    :class:`HardsubError` on any failure (yt-dlp error, empty/missing output).
    """
    # Imported lazily so callers that don't download don't pay yt-dlp's import
    # cost (and so the module imports cleanly without yt-dlp present).
    import yt_dlp

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = _build_ydl_opts(out_path, min_height, best, referer)
    logger.info(
        'yt-dlp download: %s -> %s (format=%r, best=%s)',
        embed_url,
        ydl_opts['outtmpl'],
        ydl_opts['format'],
        best,
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([embed_url])
    except yt_dlp.utils.DownloadError as exc:
        raise HardsubError(f'yt-dlp download failed for {embed_url}: {exc}') from exc

    written = _resolve_written_path(out, best)
    if written is None:
        raise HardsubError(f'yt-dlp produced no output for {embed_url}')

    logger.info('yt-dlp download complete: %d bytes', written.stat().st_size)
    return str(written)


def _resolve_written_path(out: Path, best: bool) -> Path | None:
    """Find the non-empty file yt-dlp actually wrote, or None.

    OCR mode writes the exact `out` path. Best mode templates the extension
    (`%(ext)s`), so the real file shares `out`'s stem but may differ in ext —
    pick the newest non-empty `<stem>.*` match.
    """
    if not best:
        return out if out.exists() and out.stat().st_size > 0 else None
    candidates = [
        p for p in out.parent.glob(f'{out.stem}.*') if p.is_file() and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
