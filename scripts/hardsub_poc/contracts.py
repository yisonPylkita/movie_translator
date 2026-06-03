"""Shared contracts for the hardsub-OCR PoC.

Small stable types shared by the orchestrator (`__main__`) and the
`download` module: the resolved-stream value object, the OCR resolution
floor, the host-preference order, and the error type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Domain the whole PoC targets.
OGLADAJANIME_DOMAIN = 'ogladajanime.pl'
OGLADAJANIME_BASE_URL = 'https://ogladajanime.pl'

# Lowest resolution we'll accept for OCR. Below ~480p the burned-in Polish
# text smears and Vision OCR produces garbage; above it the download is
# needlessly slow. The download module picks the smallest format whose
# height is >= this floor (falling back to the largest available if none
# meet it).
DEFAULT_MIN_HEIGHT = 480

# Host preference order for the hardsub stream. yt-dlp resolves these most
# reliably: cda is the PL-favored default and downloads cleanly; sibnet is
# the next-cleanest; dood/google/mp4upload are further fallbacks. The
# orchestrator picks the most-preferred host present among resolved players.
HOST_PREFERENCE = ('cda', 'sibnet', 'dood', 'google', 'mp4upload')


@dataclass
class VideoSource:
    """A resolved, directly-downloadable video stream.

    `url` is the direct media URL (cda's in-iframe `<video>.src`, or a
    format URL yt-dlp can fetch). `headers` carries whatever the host
    requires to serve it without a 403 — typically Referer/Origin/UA.
    `host` is the embed host we resolved it from (e.g. "cda"), for logging.
    `page_url` is the ogladajanime episode page we started from, for logs.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    host: str | None = None
    page_url: str | None = None


class HardsubError(RuntimeError):
    """Any step of the PoC failing in a way the user should see + act on."""
