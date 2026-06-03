"""Extract resolved player URLs from a mitmproxy capture of ogladajanime.pl.

This is the *reliable* path: a real browser (the user's own Chrome) solves
Cloudflare Turnstile invisibly while they click players; mitmproxy records
the JSON API responses; this tool reads the embed URLs back out. No headless
browser, no Turnstile-solving, no DOM scraping.

Two API responses carry everything (reverse-engineered from a live capture,
2026-06-03):

  * ``manager.php?action=get_player_list`` -> a nested-JSON ``data`` field
    holding the full player catalog (id, host, audio, sub, quality) plus the
    *default* player's resolved ``url``.
  * ``manager.php?action=change_player_url`` -> ``data`` is the chosen
    player's embed URL directly (e.g. ``https://ebd.cda.pl/800x400/...``),
    which yt-dlp can download.

Run it as a self-terminating mitmproxy addon (reads the file and exits)::

    mitmdump -q -n -r capture.flow -s flow_extract.py

It prints one JSON object to stdout: ``{"catalog": [...], "resolved": [...]}``.
"""

from __future__ import annotations

import json
import sys
from urllib.parse import parse_qs

from mitmproxy import ctx, http

_HOST = 'ogladajanime.pl'


class FlowExtract:
    """Collect the player catalog + every resolved embed URL, then exit."""

    def __init__(self) -> None:
        self.catalog: list[dict] = []
        # player_id (str) -> embed url, from change_player_url responses.
        self.resolved: dict[str, str] = {}
        # default url + which ep the catalog belongs to, for context.
        self.default_url: str | None = None
        self.episode_id: str | None = None

    def response(self, flow: http.HTTPFlow) -> None:
        url = flow.request.pretty_url
        if _HOST not in url or flow.response is None:
            return
        body = flow.response.get_text(strict=False) or ''

        if 'action=get_player_list' in url:
            self._parse_player_list(flow, body)
        elif 'action=change_player_url' in url:
            self._parse_change_player(flow, body)

    def _parse_player_list(self, flow: http.HTTPFlow, body: str) -> None:
        try:
            outer = json.loads(body)
            inner = json.loads(outer['data'])
        except ValueError, KeyError, TypeError:
            return
        self.default_url = inner.get('url') or self.default_url
        form = parse_qs(flow.request.get_text(strict=False) or '')
        if form.get('id'):
            self.episode_id = form['id'][0]
        for p in inner.get('players', []):
            self.catalog.append(
                {
                    'player_id': p.get('id'),
                    'host': p.get('url'),
                    'audio': p.get('audio'),
                    'sub': p.get('sub'),
                    'quality': p.get('quality'),
                    'sub_group': p.get('sub_group'),
                }
            )

    def _parse_change_player(self, flow: http.HTTPFlow, body: str) -> None:
        try:
            outer = json.loads(body)
        except ValueError:
            return
        embed = outer.get('data')
        # Server rejects bad/expired tokens with a Polish message in `data`;
        # only keep values that look like a URL.
        if not isinstance(embed, str) or not embed.startswith('http'):
            return
        form = parse_qs(flow.request.get_text(strict=False) or '')
        player_id = form.get('id', ['?'])[0]
        self.resolved[player_id] = embed

    def done(self) -> None:
        catalog_by_id = {str(c['player_id']): c for c in self.catalog}
        resolved = [
            {
                **catalog_by_id.get(pid, {'player_id': pid}),
                'embed_url': url,
            }
            for pid, url in self.resolved.items()
        ]
        out = {
            'episode_id': self.episode_id,
            'default_url': self.default_url,
            'catalog': self.catalog,
            'resolved': resolved,
        }
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write('\n')
        sys.stdout.flush()


addons = [FlowExtract()]


def done() -> None:  # module-level safety net; addons own the real work.
    ctx.master.shutdown()
