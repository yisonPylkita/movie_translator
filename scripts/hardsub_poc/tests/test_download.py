"""Tests for the low-res download PoC module.

These never touch the network. The pure routing/selector helpers are
tested directly; the direct-download path is exercised by monkeypatching
`requests.get` with a fake streaming response so no socket is opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hardsub_poc import download as download_mod
from scripts.hardsub_poc.contracts import DEFAULT_MIN_HEIGHT, HardsubError, VideoSource

# --- _is_direct_media_url ------------------------------------------------


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('https://example.com/video.MP4', True),
        ('https://example.com/clip.mkv', True),
        # ALL cda hosts (watch + embed) -> yt-dlp (it has a cda extractor),
        # even when the URL carries a media-looking extension.
        ('https://ebd123.cda.pl/o/abc.mp4', False),
        ('https://ebd99.cda.pl/620x368/somefile', False),
        ('https://www.cda.pl/video/123abc', False),
        ('https://cda.pl/video/123abc', False),
        # ogladajanime / embed pages -> yt-dlp.
        ('https://ogladajanime.pl/anime/some-slug/1', False),
        ('https://youtube.com/watch?v=xyz', False),
        # HLS / DASH manifests look file-ish but are playlists -> yt-dlp.
        ('https://cdn.example.com/stream/index.m3u8', False),
        ('https://cdn.example.com/stream/manifest.mpd', False),
    ],
)
def test_is_direct_media_url(url: str, expected: bool) -> None:
    assert download_mod._is_direct_media_url(url) is expected


# --- _format_selector ----------------------------------------------------


def test_format_selector_embeds_min_height() -> None:
    selector = download_mod._format_selector(480)
    assert 'height>=480' in selector
    # Both the adaptive and progressive branches carry the floor.
    assert selector.count('height>=480') == 2
    # A final unconditional fallback (no height floor) so we never hard-fail.
    assert selector.endswith('/bv*+ba/b')


def test_format_selector_uses_given_height() -> None:
    assert 'height>=720' in download_mod._format_selector(720)


# --- direct-download path (mocked, no network) ---------------------------


class _FakeResponse:
    """Minimal stand-in for a streaming `requests` response."""

    def __init__(self, chunks: list[bytes], *, status_ok: bool = True) -> None:
        self._chunks = chunks
        self._status_ok = status_ok
        self.captured_kwargs: dict = {}

    def raise_for_status(self) -> None:
        if not self._status_ok:
            import requests

            raise requests.HTTPError('403 Forbidden')

    def iter_content(self, chunk_size: int):  # noqa: ANN201 - mirrors requests
        yield from self._chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def patch_requests_get(monkeypatch):
    """Pin `requests.get` to return a fake response; capture call kwargs."""
    calls: dict = {}

    def _set(response: _FakeResponse) -> dict:
        def fake_get(url, **kwargs):
            calls['url'] = url
            calls['kwargs'] = kwargs
            return response

        monkeypatch.setattr(download_mod.requests, 'get', fake_get)
        return calls

    return _set


def test_direct_download_writes_file_and_passes_headers(patch_requests_get, tmp_path: Path) -> None:
    headers = {
        'Referer': 'https://ogladajanime.pl/',
        'User-Agent': 'Mozilla/5.0',
        'Origin': 'https://ogladajanime.pl',
    }
    source = VideoSource(url='https://media.example.com/o/movie.mp4', headers=headers)
    out_path = tmp_path / 'nested' / 'out.mp4'

    calls = patch_requests_get(_FakeResponse([b'hello ', b'world']))

    result = download_mod.download_lowest_legible(source, out_path)

    assert result == out_path
    assert out_path.read_bytes() == b'hello world'
    # Parent dir was created.
    assert out_path.parent.is_dir()
    # Headers were threaded through to requests.get.
    assert calls['kwargs']['headers'] == headers
    assert calls['kwargs']['stream'] is True
    assert calls['kwargs']['timeout'] is not None


def test_direct_download_empty_response_raises(patch_requests_get, tmp_path: Path) -> None:
    source = VideoSource(url='https://media.example.com/o/empty.mp4', headers={'Referer': 'x'})
    out_path = tmp_path / 'empty.mp4'

    patch_requests_get(_FakeResponse([]))

    with pytest.raises(HardsubError, match='empty'):
        download_mod.download_lowest_legible(source, out_path)


def test_direct_download_http_error_raises(patch_requests_get, tmp_path: Path) -> None:
    source = VideoSource(url='https://media.example.com/o/forbidden.mp4', headers={'Referer': 'x'})
    out_path = tmp_path / 'forbidden.mp4'

    patch_requests_get(_FakeResponse([b'x'], status_ok=False))

    with pytest.raises(HardsubError, match='Direct download failed'):
        download_mod.download_lowest_legible(source, out_path)


# --- routing: page URL goes to yt-dlp ------------------------------------


def test_page_url_routes_to_ytdlp(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_ytdlp(source: VideoSource, out_path: Path, min_height: int) -> Path:
        captured['source'] = source
        captured['min_height'] = min_height
        out_path.write_bytes(b'ok')
        return out_path

    monkeypatch.setattr(download_mod, '_download_with_ytdlp', fake_ytdlp)

    source = VideoSource(url='https://ogladajanime.pl/anime/slug/1')
    out_path = tmp_path / 'video.mp4'

    result = download_mod.download_lowest_legible(source, out_path, min_height=540)

    assert result == out_path
    assert captured['source'] is source
    assert captured['min_height'] == 540


def test_default_min_height_threaded_to_ytdlp(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_ytdlp(source: VideoSource, out_path: Path, min_height: int) -> Path:
        captured['min_height'] = min_height
        out_path.write_bytes(b'ok')
        return out_path

    monkeypatch.setattr(download_mod, '_download_with_ytdlp', fake_ytdlp)

    source = VideoSource(url='https://ogladajanime.pl/anime/slug/1')
    download_mod.download_lowest_legible(source, tmp_path / 'v.mp4')

    assert captured['min_height'] == DEFAULT_MIN_HEIGHT
