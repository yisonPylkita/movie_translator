"""Tests for the low-res download module's pure helpers.

These never touch the network: only the pure routing/selector helpers are
exercised. The actual yt-dlp download path is not unit-tested (no network).
"""

from __future__ import annotations

import pytest

from movie_translator.hardsub import download as download_mod

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
