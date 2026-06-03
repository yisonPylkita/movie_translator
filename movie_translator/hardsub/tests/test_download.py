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


# --- _build_ydl_opts -----------------------------------------------------


def test_build_opts_ocr_mode_smallest_legible() -> None:
    """OCR mode (best=False): smallest >= floor, ascending sort, exact outtmpl."""
    opts = download_mod._build_ydl_opts('/tmp/clip.mp4', min_height=480, best=False, referer=None)
    assert opts['format'] == download_mod._format_selector(480)
    # Ascending sort is what makes the >= floor resolve to the SMALLEST track.
    assert opts['format_sort'] == ['+size', '+res']
    # OCR mode writes the exact path it was given.
    assert opts['outtmpl'] == '/tmp/clip.mp4'


def test_build_opts_best_mode_highest_quality() -> None:
    """Best mode: best video+audio, no ascending sort, ext templated by yt-dlp."""
    opts = download_mod._build_ydl_opts('/tmp/show-E01.mkv', min_height=0, best=True, referer=None)
    assert opts['format'] == 'bv*+ba/b'
    # No ascending size sort -> yt-dlp's default prefers the BEST track.
    assert 'format_sort' not in opts
    # Caller's suffix is stripped; yt-dlp picks the real container ext.
    assert opts['outtmpl'] == '/tmp/show-E01.%(ext)s'
    # Land a real playable container: merged tracks go to mkv, and a remux PP
    # rescues HLS sources that would otherwise keep a bogus `.m3u8` ext on
    # what is actually MPEG-TS video.
    assert opts['merge_output_format'] == 'mkv'
    remuxers = [
        pp for pp in opts.get('postprocessors', []) if pp.get('key') == 'FFmpegVideoRemuxer'
    ]
    assert remuxers, 'best mode must remux to a sane container'
    assert remuxers[0]['preferedformat'] == 'mkv'


def test_build_opts_ocr_mode_no_remux() -> None:
    """OCR mode must NOT remux — it keeps its exact small output untouched."""
    opts = download_mod._build_ydl_opts('/tmp/clip.mp4', min_height=480, best=False, referer=None)
    assert 'merge_output_format' not in opts
    assert 'postprocessors' not in opts


def test_build_opts_referer_adds_headers() -> None:
    opts = download_mod._build_ydl_opts(
        '/tmp/clip.mp4', min_height=480, best=False, referer='https://oga.pl/'
    )
    assert opts['http_headers']['Referer'] == 'https://oga.pl/'
    assert 'User-Agent' in opts['http_headers']
