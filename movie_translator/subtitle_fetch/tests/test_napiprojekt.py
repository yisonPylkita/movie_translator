from unittest.mock import patch

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.napiprojekt import NapiProjektProvider


def _make_identity(**overrides):
    defaults = {
        'title': 'Test Movie',
        'parsed_title': 'Test Movie',
        'year': 2020,
        'season': None,
        'episode': None,
        'media_type': 'movie',
        'oshash': '0' * 16,
        'file_size': 1_000_000,
        'raw_filename': 'test.mkv',
    }
    defaults.update(overrides)
    return MediaIdentity(**defaults)  # type: ignore[invalid-argument-type]  # ty:ignore[invalid-argument-type]


class TestNapiProjektProvider:
    def test_name(self):
        assert NapiProjektProvider().name == 'napiprojekt'

    def test_search_only_supports_polish(self):
        provider = NapiProjektProvider()
        provider.set_video_path('/fake/path.mkv')
        with patch(
            'movie_translator.subtitle_fetch.providers.napiprojekt.compute_napiprojekt_hash',
            return_value='abc123',
        ):
            matches = provider.search(_make_identity(), ['eng'])
        assert matches == []

    def test_search_returns_match_when_subtitle_exists(self):
        """search() probes the API and only returns a match if subtitle exists."""
        provider = NapiProjektProvider()
        provider.set_video_path('/fake/path.mkv')
        with (
            patch(
                'movie_translator.subtitle_fetch.providers.napiprojekt.compute_napiprojekt_hash',
                return_value='abc123',
            ),
            patch.object(provider, '_fetch_subtitle', return_value=b'subtitle content here'),
        ):
            matches = provider.search(_make_identity(), ['pol'])
        assert len(matches) == 1
        assert matches[0].language == 'pol'
        assert matches[0].source == 'napiprojekt'
        assert matches[0].hash_match is True
        assert matches[0].score == 0.95
        assert matches[0].subtitle_id == 'abc123'

    def test_search_returns_empty_when_subtitle_not_found(self):
        """search() returns empty if API says no subtitle for this hash."""
        provider = NapiProjektProvider()
        provider.set_video_path('/fake/path.mkv')
        with (
            patch(
                'movie_translator.subtitle_fetch.providers.napiprojekt.compute_napiprojekt_hash',
                return_value='abc123',
            ),
            patch.object(provider, '_fetch_subtitle', return_value=None),
        ):
            matches = provider.search(_make_identity(), ['pol'])
        assert matches == []

    def test_search_returns_empty_when_hash_fails(self):
        provider = NapiProjektProvider()
        provider.set_video_path('/fake/path.mkv')
        with patch(
            'movie_translator.subtitle_fetch.providers.napiprojekt.compute_napiprojekt_hash',
            side_effect=ValueError('empty file'),
        ):
            matches = provider.search(_make_identity(), ['pol'])
        assert matches == []

    def test_search_requires_video_path(self):
        """NapiProjekt needs the actual video file to compute its hash."""
        provider = NapiProjektProvider()
        matches = provider.search(_make_identity(), ['pol'])
        assert matches == []
