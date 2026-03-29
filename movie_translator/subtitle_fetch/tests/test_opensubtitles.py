from unittest.mock import patch

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider


def _make_identity(**overrides):
    defaults = {
        'title': 'Breaking Bad',
        'parsed_title': 'Breaking Bad',
        'year': 2008,
        'season': 1,
        'episode': 3,
        'media_type': 'episode',
        'oshash': 'abc123def456abc0',
        'file_size': 1_000_000,
        'raw_filename': 'Breaking.Bad.S01E03.mkv',
    }
    defaults.update(overrides)
    return MediaIdentity(**defaults)


class TestOpenSubtitlesProvider:
    def test_name_is_opensubtitles(self):
        provider = OpenSubtitlesProvider(api_key='test-key')
        assert provider.name == 'opensubtitles'

    def test_search_returns_empty_without_api_key(self):
        provider = OpenSubtitlesProvider(api_key='')
        result = provider.search(_make_identity(), ['eng'])
        assert result == []

    def test_search_parses_api_response(self):
        api_response = {
            'data': [
                {
                    'attributes': {
                        'language': 'en',
                        'release': 'Breaking.Bad.S01E03.720p',
                        'moviehash_match': True,
                        'files': [{'file_id': 12345, 'file_name': 'subs.srt'}],
                    }
                }
            ]
        }
        provider = OpenSubtitlesProvider(api_key='test-key')
        with patch.object(provider, '_api_request', return_value=api_response):
            matches = provider.search(_make_identity(), ['eng'])

        assert len(matches) == 1
        assert matches[0].language == 'eng'
        assert matches[0].hash_match is True
        assert matches[0].subtitle_id == '12345'
        assert matches[0].score == 1.0

    def test_search_assigns_lower_score_for_query_match(self):
        api_response = {
            'data': [
                {
                    'attributes': {
                        'language': 'en',
                        'release': 'Breaking.Bad.S01E03',
                        'moviehash_match': False,
                        'files': [{'file_id': 99, 'file_name': 'subs.srt'}],
                    }
                }
            ]
        }
        provider = OpenSubtitlesProvider(api_key='test-key')
        with patch.object(provider, '_api_request', return_value=api_response):
            matches = provider.search(_make_identity(), ['eng'])

        assert matches[0].hash_match is False
        assert 0.6 <= matches[0].score < 1.0  # now range-based with release name scoring

    def test_search_uses_imdb_id_when_available(self):
        provider = OpenSubtitlesProvider(api_key='test-key')
        called_params = {}

        def capture_request(method, endpoint, params=None, body=None):
            if params and 'imdb_id' in params:
                called_params.update(params)
            return {'data': []}

        with patch.object(provider, '_api_request', side_effect=capture_request):
            identity = _make_identity(imdb_id='tt0903747')
            provider.search(identity, ['eng'])

        assert called_params.get('imdb_id') == '0903747'

    def test_search_filters_by_requested_languages(self):
        api_response = {
            'data': [
                {
                    'attributes': {
                        'language': 'en',
                        'release': 'subs-en',
                        'moviehash_match': False,
                        'files': [{'file_id': 1, 'file_name': 'en.srt'}],
                    }
                },
                {
                    'attributes': {
                        'language': 'pl',
                        'release': 'subs-pl',
                        'moviehash_match': False,
                        'files': [{'file_id': 2, 'file_name': 'pl.srt'}],
                    }
                },
            ]
        }
        provider = OpenSubtitlesProvider(api_key='test-key')
        with patch.object(provider, '_api_request', return_value=api_response):
            matches = provider.search(_make_identity(), ['pol'])

        assert len(matches) == 1
        assert matches[0].language == 'pol'
