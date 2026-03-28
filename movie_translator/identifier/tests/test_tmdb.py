from unittest.mock import patch

from movie_translator.identifier.tmdb import lookup_tmdb


class TestTmdbLookup:
    def test_returns_ids_for_movie(self):
        api_response = {
            'results': [
                {
                    'id': 1396,
                    'title': 'Breaking Bad',
                    'release_date': '2008-01-20',
                }
            ]
        }
        detail_response = {'imdb_id': 'tt0903747'}
        with patch(
            'movie_translator.identifier.tmdb._tmdb_request',
            side_effect=[api_response, detail_response],
        ):
            result = lookup_tmdb('Breaking Bad', year=2008, media_type='movie')

        assert result is not None
        assert result['tmdb_id'] == 1396
        assert result['imdb_id'] == 'tt0903747'

    def test_returns_none_without_api_key(self):
        with patch('movie_translator.identifier.tmdb._get_api_key', return_value=''):
            result = lookup_tmdb('Test', year=None, media_type='movie')
        assert result is None

    def test_returns_none_on_empty_results(self):
        api_response = {'results': []}
        with patch('movie_translator.identifier.tmdb._tmdb_request', return_value=api_response):
            result = lookup_tmdb('Nonexistent Movie', year=2020, media_type='movie')
        assert result is None

    def test_returns_none_on_network_error(self):
        with patch(
            'movie_translator.identifier.tmdb._tmdb_request', side_effect=Exception('timeout')
        ):
            result = lookup_tmdb('Test', year=None, media_type='movie')
        assert result is None

    def test_searches_tv_for_episodes(self):
        api_response = {
            'results': [
                {
                    'id': 1399,
                    'name': 'Breaking Bad',
                    'first_air_date': '2008-01-20',
                }
            ]
        }
        called_with = {}

        def capture_request(endpoint, params):
            if 'search' in endpoint:
                called_with['endpoint'] = endpoint
                return api_response
            return {}  # detail endpoint

        with patch('movie_translator.identifier.tmdb._tmdb_request', side_effect=capture_request):
            result = lookup_tmdb('Breaking Bad', year=2008, media_type='episode')

        assert '/search/tv' in called_with['endpoint']
        assert result['tmdb_id'] == 1399
