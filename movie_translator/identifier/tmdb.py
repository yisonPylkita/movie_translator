"""TMDB API integration for enriching media identity with external IDs.

Uses the TMDB API v3 (free tier). Requires TMDB_API_KEY env var.
Only called when the env var is set — completely optional.
"""

import json
import os
import urllib.parse
import urllib.request

from ..logging import logger


def _get_api_key() -> str:
    return os.environ.get('TMDB_API_KEY', '')


def _tmdb_request(endpoint: str, params: dict) -> dict:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError('TMDB_API_KEY is not set')
    params['api_key'] = api_key
    url = f'https://api.themoviedb.org/3{endpoint}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'MovieTranslator/1.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def lookup_tmdb(
    title: str,
    year: int | None,
    media_type: str,
) -> dict | None:
    """Look up a title on TMDB and return {tmdb_id, imdb_id (if available)}.

    Returns None if API key is not set, no results found, or on error.
    """
    try:
        if media_type == 'episode':
            endpoint = '/search/tv'
            params = {'query': title}
            if year:
                params['first_air_date_year'] = str(year)
        else:
            endpoint = '/search/movie'
            params = {'query': title}
            if year:
                params['year'] = str(year)

        data = _tmdb_request(endpoint, params)
        results = data.get('results', [])
        if not results:
            return None

        best = results[0]
        tmdb_id = best.get('id')
        if not tmdb_id:
            return None

        result = {'tmdb_id': tmdb_id}

        # Try to get IMDB ID from the detail endpoint
        try:
            detail_endpoint = (
                f'/tv/{tmdb_id}/external_ids' if media_type == 'episode' else f'/movie/{tmdb_id}'
            )
            detail = _tmdb_request(detail_endpoint, {})
            imdb_id = detail.get('imdb_id')
            if imdb_id:
                result['imdb_id'] = imdb_id
        except Exception:
            pass  # IMDB ID is nice-to-have, not critical

        return result

    except Exception as e:
        logger.debug(f'TMDB lookup failed: {e}')
        return None
