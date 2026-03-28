import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from ...logging import logger
from ..rate_limiter import RateLimiter
from ..types import SubtitleMatch

API_BASE = 'https://api.opensubtitles.com/api/v1'
USER_AGENT = 'MovieTranslator v1.0'

# OpenSubtitles uses ISO 639-1 (2-letter) codes, we use ISO 639-2B (3-letter)
LANG_MAP_TO_OS = {'eng': 'en', 'pol': 'pl', 'jpn': 'ja'}
LANG_MAP_FROM_OS = {v: k for k, v in LANG_MAP_TO_OS.items()}


class OpenSubtitlesProvider:
    """OpenSubtitles.com REST API v2 provider."""

    def __init__(
        self,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self._api_key = api_key or os.environ.get('OPENSUBTITLES_API_KEY', '')
        self._username = username or os.environ.get('OPENSUBTITLES_USERNAME', '')
        self._password = password or os.environ.get('OPENSUBTITLES_PASSWORD', '')
        self._token: str | None = None
        self._rate_limiter = RateLimiter(min_interval=0.25)

    @property
    def name(self) -> str:
        return 'opensubtitles'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        if not self._api_key:
            logger.debug('OpenSubtitles: no API key configured, skipping')
            return []

        os_langs = ','.join(LANG_MAP_TO_OS.get(lang, lang) for lang in languages)
        matches: list[SubtitleMatch] = []

        # Strategy 1: Search by file hash (most accurate)
        if identity.oshash:
            params = {
                'moviehash': identity.oshash,
                'languages': os_langs,
            }
            try:
                data = self._api_request('GET', '/subtitles', params)
                matches = self._parse_results(data, languages)
            except Exception as e:
                logger.debug(f'OpenSubtitles hash search failed: {e}')

        # Strategy 2: Search by query (if hash gave no results)
        if not matches:
            params = {
                'query': identity.title,
                'languages': os_langs,
            }
            if identity.season is not None:
                params['season_number'] = str(identity.season)
            if identity.episode is not None:
                params['episode_number'] = str(identity.episode)
            if identity.media_type == 'movie' and identity.year:
                params['year'] = str(identity.year)

            try:
                data = self._api_request('GET', '/subtitles', params)
                matches = self._parse_results(data, languages)
            except Exception as e:
                logger.debug(f'OpenSubtitles query search failed: {e}')

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        self._ensure_logged_in()

        data = self._api_request('POST', '/download', body={'file_id': int(match.subtitle_id)})
        download_link = data.get('link', '')
        if not download_link:
            raise RuntimeError(f'No download link in response for subtitle {match.subtitle_id}')

        req = urllib.request.Request(download_link)
        with urllib.request.urlopen(req, timeout=30) as resp:
            output_path.write_bytes(resp.read())

        logger.info(f'Downloaded subtitle: {output_path.name} ({match.source})')
        return output_path

    def _parse_results(self, data: dict, languages: list[str]) -> list[SubtitleMatch]:
        matches = []
        for item in data.get('data', []):
            attrs = item.get('attributes', {})
            os_lang = attrs.get('language', '')
            lang_3 = LANG_MAP_FROM_OS.get(os_lang, os_lang)

            if lang_3 not in languages:
                continue

            files = attrs.get('files', [])
            if not files:
                continue

            file_info = files[0]
            file_name = file_info.get('file_name', '')
            ext = file_name.rsplit('.', 1)[-1] if '.' in file_name else 'srt'
            is_hash = attrs.get('moviehash_match', False)

            matches.append(
                SubtitleMatch(
                    language=lang_3,
                    source=self.name,
                    subtitle_id=str(file_info.get('file_id', '')),
                    release_name=attrs.get('release', ''),
                    format=ext,
                    score=1.0 if is_hash else 0.7,
                    hash_match=is_hash,
                )
            )
        return matches

    def _api_request(
        self, method: str, endpoint: str, params: dict | None = None, body: dict | None = None
    ) -> dict:
        self._rate_limiter.wait()

        url = f'{API_BASE}{endpoint}'
        if params:
            query = '&'.join(f'{k}={urllib.request.quote(str(v))}' for k, v in params.items())
            url = f'{url}?{query}'

        headers = {
            'Api-Key': self._api_key,
            'User-Agent': USER_AGENT,
            'Content-Type': 'application/json',
        }
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Parse rate limit headers
                resp_headers = dict(resp.headers.items())
                self._rate_limiter.update_from_headers(resp_headers)
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get('Retry-After')
                delay = float(retry_after) if retry_after else 5.0
                self._rate_limiter.record_429(retry_after=delay)
                raise
            if e.code == 406:
                logger.warning('OpenSubtitles daily download quota exceeded')
                raise
            raise

    def _ensure_logged_in(self):
        if self._token:
            return
        if not self._username or not self._password:
            raise RuntimeError(
                'OpenSubtitles download requires OPENSUBTITLES_USERNAME and '
                'OPENSUBTITLES_PASSWORD environment variables'
            )
        data = self._api_request(
            'POST', '/login', body={'username': self._username, 'password': self._password}
        )
        self._token = data.get('token', '')
        if not self._token:
            raise RuntimeError('OpenSubtitles login failed: no token returned')
