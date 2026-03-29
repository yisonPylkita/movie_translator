"""Podnapisi.net provider — multilingual subtitle search via REST/XML API."""

import io
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import defusedxml.ElementTree as ET

from ...logging import logger
from ..types import SubtitleMatch

API_BASE = 'https://www.podnapisi.net'
SEARCH_URL = f'{API_BASE}/subtitles/search/old'
USER_AGENT = 'MovieTranslator/1.0'

# Podnapisi uses its own language IDs
LANG_TO_PODNAPI = {'eng': '2', 'pol': '23', 'jpn': '11'}
LANG_FROM_PODNAPI = {'en': 'eng', 'pl': 'pol', 'ja': 'jpn'}


class PodnapisiProvider:
    """Podnapisi.net subtitle provider."""

    @property
    def name(self) -> str:
        return 'podnapisi'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        podnapi_langs = [LANG_TO_PODNAPI[lang] for lang in languages if lang in LANG_TO_PODNAPI]
        if not podnapi_langs:
            return []

        params = {
            'sXML': '1',
            'sK': identity.parsed_title or identity.title,
            'sJ': ','.join(podnapi_langs),
        }
        if identity.season is not None:
            params['sS'] = str(identity.season)
        if identity.episode is not None:
            params['sE'] = str(identity.episode)
        if identity.year:
            params['sY'] = str(identity.year)

        url = f'{SEARCH_URL}?{urllib.parse.urlencode(params)}'

        try:
            xml_text = self._fetch_xml(url)
        except Exception as e:
            logger.debug(f'Podnapisi search failed: {e}')
            return []

        return self._parse_results(xml_text, languages)

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        url = f'{API_BASE}/subtitles/{match.subtitle_id}/download'
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})

        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()

        # Podnapisi returns a ZIP file
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                sub_files = [
                    n for n in zf.namelist() if n.lower().endswith(('.srt', '.ass', '.ssa', '.sub'))
                ]
                if not sub_files:
                    raise RuntimeError(
                        f'No subtitle file in Podnapisi ZIP (id={match.subtitle_id})'
                    )
                output_path.write_bytes(zf.read(sub_files[0]))
        else:
            # Some results return raw subtitle content
            output_path.write_bytes(content)

        logger.info(f'Downloaded subtitle: {output_path.name} (podnapisi)')
        return output_path

    def _fetch_xml(self, url: str) -> str:
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode('utf-8')

    def _parse_results(self, xml_text: str, languages: list[str]) -> list[SubtitleMatch]:
        matches = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.debug(f'Podnapisi XML parse error: {e}')
            return []

        for sub_el in root.findall('.//subtitle'):
            sub_id = sub_el.findtext('id', '')
            lang_code = sub_el.findtext('language', '')
            lang_3 = LANG_FROM_PODNAPI.get(lang_code, lang_code)
            release = sub_el.findtext('release', '')

            if lang_3 not in languages:
                continue

            matches.append(
                SubtitleMatch(
                    language=lang_3,
                    source=self.name,
                    subtitle_id=sub_id,
                    release_name=release,
                    format='srt',
                    score=0.65,
                    hash_match=False,
                )
            )

        return matches
