"""AnimeSub.info provider — Polish anime subtitles.

Scrapes animesub.info (a Polish fansubbing community) for subtitle files.
Search by anime title, download as ZIP, extract subtitle files.
"""

import io
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from ...logging import logger
from ..types import SubtitleMatch

BASE_URL = 'http://animesub.info'
USER_AGENT = 'Mozilla/5.0 (compatible; MovieTranslator/1.0)'


class _ResultParser(HTMLParser):
    """Parse animesub.info search results HTML into structured entries."""

    def __init__(self):
        super().__init__()
        self.entries: list[dict] = []
        self._current_entry: dict | None = None
        self._table_depth = 0  # Track nested tables
        self._in_td = False
        self._td_attrs: dict = {}
        self._td_text = ''

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)

        if tag == 'table':
            if attr_dict.get('class') == 'Napisy':
                self._current_entry = {}
                self._table_depth = 1
            elif self._current_entry is not None:
                self._table_depth += 1

        if tag == 'td':
            self._in_td = True
            self._td_attrs = attr_dict
            self._td_text = ''

        # Extract id and sh from hidden form inputs
        if tag == 'input' and attr_dict.get('type') == 'hidden':
            name = attr_dict.get('name', '')
            value = attr_dict.get('value', '')
            if name in ('id', 'sh') and self._current_entry is not None:
                self._current_entry[name] = value

    def handle_endtag(self, tag):
        if tag == 'td' and self._in_td:
            self._in_td = False
            text = self._td_text.strip()

            if self._current_entry is not None:
                # Title cell: width="45%"
                if self._td_attrs.get('width') == '45%' and 'title' not in self._current_entry:
                    self._current_entry['title'] = text
                # Format cell: width="20%"
                elif self._td_attrs.get('width') == '20%' and 'format' not in self._current_entry:
                    self._current_entry['format'] = text

        if tag == 'table' and self._current_entry is not None:
            self._table_depth -= 1
            if self._table_depth <= 0:
                # Only close entry when outermost Napisy table closes
                if 'id' in self._current_entry and 'sh' in self._current_entry:
                    self.entries.append(self._current_entry)
                self._current_entry = None

    def handle_data(self, data):
        if self._in_td:
            self._td_text += data


class AnimeSubProvider:
    """animesub.info subtitle provider (Polish anime subtitles)."""

    @property
    def name(self) -> str:
        return 'animesub'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        if 'pol' not in languages:
            return []

        title = identity.title
        if not title:
            return []

        matches: list[SubtitleMatch] = []

        # Try English title first, then original
        for title_type in ('en', 'org'):
            try:
                entries = self._search_page(title, title_type)
                for entry in entries:
                    fmt = entry.get('format', '').lower()
                    ext = 'ass' if 'ssa' in fmt or 'ass' in fmt else 'srt'
                    matches.append(
                        SubtitleMatch(
                            language='pol',
                            source=self.name,
                            subtitle_id=f'{entry["id"]}:{entry["sh"]}',
                            release_name=entry.get('title', ''),
                            format=ext,
                            score=0.6,  # Lower than OpenSubtitles hash/query matches
                            hash_match=False,
                        )
                    )
                if matches:
                    break  # Found results, no need to try other title type
            except Exception as e:
                logger.debug(f'AnimeSub search failed ({title_type}): {e}')

        return matches

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        sub_id, sh = match.subtitle_id.split(':', 1)

        data = urllib.parse.urlencode({'id': sub_id, 'sh': sh}).encode()
        req = urllib.request.Request(
            f'{BASE_URL}/sciagnij.php',
            data=data,
            headers={'User-Agent': USER_AGENT},
            method='POST',
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()

        # Extract subtitle file from ZIP
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            sub_files = [
                n for n in zf.namelist() if n.lower().endswith(('.srt', '.ass', '.ssa', '.sub'))
            ]
            if not sub_files:
                raise RuntimeError(f'No subtitle file found in ZIP from AnimeSub (id={sub_id})')

            subtitle_content = zf.read(sub_files[0])
            output_path.write_bytes(subtitle_content)

        logger.info(f'Downloaded subtitle: {output_path.name} (animesub.info)')
        return output_path

    def _search_page(self, title: str, title_type: str = 'en', page: int = 0) -> list[dict]:
        """Fetch and parse one page of search results."""
        query = urllib.request.quote(title)
        url = f'{BASE_URL}/szukaj.php?szukane={query}&pTitle={title_type}&od={page}'

        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('iso-8859-2', errors='replace')

        parser = _ResultParser()
        parser.feed(html)
        return parser.entries
