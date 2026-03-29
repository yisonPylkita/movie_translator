"""AnimeSub.info provider — Polish anime subtitles.

Scrapes animesub.info (a Polish fansubbing community) for subtitle files.
Search by anime title, download as ZIP, extract subtitle files.
"""

import http.cookiejar
import io
import re
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from ...logging import logger
from ..types import SubtitleMatch

BASE_URL = 'http://animesub.info'
USER_AGENT = 'Mozilla/5.0 (compatible; MovieTranslator/1.0)'


def _extract_season_from_title(base_title: str, entry_title: str) -> int | None:
    """Infer the season number from an AnimeSub entry title.

    Anime season conventions on AnimeSub:
      "Title ep01"              → Season 1 (no suffix)
      "Title 2 ep08"           → Season 2 (number suffix)
      "Title 3 ep01"           → Season 3
      "Title S2 ep01-10"       → Season 2 (explicit S-prefix)
      "Title OVA ep01"         → None (special, not a season)
      "Title 3: Bonus Stage"   → None (special)

    Args:
        base_title: The anime title we searched for (e.g., "Kono Subarashii...")
        entry_title: The full entry title from AnimeSub results

    Returns:
        Inferred season number, or None if it's a special/OVA/movie.
    """
    # Strip the base title to get the suffix
    # Case-insensitive prefix removal
    suffix = entry_title
    if entry_title.lower().startswith(base_title.lower()):
        suffix = entry_title[len(base_title) :].strip()

    # If no suffix before the episode marker → Season 1
    if not suffix or suffix.lower().startswith('ep'):
        return 1

    # Check for specials first — these are NOT numbered seasons
    specials = ('ova', 'movie', 'film', 'special', 'bonus', 'recap')
    suffix_lower = suffix.lower()
    if any(s in suffix_lower for s in specials):
        return None

    # "S2 ep..." or "S3 ep..."
    s_match = re.match(r's(\d+)\b', suffix_lower)
    if s_match:
        return int(s_match.group(1))

    # "2 ep..." or "3 ep..." or "2: Something ep..."
    num_match = re.match(r'(\d+)\b', suffix)
    if num_match:
        return int(num_match.group(1))

    # Unrecognized suffix — don't assume a season
    return None


def _entry_matches(title: str, base_title: str, season: int | None, episode: int) -> bool:
    """Check if an AnimeSub result matches the requested season and episode.

    Args:
        title: The entry title from AnimeSub results
        base_title: The anime title we searched for
        season: Requested season (None = don't filter by season)
        episode: Requested episode number
    """
    # Check episode number
    patterns = re.findall(r'(?:ep|episode\s*|e)(\d+)', title.lower())
    if not patterns:
        return False
    if not any(int(n) == episode for n in patterns):
        return False

    # Check season (if requested)
    if season is not None:
        entry_season = _extract_season_from_title(base_title, title)
        # Reject if the entry is a different season OR a special/OVA
        if entry_season != season:
            return False

    return True


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

    def __init__(self):
        # Cookie jar shared across search and download to maintain session
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )

    @property
    def name(self) -> str:
        return 'animesub'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        if 'pol' not in languages:
            return []

        # Prefer parsed_title (from filename) over container metadata title
        # Container titles are often polluted with release info
        title = getattr(identity, 'parsed_title', None) or identity.title
        if not title:
            return []

        episode = identity.episode
        matches: list[SubtitleMatch] = []

        # Try English title first, then original
        for title_type in ('en', 'org'):
            try:
                entries = self._search_page(title, title_type)
                for entry in entries:
                    entry_title = entry.get('title', '')

                    # Filter by season and episode
                    if episode is not None:
                        if not _entry_matches(entry_title, title, identity.season, episode):
                            continue

                    fmt = entry.get('format', '').lower()
                    ext = 'ass' if 'ssa' in fmt or 'ass' in fmt else 'srt'
                    matches.append(
                        SubtitleMatch(
                            language='pol',
                            source=self.name,
                            subtitle_id=f'{entry["id"]}:{entry["sh"]}',
                            release_name=entry_title,
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

        with self._opener.open(req, timeout=5) as resp:
            content_type = resp.headers.get('Content-Type', '')
            zip_bytes = resp.read()

        if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
            raise RuntimeError(
                f'AnimeSub returned non-ZIP response (content-type: {content_type}, '
                f'{len(zip_bytes)} bytes) for subtitle id={sub_id}'
            )

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
        with self._opener.open(req, timeout=5) as resp:
            html = resp.read().decode('iso-8859-2', errors='replace')

        parser = _ResultParser()
        parser.feed(html)
        return parser.entries
