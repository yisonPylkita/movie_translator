"""NapiProjekt provider — Polish subtitles via hash-based lookup.

NapiProjekt is the largest Polish subtitle database. Subtitles are matched
by computing MD5 of the first 10MB of the video file.
"""

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

from ...identifier.napihash import compute_napiprojekt_hash
from ...logging import logger
from ..types import SubtitleMatch

API_URL = 'http://napiprojekt.pl/unit_napisy/dl.php'
MAGIC_PREFIX = 'iBlm8NTigvXkI6'
USER_AGENT = 'MovieTranslator/1.0'


class NapiProjektProvider:
    """NapiProjekt subtitle provider (Polish subtitles, hash-based)."""

    def __init__(self):
        self._video_path: Path | None = None

    def set_video_path(self, path: str | Path) -> None:
        """Set the video file path (needed for hash computation)."""
        self._video_path = Path(path)

    @property
    def name(self) -> str:
        return 'napiprojekt'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        if 'pol' not in languages:
            return []

        if self._video_path is None:
            logger.debug('NapiProjekt: no video_path set, cannot compute hash')
            return []

        try:
            file_hash = compute_napiprojekt_hash(self._video_path)
        except Exception as e:
            logger.debug(f'NapiProjekt hash failed: {e}')
            return []

        # Probe the API to check if a subtitle actually exists for this hash
        content = self._fetch_subtitle(file_hash)
        if content is None:
            logger.debug(f'NapiProjekt: no subtitle for hash {file_hash[:8]}')
            return []

        # Cache the content so download() doesn't hit the API again
        self._cached_content = content
        self._cached_hash = file_hash

        return [
            SubtitleMatch(
                language='pol',
                source=self.name,
                subtitle_id=file_hash,
                release_name=f'napiprojekt-{file_hash[:8]}',
                format='srt',
                score=0.95,
                hash_match=True,
            )
        ]

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        file_hash = match.subtitle_id

        # Use cached content from search() if available
        if hasattr(self, '_cached_content') and self._cached_hash == file_hash:
            content = self._cached_content
            self._cached_content = None
        else:
            content = self._fetch_subtitle(file_hash)
            if content is None:
                raise RuntimeError(f'NapiProjekt: subtitle not found for hash {file_hash}')

        assert content is not None
        output_path.write_bytes(content)
        logger.info(f'Downloaded subtitle: {output_path.name} (napiprojekt)')
        return output_path

    def _fetch_subtitle(self, file_hash: str) -> bytes | None:
        """Fetch subtitle content from NapiProjekt API. Returns None if not found."""
        token = hashlib.md5((MAGIC_PREFIX + file_hash).encode()).hexdigest()

        params = urllib.parse.urlencode(
            {
                'f': file_hash,
                't': token,
                'v': 'pynapi',
                'l': 'PL',
                'n': file_hash,
                'p': '0',
            }
        ).encode()

        req = urllib.request.Request(
            API_URL,
            data=params,
            headers={'User-Agent': USER_AGENT},
            method='POST',
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()

        if content.startswith(b'NPc0') or len(content) < 10:
            return None

        return content
