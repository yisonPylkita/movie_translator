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

    def set_video_path(self, path: Path) -> None:
        """Set the video file path (needed for hash computation)."""
        self._video_path = path

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

        # Return the match optimistically — download() will raise if not found.
        # This avoids downloading the subtitle twice (once to check, once to save).
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

        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()

        if content.startswith(b'NPc0') or len(content) < 10:
            raise RuntimeError(f'NapiProjekt: subtitle not found for hash {file_hash}')

        output_path.write_bytes(content)
        logger.info(f'Downloaded subtitle: {output_path.name} (napiprojekt)')
        return output_path
