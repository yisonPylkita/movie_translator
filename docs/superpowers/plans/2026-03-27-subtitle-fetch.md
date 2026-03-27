# Subtitle Fetch & Media Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download existing Polish/English subtitles from OpenSubtitles before falling back to OCR/translation.

**Architecture:** Two new modules — `identifier/` (multi-signal media identification from filename, container metadata, file hash) and `subtitle_fetch/` (provider-based subtitle download with OpenSubtitles as first backend). Pipeline updated with new decision logic: fetch > embedded > OCR, translate only when needed.

**Tech Stack:** guessit (filename parsing), urllib.request (HTTP), OpenSubtitles REST API v2, ffprobe (container metadata)

**Spec:** `docs/superpowers/specs/2026-03-27-subtitle-fetch-design.md`

---

### Task 1: MediaIdentity type and OpenSubtitles hash

**Files:**
- Create: `movie_translator/identifier/__init__.py`
- Create: `movie_translator/identifier/types.py`
- Create: `movie_translator/identifier/hasher.py`
- Create: `movie_translator/identifier/tests/__init__.py`
- Create: `movie_translator/identifier/tests/test_hasher.py`

- [ ] **Step 1: Write the failing test for OpenSubtitles hash**

The OpenSubtitles hash algorithm: sum all 8-byte little-endian uint64 chunks from the first 64KB and last 64KB, add file size, return as 16-char lowercase hex (overflow wraps at 2^64).

```python
# movie_translator/identifier/tests/test_hasher.py
import struct

import pytest

from movie_translator.identifier.hasher import compute_oshash


class TestComputeOshash:
    def test_computes_hash_for_small_file(self, tmp_path):
        """A file smaller than 128KB: entire file is both 'first 64KB' and 'last 64KB'."""
        f = tmp_path / 'small.bin'
        # Write 16 bytes: two uint64 values (1 and 2)
        data = struct.pack('<QQ', 1, 2)
        f.write_bytes(data)
        result = compute_oshash(f)
        # hash = filesize + sum_of_chunks
        # filesize=16, first_64k covers whole file, last_64k overlaps
        # For files < 128KB, read first min(64KB, filesize) and last min(64KB, filesize)
        # Both reads cover the same data, so chunks are summed twice
        # sum = (1+2)*2 + 16 = 22
        expected = format(22, '016x')
        assert result == expected
        assert len(result) == 16

    def test_returns_16_char_hex_string(self, tmp_path):
        f = tmp_path / 'zeros.bin'
        f.write_bytes(b'\x00' * 256)
        result = compute_oshash(f)
        assert len(result) == 16
        int(result, 16)  # Should not raise — valid hex

    def test_hash_changes_with_content(self, tmp_path):
        f1 = tmp_path / 'a.bin'
        f2 = tmp_path / 'b.bin'
        f1.write_bytes(b'\x01' * 1024)
        f2.write_bytes(b'\x02' * 1024)
        assert compute_oshash(f1) != compute_oshash(f2)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')
        with pytest.raises(ValueError, match='empty'):
            compute_oshash(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/identifier/tests/test_hasher.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create the types and hash implementation**

```python
# movie_translator/identifier/types.py
from typing import NamedTuple


class MediaIdentity(NamedTuple):
    title: str                    # Best-guess title
    year: int | None              # Release year
    season: int | None            # Season number
    episode: int | None           # Episode number
    media_type: str               # 'movie' or 'episode'
    oshash: str                   # OpenSubtitles file hash (16 hex chars)
    file_size: int                # Bytes (needed for OpenSubtitles API)
    raw_filename: str             # Original filename for fallback search
```

```python
# movie_translator/identifier/hasher.py
import struct
from pathlib import Path

CHUNK_SIZE = 65536  # 64KB


def compute_oshash(path: Path) -> str:
    """Compute the OpenSubtitles hash for a video file.

    Algorithm: sum all 8-byte little-endian uint64 values from the first
    64KB and last 64KB of the file, add the file size. Return as 16-char
    lowercase hex. Overflow wraps at 2^64.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f'Cannot hash empty file: {path}')

    hash_val = file_size
    read_size = min(CHUNK_SIZE, file_size)

    with open(path, 'rb') as f:
        # First 64KB
        buf = f.read(read_size)
        hash_val = _sum_chunks(buf, hash_val)

        # Last 64KB
        f.seek(max(0, file_size - CHUNK_SIZE))
        buf = f.read(read_size)
        hash_val = _sum_chunks(buf, hash_val)

    return format(hash_val, '016x')


def _sum_chunks(buf: bytes, initial: int) -> int:
    """Sum all 8-byte little-endian chunks, wrapping at 2^64."""
    # Pad to multiple of 8
    remainder = len(buf) % 8
    if remainder:
        buf += b'\x00' * (8 - remainder)

    val = initial
    for (chunk,) in struct.iter_unpack('<Q', buf):
        val = (val + chunk) & 0xFFFFFFFFFFFFFFFF
    return val
```

```python
# movie_translator/identifier/__init__.py
from .types import MediaIdentity

__all__ = ['MediaIdentity']
```

```python
# movie_translator/identifier/tests/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/identifier/tests/test_hasher.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add movie_translator/identifier/
git commit -m "feat(identifier): add MediaIdentity type and OpenSubtitles hash"
```

---

### Task 2: Filename parser (guessit wrapper)

**Files:**
- Modify: `pyproject.toml` (add guessit dependency)
- Create: `movie_translator/identifier/parser.py`
- Create: `movie_translator/identifier/tests/test_parser.py`

- [ ] **Step 1: Add guessit dependency**

In `pyproject.toml`, add to the `dependencies` list:
```
    "guessit>=3.8",
```

Run: `uv sync`

- [ ] **Step 2: Write the failing test**

```python
# movie_translator/identifier/tests/test_parser.py
from movie_translator.identifier.parser import parse_filename


class TestParseFilename:
    def test_anime_bracket_format(self):
        result = parse_filename(
            '[One Pace][101-102] Reverse Mountain 01 [1080p][En Sub][583096D8].mp4'
        )
        assert result['title'] is not None
        assert isinstance(result['title'], str)

    def test_standard_tv_episode(self):
        result = parse_filename('Breaking.Bad.S01E03.720p.BluRay.x264.mkv')
        assert result['title'] == 'Breaking Bad'
        assert result['season'] == 1
        assert result['episode'] == 3

    def test_movie_with_year(self):
        result = parse_filename('Spirited.Away.2001.1080p.BluRay.mkv')
        assert result['title'] == 'Spirited Away'
        assert result['year'] == 2001
        assert result['media_type'] == 'movie'

    def test_episode_detected_as_episode_type(self):
        result = parse_filename('Naruto.S02E15.720p.mkv')
        assert result['media_type'] == 'episode'

    def test_folder_provides_series_context(self):
        result = parse_filename('Episode 01 [1080p].mkv', folder_name='One Piece')
        # Folder name should provide the series title
        assert result['title'] is not None

    def test_returns_none_for_missing_fields(self):
        result = parse_filename('random_video.mp4')
        assert result['season'] is None
        assert result['year'] is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest movie_translator/identifier/tests/test_parser.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement the parser**

```python
# movie_translator/identifier/parser.py
from guessit import guessit


def parse_filename(
    filename: str,
    folder_name: str | None = None,
) -> dict:
    """Parse a video filename (and optional folder name) into structured metadata.

    Returns dict with keys: title, year, season, episode, media_type.
    Missing fields are None.
    """
    info = guessit(filename)

    title = info.get('title')
    season = info.get('season')
    episode = info.get('episode')
    year = info.get('year')

    # If guessit couldn't extract a title but we have a folder name, try that
    if not title and folder_name:
        folder_info = guessit(folder_name)
        title = folder_info.get('title', folder_name)
        if not season:
            season = folder_info.get('season')

    # Determine media type
    guess_type = info.get('type', 'movie')
    if season is not None or episode is not None:
        media_type = 'episode'
    elif guess_type == 'episode':
        media_type = 'episode'
    else:
        media_type = 'movie'

    return {
        'title': title,
        'year': year,
        'season': season,
        'episode': episode,
        'media_type': media_type,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest movie_translator/identifier/tests/test_parser.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add movie_translator/identifier/parser.py movie_translator/identifier/tests/test_parser.py pyproject.toml
git commit -m "feat(identifier): add filename parser using guessit"
```

---

### Task 3: Container metadata extraction

**Files:**
- Create: `movie_translator/identifier/metadata.py`
- Create: `movie_translator/identifier/tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# movie_translator/identifier/tests/test_metadata.py
from unittest.mock import patch

from movie_translator.identifier.metadata import extract_container_metadata


class TestExtractContainerMetadata:
    def test_extracts_title_from_format_tags(self):
        mock_info = {
            'format': {
                'tags': {'title': 'One Piece Episode 101'},
            },
            'streams': [],
        }
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] == 'One Piece Episode 101'

    def test_returns_none_for_missing_tags(self):
        mock_info = {'format': {}, 'streams': []}
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] is None
        assert result['episode'] is None

    def test_extracts_episode_tag(self):
        mock_info = {
            'format': {
                'tags': {'title': 'One Piece', 'episode_id': '101'},
            },
            'streams': [],
        }
        with patch('movie_translator.identifier.metadata.get_video_info', return_value=mock_info):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] == 'One Piece'

    def test_handles_ffprobe_failure_gracefully(self):
        with patch(
            'movie_translator.identifier.metadata.get_video_info',
            side_effect=Exception('ffprobe failed'),
        ):
            result = extract_container_metadata('/fake/path.mkv')
        assert result['title'] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/identifier/tests/test_metadata.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement metadata extraction**

```python
# movie_translator/identifier/metadata.py
from pathlib import Path

from ..ffmpeg import get_video_info
from ..logging import logger


def extract_container_metadata(video_path: str | Path) -> dict:
    """Extract title and episode metadata from video container tags.

    Returns dict with keys: title, episode. Missing fields are None.
    """
    try:
        info = get_video_info(Path(video_path))
    except Exception as e:
        logger.debug(f'Could not read container metadata: {e}')
        return {'title': None, 'episode': None}

    tags = info.get('format', {}).get('tags', {})

    # Common tag names for title and episode across containers
    title = tags.get('title') or tags.get('TITLE')
    episode = tags.get('episode_id') or tags.get('episode_sort') or tags.get('track')

    return {
        'title': title,
        'episode': episode,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/identifier/tests/test_metadata.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add movie_translator/identifier/metadata.py movie_translator/identifier/tests/test_metadata.py
git commit -m "feat(identifier): add container metadata extraction via ffprobe"
```

---

### Task 4: identify_media() orchestrator

**Files:**
- Modify: `movie_translator/identifier/__init__.py`
- Create: `movie_translator/identifier/identify.py`
- Create: `movie_translator/identifier/tests/test_identify.py`

- [ ] **Step 1: Write the failing test**

```python
# movie_translator/identifier/tests/test_identify.py
from pathlib import Path
from unittest.mock import patch

from movie_translator.identifier import identify_media
from movie_translator.identifier.types import MediaIdentity


class TestIdentifyMedia:
    def test_combines_filename_and_hash(self, tmp_path):
        video = tmp_path / 'Breaking.Bad.S01E03.720p.mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': None, 'episode': None},
        ):
            result = identify_media(video)

        assert isinstance(result, MediaIdentity)
        assert result.title == 'Breaking Bad'
        assert result.season == 1
        assert result.episode == 3
        assert result.media_type == 'episode'
        assert len(result.oshash) == 16
        assert result.file_size == 1024

    def test_container_metadata_overrides_filename(self, tmp_path):
        video = tmp_path / 'random_name.mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': 'The Real Title', 'episode': None},
        ):
            result = identify_media(video)

        assert result.title == 'The Real Title'

    def test_folder_name_fills_missing_title(self, tmp_path):
        folder = tmp_path / 'One Piece'
        folder.mkdir()
        video = folder / 'Episode 05 [1080p].mkv'
        video.write_bytes(b'\x00' * 1024)

        with patch(
            'movie_translator.identifier.identify.extract_container_metadata',
            return_value={'title': None, 'episode': None},
        ):
            result = identify_media(video)

        assert result.title is not None
        assert result.raw_filename == 'Episode 05 [1080p].mkv'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/identifier/tests/test_identify.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement identify_media**

```python
# movie_translator/identifier/identify.py
from pathlib import Path

from ..logging import logger
from .hasher import compute_oshash
from .metadata import extract_container_metadata
from .parser import parse_filename
from .types import MediaIdentity


def identify_media(video_path: Path) -> MediaIdentity:
    """Identify a video file using filename, container metadata, and file hash.

    Combines multiple signals with priority:
    container metadata > filename > folder name.
    """
    filename = video_path.name
    folder_name = video_path.parent.name

    # Signal 1: Parse filename (and folder as fallback context)
    parsed = parse_filename(filename, folder_name=folder_name)

    # Signal 2: Container metadata (overrides filename when present)
    container = extract_container_metadata(video_path)

    # Signal 3: File hash
    try:
        oshash = compute_oshash(video_path)
    except Exception as e:
        logger.warning(f'Could not compute file hash: {e}')
        oshash = ''

    file_size = video_path.stat().st_size

    # Merge: container overrides filename
    title = container.get('title') or parsed.get('title') or filename
    season = parsed.get('season')
    episode = parsed.get('episode')
    year = parsed.get('year')
    media_type = parsed.get('media_type', 'movie')

    # If container has episode info, try to use it
    container_episode = container.get('episode')
    if container_episode and not episode:
        try:
            episode = int(container_episode)
        except (ValueError, TypeError):
            pass

    logger.info(f'Identified: "{title}" (type={media_type}, S{season}E{episode}, year={year})')

    return MediaIdentity(
        title=title,
        year=year,
        season=season,
        episode=episode,
        media_type=media_type,
        oshash=oshash,
        file_size=file_size,
        raw_filename=filename,
    )
```

Update `movie_translator/identifier/__init__.py`:

```python
from .identify import identify_media
from .types import MediaIdentity

__all__ = ['MediaIdentity', 'identify_media']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/identifier/tests/test_identify.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run all identifier tests**

Run: `uv run pytest movie_translator/identifier/tests/ -v`
Expected: PASS (all tests from tasks 1-4)

- [ ] **Step 6: Commit**

```bash
git add movie_translator/identifier/
git commit -m "feat(identifier): add identify_media orchestrator"
```

---

### Task 5: SubtitleProvider protocol and SubtitleMatch type

**Files:**
- Create: `movie_translator/subtitle_fetch/__init__.py`
- Create: `movie_translator/subtitle_fetch/types.py`
- Create: `movie_translator/subtitle_fetch/providers/__init__.py`
- Create: `movie_translator/subtitle_fetch/providers/base.py`
- Create: `movie_translator/subtitle_fetch/tests/__init__.py`

- [ ] **Step 1: Create the types and protocol**

```python
# movie_translator/subtitle_fetch/types.py
from typing import NamedTuple


class SubtitleMatch(NamedTuple):
    language: str           # ISO 639-2B code ('eng', 'pol')
    source: str             # Provider name (e.g., 'opensubtitles')
    subtitle_id: str        # Provider-specific identifier
    release_name: str       # Subtitle release name
    format: str             # File format ('srt', 'ass', 'sub')
    score: float            # Match confidence 0.0-1.0
    hash_match: bool        # True if matched by file hash
```

```python
# movie_translator/subtitle_fetch/providers/base.py
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...identifier.types import MediaIdentity
from ..types import SubtitleMatch


@runtime_checkable
class SubtitleProvider(Protocol):
    """Protocol for subtitle download providers."""

    @property
    def name(self) -> str: ...

    def search(
        self, identity: MediaIdentity, languages: list[str]
    ) -> list[SubtitleMatch]:
        """Search for subtitles. Returns matches sorted by score descending."""
        ...

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        """Download subtitle file. Returns the path written to."""
        ...
```

```python
# movie_translator/subtitle_fetch/__init__.py
from .providers.base import SubtitleProvider
from .types import SubtitleMatch

__all__ = ['SubtitleMatch', 'SubtitleProvider']
```

```python
# movie_translator/subtitle_fetch/providers/__init__.py
```

```python
# movie_translator/subtitle_fetch/tests/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add movie_translator/subtitle_fetch/
git commit -m "feat(subtitle_fetch): add SubtitleMatch type and SubtitleProvider protocol"
```

---

### Task 6: OpenSubtitles provider

**Files:**
- Create: `movie_translator/subtitle_fetch/providers/opensubtitles.py`
- Create: `movie_translator/subtitle_fetch/tests/test_opensubtitles.py`

- [ ] **Step 1: Write the failing tests**

```python
# movie_translator/subtitle_fetch/tests/test_opensubtitles.py
import json
from unittest.mock import MagicMock, patch

import pytest

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider


def _make_identity(**overrides):
    defaults = {
        'title': 'Breaking Bad',
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
                        'files': [
                            {'file_id': 12345, 'file_name': 'subs.srt'}
                        ],
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
                        'files': [
                            {'file_id': 99, 'file_name': 'subs.srt'}
                        ],
                    }
                }
            ]
        }
        provider = OpenSubtitlesProvider(api_key='test-key')
        with patch.object(provider, '_api_request', return_value=api_response):
            matches = provider.search(_make_identity(), ['eng'])

        assert matches[0].hash_match is False
        assert matches[0].score == 0.7

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement OpenSubtitles provider**

```python
# movie_translator/subtitle_fetch/providers/opensubtitles.py
import json
import os
import urllib.request
from pathlib import Path

from ...logging import logger
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

    @property
    def name(self) -> str:
        return 'opensubtitles'

    def search(self, identity, languages: list[str]) -> list[SubtitleMatch]:
        if not self._api_key:
            logger.debug('OpenSubtitles: no API key configured, skipping')
            return []

        os_langs = ','.join(LANG_MAP_TO_OS.get(l, l) for l in languages)
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

        data = self._api_request(
            'POST', '/download', body={'file_id': int(match.subtitle_id)}
        )
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

    def _api_request(self, method: str, endpoint: str, params: dict | None = None, body: dict | None = None) -> dict:
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

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/providers/opensubtitles.py movie_translator/subtitle_fetch/tests/test_opensubtitles.py
git commit -m "feat(subtitle_fetch): add OpenSubtitles REST API v2 provider"
```

---

### Task 7: SubtitleFetcher orchestrator

**Files:**
- Create: `movie_translator/subtitle_fetch/fetcher.py`
- Create: `movie_translator/subtitle_fetch/tests/test_fetcher.py`
- Modify: `movie_translator/subtitle_fetch/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# movie_translator/subtitle_fetch/tests/test_fetcher.py
from pathlib import Path

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.fetcher import SubtitleFetcher
from movie_translator.subtitle_fetch.types import SubtitleMatch


def _make_identity():
    return MediaIdentity(
        title='Test',
        year=None,
        season=1,
        episode=1,
        media_type='episode',
        oshash='0' * 16,
        file_size=1000,
        raw_filename='test.mkv',
    )


class FakeProvider:
    def __init__(self, name, matches):
        self._name = name
        self._matches = matches
        self.downloaded = []

    @property
    def name(self):
        return self._name

    def search(self, identity, languages):
        return [m for m in self._matches if m.language in languages]

    def download(self, match, output_path):
        output_path.write_text(f'subtitle content from {self._name}')
        self.downloaded.append(match)
        return output_path


class TestSubtitleFetcher:
    def test_returns_best_match_per_language(self, tmp_path):
        provider = FakeProvider('fake', [
            SubtitleMatch('eng', 'fake', '1', 'rel', 'srt', 0.7, False),
            SubtitleMatch('eng', 'fake', '2', 'rel', 'srt', 1.0, True),
            SubtitleMatch('pol', 'fake', '3', 'rel', 'srt', 0.8, False),
        ])
        fetcher = SubtitleFetcher([provider])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng', 'pol'], tmp_path)

        assert 'eng' in result
        assert 'pol' in result
        # Should have downloaded the hash-matched English (id=2) and Polish (id=3)
        ids = [m.subtitle_id for m in provider.downloaded]
        assert '2' in ids
        assert '3' in ids

    def test_returns_empty_when_no_matches(self, tmp_path):
        provider = FakeProvider('fake', [])
        fetcher = SubtitleFetcher([provider])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)
        assert result == {}

    def test_tries_multiple_providers(self, tmp_path):
        p1 = FakeProvider('p1', [])
        p2 = FakeProvider('p2', [
            SubtitleMatch('eng', 'p2', '99', 'rel', 'srt', 0.7, False),
        ])
        fetcher = SubtitleFetcher([p1, p2])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)

        assert 'eng' in result
        assert len(p2.downloaded) == 1

    def test_prefers_hash_match_over_query_match(self, tmp_path):
        provider = FakeProvider('fake', [
            SubtitleMatch('eng', 'fake', 'query', 'rel', 'srt', 0.7, False),
            SubtitleMatch('eng', 'fake', 'hash', 'rel', 'srt', 1.0, True),
        ])
        fetcher = SubtitleFetcher([provider])
        result = fetcher.fetch_subtitles(_make_identity(), ['eng'], tmp_path)

        assert provider.downloaded[0].subtitle_id == 'hash'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest movie_translator/subtitle_fetch/tests/test_fetcher.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the fetcher**

```python
# movie_translator/subtitle_fetch/fetcher.py
from pathlib import Path

from ..identifier.types import MediaIdentity
from ..logging import logger
from .types import SubtitleMatch


class SubtitleFetcher:
    """Orchestrates subtitle search across multiple providers."""

    def __init__(self, providers: list):
        self._providers = providers

    def fetch_subtitles(
        self,
        identity: MediaIdentity,
        languages: list[str],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Search all providers and download best subtitle per language.

        Returns {language_code: subtitle_file_path} for successfully downloaded subtitles.
        """
        # Collect all matches from all providers
        all_matches: list[SubtitleMatch] = []
        for provider in self._providers:
            try:
                matches = provider.search(identity, languages)
                all_matches.extend(matches)
                logger.debug(f'{provider.name}: found {len(matches)} matches')
            except Exception as e:
                logger.warning(f'{provider.name} search failed: {e}')

        if not all_matches:
            logger.info('No subtitles found from any provider')
            return {}

        # Pick best match per language (highest score wins, hash_match breaks ties)
        best: dict[str, SubtitleMatch] = {}
        for match in sorted(all_matches, key=lambda m: (m.score, m.hash_match), reverse=True):
            if match.language not in best:
                best[match.language] = match

        # Download best matches
        result: dict[str, Path] = {}
        for lang, match in best.items():
            output_path = output_dir / f'fetched_{lang}.{match.format}'
            try:
                provider = self._find_provider(match.source)
                if provider:
                    provider.download(match, output_path)
                    result[lang] = output_path
                    logger.info(
                        f'Fetched {lang} subtitles: {match.release_name} '
                        f'({"hash" if match.hash_match else "query"} match, {match.source})'
                    )
            except Exception as e:
                logger.warning(f'Failed to download {lang} subtitle: {e}')

        return result

    def _find_provider(self, name: str):
        for p in self._providers:
            if p.name == name:
                return p
        return None
```

Update `movie_translator/subtitle_fetch/__init__.py`:

```python
from .fetcher import SubtitleFetcher
from .providers.base import SubtitleProvider
from .types import SubtitleMatch

__all__ = ['SubtitleFetcher', 'SubtitleMatch', 'SubtitleProvider']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest movie_translator/subtitle_fetch/tests/test_fetcher.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run all subtitle_fetch tests**

Run: `uv run pytest movie_translator/subtitle_fetch/tests/ -v`
Expected: PASS (all tests from tasks 5-7)

- [ ] **Step 6: Commit**

```bash
git add movie_translator/subtitle_fetch/
git commit -m "feat(subtitle_fetch): add SubtitleFetcher orchestrator"
```

---

### Task 8: Pipeline integration — new decision logic

**Files:**
- Modify: `movie_translator/pipeline.py`
- Modify: `movie_translator/main.py`

- [ ] **Step 1: Add --no-fetch CLI flag**

In `movie_translator/main.py`, add after the `--enable-ocr` argument (after line 65):

```python
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Disable online subtitle fetching (use only local extraction/OCR)',
    )
```

- [ ] **Step 2: Update TranslationPipeline to accept enable_fetch parameter**

In `movie_translator/pipeline.py`, update the imports at the top of the file:

```python
import os
import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .identifier import identify_media
from .inpainting import remove_burned_in_subtitles
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available
from .subtitle_fetch import SubtitleFetcher
from .subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .types import OCRResult
from .video import VideoOperations
```

Update `__init__` (add `enable_fetch` parameter):

```python
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_ocr: bool = False,
        enable_fetch: bool = True,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_ocr = enable_ocr
        self.enable_fetch = enable_fetch
        self._extractor = None
        self._video_ops = None
        self._ocr_results: list[OCRResult] | None = None
```

- [ ] **Step 3: Add _try_fetch_subtitles helper method**

Add this method to the `TranslationPipeline` class:

```python
    def _try_fetch_subtitles(
        self, video_path: Path, output_dir: Path
    ) -> dict[str, Path]:
        """Try to fetch Polish and English subtitles from online databases.

        Returns {language_code: subtitle_path} for successfully fetched subtitles.
        """
        if not self.enable_fetch:
            return {}

        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if not api_key:
            logger.info('No OPENSUBTITLES_API_KEY set — skipping subtitle fetch')
            return {}

        try:
            identity = identify_media(video_path)
        except Exception as e:
            logger.warning(f'Media identification failed: {e}')
            return {}

        provider = OpenSubtitlesProvider(api_key=api_key)
        fetcher = SubtitleFetcher([provider])

        try:
            return fetcher.fetch_subtitles(identity, ['eng', 'pol'], output_dir)
        except Exception as e:
            logger.warning(f'Subtitle fetch failed: {e}')
            return {}
```

- [ ] **Step 4: Rewrite process_video_file with new decision logic**

Replace the `process_video_file` method with:

```python
    def process_video_file(self, video_path: Path, temp_dir: Path, dry_run: bool = False) -> bool:
        logger.info(f'Processing: {video_path.name}')
        self._ocr_results = None

        try:
            # Step 1: Try fetching subtitles from online databases
            fetched = self._try_fetch_subtitles(video_path, temp_dir)
            fetched_eng = fetched.get('eng')
            fetched_pol = fetched.get('pol')

            # Step 2: Determine English subtitle source
            if fetched_eng:
                logger.info(f'Using fetched English subtitles: {fetched_eng.name}')
                extracted_ass = fetched_eng
            else:
                extracted_ass = self._extract_subtitles(video_path, temp_dir)
                if not extracted_ass:
                    return False

            # Step 3: Determine Polish subtitle source
            if fetched_pol:
                logger.info(f'Using fetched Polish subtitles: {fetched_pol.name}')
                polish_ass = fetched_pol
                # Still need English dialogue lines for the English track
                dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
                if not dialogue_lines:
                    logger.error('No dialogue lines found in English subtitles')
                    return False
                translated_dialogue = None  # Not needed — we have fetched Polish
            else:
                # Need to translate: parse English dialogue and translate
                logger.info('Parsing dialogue...')
                dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
                if not dialogue_lines:
                    logger.error('No dialogue lines found')
                    return False

                logger.info(f'Translating {len(dialogue_lines)} lines...')
                try:
                    translated_dialogue = translate_dialogue_lines(
                        dialogue_lines, self.device, self.batch_size, self.model
                    )
                    if not translated_dialogue:
                        logger.error('Translation failed')
                        return False
                except Exception as e:
                    logger.error(f'Translation failed: {e}')
                    return False
                polish_ass = None  # Will be created below

            # Step 4: Create subtitle files
            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            logger.info('Creating subtitle files...')
            clean_english_ass = temp_dir / f'{video_path.stem}_english_clean.ass'
            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            if polish_ass is None:
                # Create Polish from translation
                polish_ass = temp_dir / f'{video_path.stem}_polish.ass'
                replace_chars = not fonts_support_polish
                SubtitleProcessor.create_polish_subtitles(
                    extracted_ass, translated_dialogue, polish_ass, replace_chars
                )

            # Step 5: Inpaint burned-in subtitles if detected
            source_video = video_path
            if self._ocr_results:
                logger.info('Removing burned-in subtitles from video...')
                inpainted_video = temp_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
                remove_burned_in_subtitles(
                    video_path,
                    inpainted_video,
                    self._ocr_results,
                    self.device,
                )
                source_video = inpainted_video

            # Step 6: Create final video
            logger.info('Creating video...')
            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(source_video, clean_english_ass, polish_ass, temp_video)
            video_ops.verify_result(temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            logger.info(f'Completed: {video_path.name}')
            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False
```

- [ ] **Step 5: Update main.py to pass enable_fetch to pipeline**

In `main.py`, find where `TranslationPipeline` is constructed and add `enable_fetch`:

```python
            pipeline = TranslationPipeline(
                device=args.device,
                batch_size=args.batch_size,
                model=args.model,
                enable_ocr=args.enable_ocr,
                enable_fetch=not args.no_fetch,
            )
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest movie_translator/ -m "not slow" -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add movie_translator/pipeline.py movie_translator/main.py
git commit -m "feat: integrate subtitle fetch into pipeline with decision logic

New priority chain: fetch > embedded track > OCR.
Translation only runs when Polish subtitles not available.
--no-fetch flag disables online subtitle fetching."
```

---

### Task 9: Final verification and cleanup

**Files:**
- Verify all modules

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest movie_translator/ -m "not slow" -v`
Expected: ALL PASS

- [ ] **Step 2: Run linter**

Run: `uv run ruff check movie_translator/ --fix && uv run ruff format movie_translator/`

- [ ] **Step 3: Verify the pipeline works end-to-end (dry run)**

Run: `uv run movie-translator test_workdir --dry-run --enable-ocr --verbose`
Expected: Pipeline runs through new decision logic (fetch step logs "no API key" and falls back to current behavior)

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint and formatting fixes for subtitle fetch feature"
```
