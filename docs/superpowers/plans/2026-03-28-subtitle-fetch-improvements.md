# Subtitle Fetch Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rate limiting, two new providers (Podnapisi, NapiProjekt), TMDB enrichment, and smarter scoring to the subtitle fetching pipeline.

**Architecture:** Keep existing `SubtitleProvider` protocol. Add a shared rate limiter utility. New providers follow the same pattern as `opensubtitles.py` and `animesub.py` (stdlib `urllib` only, no new dependencies). Extend `MediaIdentity` with optional IMDB/TMDB IDs. Enhance scoring from binary 1.0/0.7 to multi-factor weights.

**Tech Stack:** Python stdlib only (urllib, hashlib, xml.etree, struct). No new dependencies.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `movie_translator/subtitle_fetch/rate_limiter.py` | Thread-safe rate limiter with header parsing and backoff |
| Create | `movie_translator/subtitle_fetch/tests/test_rate_limiter.py` | Rate limiter tests |
| Modify | `movie_translator/subtitle_fetch/providers/opensubtitles.py` | Integrate rate limiter |
| Modify | `movie_translator/subtitle_fetch/tests/test_opensubtitles.py` | Test rate limiter integration |
| Create | `movie_translator/subtitle_fetch/providers/podnapisi.py` | Podnapisi.net provider |
| Create | `movie_translator/subtitle_fetch/tests/test_podnapisi.py` | Podnapisi tests |
| Create | `movie_translator/identifier/napihash.py` | NapiProjekt MD5 hash (first 10MB) |
| Create | `movie_translator/identifier/tests/test_napihash.py` | NapiProjekt hash tests |
| Create | `movie_translator/subtitle_fetch/providers/napiprojekt.py` | NapiProjekt provider |
| Create | `movie_translator/subtitle_fetch/tests/test_napiprojekt.py` | NapiProjekt provider tests |
| Modify | `movie_translator/identifier/types.py` | Add `imdb_id`, `tmdb_id` to MediaIdentity |
| Create | `movie_translator/identifier/tmdb.py` | TMDB API lookup for enrichment |
| Create | `movie_translator/identifier/tests/test_tmdb.py` | TMDB enrichment tests |
| Modify | `movie_translator/identifier/identify.py` | Call TMDB enrichment |
| Modify | `movie_translator/subtitle_fetch/providers/opensubtitles.py` | Use IMDB/TMDB IDs in search |
| Modify | `movie_translator/subtitle_fetch/scoring.py` | Release name token matching |
| Modify | `movie_translator/pipeline.py` | Wire new providers |

---

### Task 1: Rate Limiter Module

**Files:**
- Create: `movie_translator/subtitle_fetch/rate_limiter.py`
- Create: `movie_translator/subtitle_fetch/tests/test_rate_limiter.py`

- [ ] **Step 1: Write failing tests for rate limiter**

```python
# movie_translator/subtitle_fetch/tests/test_rate_limiter.py
import time

from movie_translator.subtitle_fetch.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_call_allowed_immediately(self):
        limiter = RateLimiter(min_interval=0.5)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_second_call_delayed(self):
        limiter = RateLimiter(min_interval=0.3)
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25  # allow small tolerance

    def test_update_from_headers_respects_remaining(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.update_from_headers({
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': '1',
        })
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.8  # should sleep ~1s

    def test_backoff_on_429(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.record_429(retry_after=0.3)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25

    def test_no_delay_when_remaining_is_high(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.update_from_headers({
            'X-RateLimit-Remaining': '40',
            'X-RateLimit-Reset': '60',
        })
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_rate_limiter.py -v`
Expected: FAIL (ImportError - module does not exist)

- [ ] **Step 3: Implement rate limiter**

```python
# movie_translator/subtitle_fetch/rate_limiter.py
"""Thread-safe rate limiter with HTTP header awareness.

Designed for APIs that return X-RateLimit-* headers (e.g., OpenSubtitles).
"""

import threading
import time


class RateLimiter:
    """Rate limiter that enforces minimum intervals and respects API rate limit headers."""

    def __init__(self, min_interval: float = 0.25):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request: float = 0.0
        self._blocked_until: float = 0.0

    def wait(self) -> None:
        """Block until it is safe to make the next request."""
        with self._lock:
            now = time.monotonic()

            # Respect 429 / header-based block
            if now < self._blocked_until:
                delay = self._blocked_until - now
                time.sleep(delay)

            # Respect minimum interval
            elapsed = time.monotonic() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request = time.monotonic()

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Parse X-RateLimit-* headers to adjust pacing."""
        remaining = headers.get('X-RateLimit-Remaining')
        reset = headers.get('X-RateLimit-Reset')

        if remaining is not None and reset is not None:
            try:
                remaining_int = int(remaining)
                reset_secs = float(reset)
            except (ValueError, TypeError):
                return

            if remaining_int <= 1 and reset_secs > 0:
                with self._lock:
                    self._blocked_until = time.monotonic() + reset_secs

    def record_429(self, retry_after: float | None = None) -> None:
        """Record a 429 response. Back off for retry_after seconds (default: 5s)."""
        delay = retry_after if retry_after is not None else 5.0
        with self._lock:
            self._blocked_until = time.monotonic() + delay
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_rate_limiter.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/rate_limiter.py movie_translator/subtitle_fetch/tests/test_rate_limiter.py
git commit -m "feat(rate-limiter): add thread-safe rate limiter with header parsing"
```

---

### Task 2: Integrate Rate Limiter into OpenSubtitles Provider

**Files:**
- Modify: `movie_translator/subtitle_fetch/providers/opensubtitles.py`
- Modify: `movie_translator/subtitle_fetch/tests/test_opensubtitles.py`

- [ ] **Step 1: Write failing test for rate limiting integration**

Add to `test_opensubtitles.py`:

```python
def test_api_request_calls_rate_limiter(self):
    from unittest.mock import MagicMock, patch

    provider = OpenSubtitlesProvider(api_key='test-key')
    mock_limiter = MagicMock()
    provider._rate_limiter = mock_limiter

    api_response = {'data': []}
    with patch.object(provider, '_api_request', wraps=provider._api_request) as wrapped:
        # We need to mock urlopen to avoid real HTTP
        with patch('movie_translator.subtitle_fetch.providers.opensubtitles.urllib.request.urlopen') as mock_urlopen:
            import io
            import json
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(api_response).encode()
            mock_resp.headers = {}
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            provider._api_request('GET', '/subtitles', {'languages': 'en'})

    mock_limiter.wait.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py::TestOpenSubtitlesProvider::test_api_request_calls_rate_limiter -v`
Expected: FAIL

- [ ] **Step 3: Integrate rate limiter into OpenSubtitles provider**

Modify `movie_translator/subtitle_fetch/providers/opensubtitles.py`:

Add import at top:
```python
from ..rate_limiter import RateLimiter
```

In `__init__`, add:
```python
self._rate_limiter = RateLimiter(min_interval=0.25)
```

In `_api_request`, add rate limiter calls:
```python
def _api_request(self, method, endpoint, params=None, body=None):
    self._rate_limiter.wait()

    url = f'{API_BASE}{endpoint}'
    # ... existing url building ...

    # ... existing request building ...

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Parse rate limit headers
            headers = {k: v for k, v in resp.headers.items()}
            self._rate_limiter.update_from_headers(headers)
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
```

- [ ] **Step 4: Run all OpenSubtitles tests**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/providers/opensubtitles.py movie_translator/subtitle_fetch/tests/test_opensubtitles.py
git commit -m "feat(opensubtitles): integrate rate limiter with 429/406 handling"
```

---

### Task 3: Podnapisi Provider

**Files:**
- Create: `movie_translator/subtitle_fetch/providers/podnapisi.py`
- Create: `movie_translator/subtitle_fetch/tests/test_podnapisi.py`

- [ ] **Step 1: Write failing tests**

```python
# movie_translator/subtitle_fetch/tests/test_podnapisi.py
from unittest.mock import patch, MagicMock

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.podnapisi import PodnapisiProvider


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


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<results>
  <pagination><results>2</results></pagination>
  <subtitle>
    <id>12345</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.720p.BluRay</release>
    <language>en</language>
    <flags>0</flags>
    <rating>4.8</rating>
    <downloads>5432</downloads>
  </subtitle>
  <subtitle>
    <id>67890</id>
    <title>Breaking Bad S01E03</title>
    <release>Breaking.Bad.S01E03.1080p.WEB</release>
    <language>pl</language>
    <flags>0</flags>
    <rating>4.5</rating>
    <downloads>1234</downloads>
  </subtitle>
</results>"""


class TestPodnapisiProvider:
    def test_name(self):
        assert PodnapisiProvider().name == 'podnapisi'

    def test_search_parses_xml_response(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', return_value=SAMPLE_XML):
            matches = provider.search(_make_identity(), ['eng', 'pol'])

        assert len(matches) == 2
        langs = {m.language for m in matches}
        assert langs == {'eng', 'pol'}
        assert matches[0].source == 'podnapisi'

    def test_search_filters_by_language(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', return_value=SAMPLE_XML):
            matches = provider.search(_make_identity(), ['pol'])

        assert len(matches) == 1
        assert matches[0].language == 'pol'

    def test_search_includes_season_episode_params_for_episodes(self):
        provider = PodnapisiProvider()
        called_params = {}

        def capture_fetch(url):
            called_params['url'] = url
            return SAMPLE_XML

        with patch.object(provider, '_fetch_xml', side_effect=capture_fetch):
            provider.search(_make_identity(season=2, episode=5), ['eng'])

        assert 'sS=2' in called_params['url']
        assert 'sE=5' in called_params['url']

    def test_search_returns_empty_on_error(self):
        provider = PodnapisiProvider()
        with patch.object(provider, '_fetch_xml', side_effect=Exception('network error')):
            matches = provider.search(_make_identity(), ['eng'])
        assert matches == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_podnapisi.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement Podnapisi provider**

```python
# movie_translator/subtitle_fetch/providers/podnapisi.py
"""Podnapisi.net provider — multilingual subtitle search via REST/XML API."""

import io
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

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
        podnapi_langs = [LANG_TO_PODNAPI[l] for l in languages if l in LANG_TO_PODNAPI]
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

        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()

        # Podnapisi returns a ZIP file
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                sub_files = [
                    n for n in zf.namelist()
                    if n.lower().endswith(('.srt', '.ass', '.ssa', '.sub'))
                ]
                if not sub_files:
                    raise RuntimeError(f'No subtitle file in Podnapisi ZIP (id={match.subtitle_id})')
                output_path.write_bytes(zf.read(sub_files[0]))
        else:
            # Some results return raw subtitle content
            output_path.write_bytes(content)

        logger.info(f'Downloaded subtitle: {output_path.name} (podnapisi)')
        return output_path

    def _fetch_xml(self, url: str) -> str:
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
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

            matches.append(SubtitleMatch(
                language=lang_3,
                source=self.name,
                subtitle_id=sub_id,
                release_name=release,
                format='srt',
                score=0.65,
                hash_match=False,
            ))

        return matches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_podnapisi.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/providers/podnapisi.py movie_translator/subtitle_fetch/tests/test_podnapisi.py
git commit -m "feat(podnapisi): add Podnapisi.net subtitle provider"
```

---

### Task 4: NapiProjekt Hash + Provider

**Files:**
- Create: `movie_translator/identifier/napihash.py`
- Create: `movie_translator/identifier/tests/test_napihash.py`
- Create: `movie_translator/subtitle_fetch/providers/napiprojekt.py`
- Create: `movie_translator/subtitle_fetch/tests/test_napiprojekt.py`

- [ ] **Step 1: Write failing tests for NapiProjekt hash**

```python
# movie_translator/identifier/tests/test_napihash.py
from pathlib import Path

from movie_translator.identifier.napihash import compute_napiprojekt_hash


class TestNapiProjektHash:
    def test_hash_of_small_file(self, tmp_path):
        """File smaller than 10MB: hash entire content."""
        f = tmp_path / 'small.mkv'
        f.write_bytes(b'hello world')
        result = compute_napiprojekt_hash(f)
        # MD5 of b'hello world'
        assert result == '5eb63bbbe01eeed093cb22bb8f5acdc3'

    def test_hash_of_large_file_reads_only_10mb(self, tmp_path):
        """File larger than 10MB: hash only first 10MB."""
        f = tmp_path / 'large.mkv'
        chunk = b'\x00' * (10 * 1024 * 1024)  # exactly 10MB of zeros
        extra = b'\xff' * 1024  # extra data beyond 10MB
        f.write_bytes(chunk + extra)
        result = compute_napiprojekt_hash(f)
        # Should be MD5 of 10MB of zeros, NOT the full file
        import hashlib
        expected = hashlib.md5(chunk).hexdigest()
        assert result == expected

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.mkv'
        f.write_bytes(b'')
        import pytest
        with pytest.raises(ValueError):
            compute_napiprojekt_hash(f)
```

- [ ] **Step 2: Run hash tests to verify they fail**

Run: `python -m pytest movie_translator/identifier/tests/test_napihash.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement NapiProjekt hash**

```python
# movie_translator/identifier/napihash.py
"""NapiProjekt hash: MD5 of the first 10MB of a video file."""

import hashlib
from pathlib import Path

NAPIPROJEKT_READ_SIZE = 10 * 1024 * 1024  # 10MB


def compute_napiprojekt_hash(path: Path) -> str:
    """Compute the NapiProjekt hash for a video file.

    Returns the MD5 hex digest of the first 10MB of the file.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f'Cannot hash empty file: {path}')

    with open(path, 'rb') as f:
        data = f.read(NAPIPROJEKT_READ_SIZE)

    return hashlib.md5(data).hexdigest()
```

- [ ] **Step 4: Run hash tests**

Run: `python -m pytest movie_translator/identifier/tests/test_napihash.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Write failing tests for NapiProjekt provider**

```python
# movie_translator/subtitle_fetch/tests/test_napiprojekt.py
from unittest.mock import patch

from movie_translator.identifier.types import MediaIdentity
from movie_translator.subtitle_fetch.providers.napiprojekt import NapiProjektProvider


def _make_identity(**overrides):
    defaults = {
        'title': 'Test Movie',
        'parsed_title': 'Test Movie',
        'year': 2020,
        'season': None,
        'episode': None,
        'media_type': 'movie',
        'oshash': '0' * 16,
        'file_size': 1_000_000,
        'raw_filename': 'test.mkv',
    }
    defaults.update(overrides)
    return MediaIdentity(**defaults)


class TestNapiProjektProvider:
    def test_name(self):
        assert NapiProjektProvider().name == 'napiprojekt'

    def test_search_only_supports_polish(self):
        provider = NapiProjektProvider()
        # NapiProjekt only has Polish subtitles
        with patch.object(provider, '_check_hash', return_value=True):
            matches = provider.search(_make_identity(), ['eng'])
        assert matches == []

    def test_search_returns_match_when_hash_found(self):
        provider = NapiProjektProvider()
        with patch.object(provider, '_check_hash', return_value=True):
            matches = provider.search(_make_identity(), ['pol'])
        assert len(matches) == 1
        assert matches[0].language == 'pol'
        assert matches[0].source == 'napiprojekt'
        assert matches[0].hash_match is True
        assert matches[0].score == 0.95

    def test_search_returns_empty_when_hash_not_found(self):
        provider = NapiProjektProvider()
        with patch.object(provider, '_check_hash', return_value=False):
            matches = provider.search(_make_identity(), ['pol'])
        assert matches == []

    def test_search_requires_video_path(self):
        """NapiProjekt needs the actual video file to compute its hash."""
        provider = NapiProjektProvider()
        # No video_path set -> no hash -> no results
        matches = provider.search(_make_identity(), ['pol'])
        assert matches == []
```

- [ ] **Step 6: Run provider tests to verify they fail**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_napiprojekt.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 7: Implement NapiProjekt provider**

```python
# movie_translator/subtitle_fetch/providers/napiprojekt.py
"""NapiProjekt provider — Polish subtitles via hash-based lookup.

NapiProjekt is the largest Polish subtitle database. Subtitles are matched
by computing MD5 of the first 10MB of the video file.
"""

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

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
            from ...identifier.napihash import compute_napiprojekt_hash
            file_hash = compute_napiprojekt_hash(self._video_path)
        except Exception as e:
            logger.debug(f'NapiProjekt hash failed: {e}')
            return []

        if not self._check_hash(file_hash):
            return []

        return [SubtitleMatch(
            language='pol',
            source=self.name,
            subtitle_id=file_hash,
            release_name=f'napiprojekt-{file_hash[:8]}',
            format='srt',
            score=0.95,  # Hash-based match, very reliable
            hash_match=True,
        )]

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        file_hash = match.subtitle_id
        token = hashlib.md5((MAGIC_PREFIX + file_hash).encode()).hexdigest()

        params = urllib.parse.urlencode({
            'f': file_hash,
            't': token,
            'v': 'pynapi',
            'l': 'PL',
            'n': file_hash,
            'p': '0',
        }).encode()

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

    def _check_hash(self, file_hash: str) -> bool:
        """Check if NapiProjekt has subtitles for the given hash."""
        token = hashlib.md5((MAGIC_PREFIX + file_hash).encode()).hexdigest()

        params = urllib.parse.urlencode({
            'f': file_hash,
            't': token,
            'v': 'pynapi',
            'l': 'PL',
            'n': file_hash,
            'p': '0',
        }).encode()

        req = urllib.request.Request(
            API_URL,
            data=params,
            headers={'User-Agent': USER_AGENT},
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
            return not content.startswith(b'NPc0') and len(content) > 10
        except Exception as e:
            logger.debug(f'NapiProjekt check failed: {e}')
            return False
```

- [ ] **Step 8: Run all NapiProjekt tests**

Run: `python -m pytest movie_translator/identifier/tests/test_napihash.py movie_translator/subtitle_fetch/tests/test_napiprojekt.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add movie_translator/identifier/napihash.py movie_translator/identifier/tests/test_napihash.py movie_translator/subtitle_fetch/providers/napiprojekt.py movie_translator/subtitle_fetch/tests/test_napiprojekt.py
git commit -m "feat(napiprojekt): add NapiProjekt provider with MD5 hash lookup"
```

---

### Task 5: Extend MediaIdentity with IMDB/TMDB IDs

**Files:**
- Modify: `movie_translator/identifier/types.py`

- [ ] **Step 1: Verify existing tests still work with extended NamedTuple**

The new fields have defaults, so existing code using positional or keyword construction is unaffected. Run existing tests first:

Run: `python -m pytest movie_translator/identifier/tests/ movie_translator/subtitle_fetch/tests/ -v`
Expected: All PASS (baseline)

- [ ] **Step 2: Add IMDB/TMDB fields to MediaIdentity**

Modify `movie_translator/identifier/types.py`:

```python
from typing import NamedTuple


class MediaIdentity(NamedTuple):
    title: str  # Best-guess title (container metadata preferred)
    parsed_title: str  # Title from filename parsing (cleaner, better for text search)
    year: int | None  # Release year
    season: int | None  # Season number
    episode: int | None  # Episode number
    media_type: str  # 'movie' or 'episode'
    oshash: str  # OpenSubtitles file hash (16 hex chars)
    file_size: int  # Bytes (needed for OpenSubtitles API)
    raw_filename: str  # Original filename for fallback search
    imdb_id: str | None = None  # e.g. 'tt0903747'
    tmdb_id: int | None = None  # TMDB numeric ID
```

- [ ] **Step 3: Run all existing tests to confirm no breakage**

Run: `python -m pytest movie_translator/identifier/tests/ movie_translator/subtitle_fetch/tests/ -v`
Expected: All PASS (new fields have defaults, backward compatible)

- [ ] **Step 4: Commit**

```bash
git add movie_translator/identifier/types.py
git commit -m "feat(identity): add optional imdb_id and tmdb_id fields"
```

---

### Task 6: TMDB Enrichment

**Files:**
- Create: `movie_translator/identifier/tmdb.py`
- Create: `movie_translator/identifier/tests/test_tmdb.py`
- Modify: `movie_translator/identifier/identify.py`

- [ ] **Step 1: Write failing tests for TMDB lookup**

```python
# movie_translator/identifier/tests/test_tmdb.py
from unittest.mock import patch

from movie_translator.identifier.tmdb import lookup_tmdb


class TestTmdbLookup:
    def test_returns_ids_for_movie(self):
        api_response = {
            'results': [{
                'id': 1396,
                'title': 'Breaking Bad',
                'release_date': '2008-01-20',
            }]
        }
        with patch('movie_translator.identifier.tmdb._tmdb_request', return_value=api_response):
            result = lookup_tmdb('Breaking Bad', year=2008, media_type='movie')

        assert result is not None
        assert result['tmdb_id'] == 1396

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
        with patch('movie_translator.identifier.tmdb._tmdb_request', side_effect=Exception('timeout')):
            result = lookup_tmdb('Test', year=None, media_type='movie')
        assert result is None

    def test_searches_tv_for_episodes(self):
        api_response = {
            'results': [{
                'id': 1399,
                'name': 'Breaking Bad',
                'first_air_date': '2008-01-20',
            }]
        }
        called_with = {}

        def capture_request(endpoint, params):
            called_with['endpoint'] = endpoint
            return api_response

        with patch('movie_translator.identifier.tmdb._tmdb_request', side_effect=capture_request):
            result = lookup_tmdb('Breaking Bad', year=2008, media_type='episode')

        assert '/search/tv' in called_with['endpoint']
        assert result['tmdb_id'] == 1399
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest movie_translator/identifier/tests/test_tmdb.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement TMDB lookup**

```python
# movie_translator/identifier/tmdb.py
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
    if not _get_api_key():
        return None

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
            detail_endpoint = f'/tv/{tmdb_id}/external_ids' if media_type == 'episode' else f'/movie/{tmdb_id}'
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
```

- [ ] **Step 4: Run TMDB tests**

Run: `python -m pytest movie_translator/identifier/tests/test_tmdb.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Integrate TMDB enrichment into identify_media**

Modify `movie_translator/identifier/identify.py` — add after the existing `return` statement is built up, enrich with TMDB:

Add import at top:
```python
from .tmdb import lookup_tmdb
```

Replace the final `return MediaIdentity(...)` block with:

```python
    # Signal 4: TMDB enrichment (optional, requires TMDB_API_KEY)
    imdb_id = None
    tmdb_id = None
    try:
        tmdb_result = lookup_tmdb(parsed_title, year, media_type)
        if tmdb_result:
            tmdb_id = tmdb_result.get('tmdb_id')
            imdb_id = tmdb_result.get('imdb_id')
            logger.debug(f'TMDB enrichment: tmdb_id={tmdb_id}, imdb_id={imdb_id}')
    except Exception as e:
        logger.debug(f'TMDB enrichment skipped: {e}')

    logger.info(f'Identified: "{title}" (type={media_type}, S{season}E{episode}, year={year})')

    return MediaIdentity(
        title=title,
        parsed_title=parsed_title,
        year=year,
        season=season,
        episode=episode,
        media_type=media_type,
        oshash=oshash,
        file_size=file_size,
        raw_filename=filename,
        imdb_id=imdb_id,
        tmdb_id=tmdb_id,
    )
```

- [ ] **Step 6: Run all identifier tests**

Run: `python -m pytest movie_translator/identifier/tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add movie_translator/identifier/tmdb.py movie_translator/identifier/tests/test_tmdb.py movie_translator/identifier/identify.py
git commit -m "feat(tmdb): add optional TMDB enrichment for IMDB/TMDB IDs"
```

---

### Task 7: Use IMDB/TMDB IDs in OpenSubtitles Search

**Files:**
- Modify: `movie_translator/subtitle_fetch/providers/opensubtitles.py`
- Modify: `movie_translator/subtitle_fetch/tests/test_opensubtitles.py`

- [ ] **Step 1: Write failing test**

Add to `test_opensubtitles.py`:

```python
def test_search_uses_imdb_id_when_available(self):
    provider = OpenSubtitlesProvider(api_key='test-key')
    called_params = {}

    def capture_request(method, endpoint, params=None, body=None):
        called_params.update(params or {})
        return {'data': []}

    with patch.object(provider, '_api_request', side_effect=capture_request):
        identity = _make_identity(imdb_id='tt0903747')
        provider.search(identity, ['eng'])

    # Should include imdb_id in query search params
    assert 'imdb_id' in called_params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py::TestOpenSubtitlesProvider::test_search_uses_imdb_id_when_available -v`
Expected: FAIL

- [ ] **Step 3: Update OpenSubtitles provider to use IMDB/TMDB IDs**

In `opensubtitles.py`, in the `search` method's Strategy 2 block (query search), add IMDB/TMDB params:

```python
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

            # Use IMDB/TMDB IDs for more precise search
            imdb_id = getattr(identity, 'imdb_id', None)
            if imdb_id:
                # OpenSubtitles expects numeric IMDB ID without 'tt' prefix
                params['imdb_id'] = imdb_id.replace('tt', '') if imdb_id.startswith('tt') else imdb_id
            tmdb_id = getattr(identity, 'tmdb_id', None)
            if tmdb_id:
                params['tmdb_id'] = str(tmdb_id)
```

- [ ] **Step 4: Run all OpenSubtitles tests**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_opensubtitles.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/subtitle_fetch/providers/opensubtitles.py movie_translator/subtitle_fetch/tests/test_opensubtitles.py
git commit -m "feat(opensubtitles): use IMDB/TMDB IDs for more accurate search"
```

---

### Task 8: Pipeline Integration — Wire New Providers

**Files:**
- Modify: `movie_translator/pipeline.py`
- Modify: `movie_translator/subtitle_fetch/__init__.py`

- [ ] **Step 1: Update pipeline to use new providers**

Modify `pipeline.py` `_build_fetcher` method:

```python
    def _build_fetcher(self, video_path: Path | None = None) -> SubtitleFetcher | None:
        """Create a SubtitleFetcher with all configured providers."""
        if not self.enable_fetch:
            return None
        providers: list = [AnimeSubProvider()]

        # Podnapisi (always available, no API key needed)
        from .subtitle_fetch.providers.podnapisi import PodnapisiProvider
        providers.append(PodnapisiProvider())

        # NapiProjekt (needs video path for hash computation)
        from .subtitle_fetch.providers.napiprojekt import NapiProjektProvider
        if video_path is not None:
            napi = NapiProjektProvider()
            napi.set_video_path(video_path)
            providers.append(napi)

        # OpenSubtitles (needs API key)
        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))

        return SubtitleFetcher(providers)
```

Update `_search_and_validate` to pass `video_path` to `_build_fetcher`:

Change line:
```python
        fetcher = self._build_fetcher()
```
to:
```python
        fetcher = self._build_fetcher(video_path=video_path)
```

- [ ] **Step 2: Update `__init__.py` exports**

Modify `movie_translator/subtitle_fetch/__init__.py`:

```python
from .fetcher import SubtitleFetcher
from .providers.base import SubtitleProvider
from .types import SubtitleMatch
from .validator import SubtitleValidator

__all__ = ['SubtitleFetcher', 'SubtitleMatch', 'SubtitleProvider', 'SubtitleValidator']
```

(No change needed — providers are imported directly in pipeline.py)

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest movie_translator/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add movie_translator/pipeline.py
git commit -m "feat(pipeline): wire Podnapisi and NapiProjekt providers"
```

---

### Task 9: Scoring Improvements

**Files:**
- Create: `movie_translator/subtitle_fetch/scoring.py`
- Create: `movie_translator/subtitle_fetch/tests/test_scoring.py`
- Modify: `movie_translator/subtitle_fetch/providers/opensubtitles.py`

- [ ] **Step 1: Write failing tests for release name scoring**

```python
# movie_translator/subtitle_fetch/tests/test_scoring.py
from movie_translator.subtitle_fetch.scoring import compute_release_score


class TestReleaseScoring:
    def test_exact_match_scores_high(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Breaking.Bad.S01E03.720p.BluRay',
        )
        assert score >= 0.9

    def test_partial_match_scores_medium(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Breaking.Bad.S01E03.1080p.WEB',
        )
        assert 0.3 < score < 0.9

    def test_no_match_scores_low(self):
        score = compute_release_score(
            'Breaking.Bad.S01E03.720p.BluRay',
            'Totally.Different.Movie.2024',
        )
        assert score < 0.3

    def test_empty_strings(self):
        assert compute_release_score('', '') == 0.0
        assert compute_release_score('test', '') == 0.0

    def test_case_insensitive(self):
        s1 = compute_release_score('Breaking.Bad', 'breaking.bad')
        s2 = compute_release_score('Breaking.Bad', 'Breaking.Bad')
        assert s1 == s2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_scoring.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement release name scoring**

```python
# movie_translator/subtitle_fetch/scoring.py
"""Subtitle match scoring utilities.

Provides release name token matching to supplement provider-level scoring.
"""

import re


def _tokenize(name: str) -> set[str]:
    """Split a release name into lowercase tokens."""
    if not name:
        return set()
    return set(re.split(r'[\.\-_\s\[\]()]+', name.lower())) - {''}


def compute_release_score(video_name: str, release_name: str) -> float:
    """Score how well a subtitle release name matches a video filename.

    Returns 0.0 to 1.0 based on token overlap (Jaccard similarity).
    """
    video_tokens = _tokenize(video_name)
    release_tokens = _tokenize(release_name)

    if not video_tokens or not release_tokens:
        return 0.0

    intersection = video_tokens & release_tokens
    union = video_tokens | release_tokens

    return len(intersection) / len(union)
```

- [ ] **Step 4: Run scoring tests**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/test_scoring.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Integrate scoring into OpenSubtitles provider**

In `opensubtitles.py`, update `_parse_results` to use release name scoring for non-hash matches:

Add import:
```python
from ..scoring import compute_release_score
```

Update the score computation in `_parse_results`:
```python
            is_hash = attrs.get('moviehash_match', False)
            if is_hash:
                score = 1.0
            else:
                base_score = 0.6
                release_bonus = compute_release_score(
                    identity.raw_filename, attrs.get('release', '')
                ) * 0.3  # up to 0.3 bonus
                score = base_score + release_bonus
```

Note: `_parse_results` needs access to `identity` — add it as a parameter:

Change signature from:
```python
def _parse_results(self, data: dict, languages: list[str]) -> list[SubtitleMatch]:
```
to:
```python
def _parse_results(self, data: dict, languages: list[str], identity=None) -> list[SubtitleMatch]:
```

And update both call sites in `search()`:
```python
matches = self._parse_results(data, languages, identity)
```

- [ ] **Step 6: Update existing test for new score range**

In `test_opensubtitles.py`, update `test_search_assigns_lower_score_for_query_match`:
```python
    def test_search_assigns_lower_score_for_query_match(self):
        # ... existing setup ...
        assert matches[0].hash_match is False
        assert 0.6 <= matches[0].score < 1.0  # now range-based, not fixed 0.7
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest movie_translator/subtitle_fetch/tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add movie_translator/subtitle_fetch/scoring.py movie_translator/subtitle_fetch/tests/test_scoring.py movie_translator/subtitle_fetch/providers/opensubtitles.py movie_translator/subtitle_fetch/tests/test_opensubtitles.py
git commit -m "feat(scoring): add release name token matching for smarter scoring"
```

---

### Task 10: Final Integration Test

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest movie_translator/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `ruff check movie_translator/`
Expected: No errors

- [ ] **Step 3: Run formatter**

Run: `ruff format movie_translator/`

- [ ] **Step 4: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: format new modules with ruff"
```
