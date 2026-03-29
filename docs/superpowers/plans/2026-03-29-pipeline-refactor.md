# Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic pipeline into discrete, testable stages with recursive directory scanning, original English track preservation, and source-attributed track naming.

**Architecture:** Context-passing stage pipeline. A `PipelineContext` dataclass flows through a chain of `Stage` classes, each responsible for one step (identify, fetch, extract, translate, create tracks, mux). The orchestrator is thin glue code.

**Tech Stack:** Python 3.10+, pysubs2, mkvmerge, ffmpeg, pytest

**Spec:** `docs/superpowers/specs/2026-03-29-pipeline-refactor-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `movie_translator/context.py` | PipelineContext, PipelineConfig, FetchedSubtitle, FontInfo, OriginalTrack dataclasses |
| `movie_translator/discovery.py` | Recursive video finding + work dir creation |
| `movie_translator/stages/__init__.py` | Re-exports all stage classes |
| `movie_translator/stages/identify.py` | IdentifyStage — media identification |
| `movie_translator/stages/extract_ref.py` | ExtractReferenceStage — reference subtitle + original track info |
| `movie_translator/stages/fetch.py` | FetchSubtitlesStage — provider search, download, validation |
| `movie_translator/stages/extract_english.py` | ExtractEnglishStage — determine English source, extract dialogue |
| `movie_translator/stages/translate.py` | TranslateStage — AI translation + font check |
| `movie_translator/stages/create_tracks.py` | CreateTracksStage — build ASS files and track list |
| `movie_translator/stages/mux.py` | MuxStage — mux video, preserve original English, replace file |
| `movie_translator/tests/test_discovery.py` | Discovery tests |
| `movie_translator/stages/tests/__init__.py` | Test package |
| `movie_translator/stages/tests/test_identify.py` | IdentifyStage tests |
| `movie_translator/stages/tests/test_extract_ref.py` | ExtractReferenceStage tests |
| `movie_translator/stages/tests/test_fetch.py` | FetchSubtitlesStage tests |
| `movie_translator/stages/tests/test_extract_english.py` | ExtractEnglishStage tests |
| `movie_translator/stages/tests/test_translate.py` | TranslateStage tests |
| `movie_translator/stages/tests/test_create_tracks.py` | CreateTracksStage tests |
| `movie_translator/stages/tests/test_mux.py` | MuxStage tests |

### Modified Files

| File | Change |
|------|--------|
| `movie_translator/pipeline.py` | Replace 443-line monolith with ~60-line stage orchestrator |
| `movie_translator/main.py` | Use `discovery.find_videos()`, slim down to pure CLI |
| `movie_translator/ffmpeg.py` | Add `original_subtitle_track` param to mux functions |
| `movie_translator/video/operations.py` | Pass through original track to mux |
| `tests/test_integration.py` | Update to new pipeline API |

---

## Task 1: Create context.py with PipelineContext and supporting dataclasses

**Files:**
- Create: `movie_translator/context.py`

- [ ] **Step 1: Create context.py**

```python
# movie_translator/context.py
"""Pipeline context and configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

from .types import DialogueLine, OCRResult, SubtitleFile


@dataclass
class PipelineConfig:
    device: str = 'mps'
    batch_size: int = 16
    model: str = 'allegro'
    enable_fetch: bool = True
    dry_run: bool = False


@dataclass
class FetchedSubtitle:
    path: Path
    source: str  # provider name, e.g. "animesub"


@dataclass
class FontInfo:
    supports_polish: bool
    font_attachments: list[Path] = field(default_factory=list)
    fallback_font_family: str | None = None


@dataclass
class OriginalTrack:
    stream_index: int
    subtitle_index: int
    codec: str  # "subrip", "ass", etc.
    language: str


@dataclass
class PipelineContext:
    # Inputs (set before pipeline runs)
    video_path: Path
    work_dir: Path
    config: PipelineConfig

    # Stage outputs (set progressively by each stage)
    identity: object | None = None  # MediaIdentity (avoid circular import)
    reference_path: Path | None = None
    original_english_track: OriginalTrack | None = None
    fetched_subtitles: dict[str, FetchedSubtitle] | None = None
    english_source: Path | None = None
    dialogue_lines: list[DialogueLine] | None = None
    translated_lines: list[DialogueLine] | None = None
    font_info: FontInfo | None = None
    subtitle_tracks: list[SubtitleFile] | None = None
    ocr_results: list[OCRResult] | None = None
    inpainted_video: Path | None = None
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `cd /Users/w/h_dev/movie_translator && .venv/bin/python -c "from movie_translator.context import PipelineContext, PipelineConfig, FetchedSubtitle, FontInfo, OriginalTrack; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add movie_translator/context.py
git commit -m "feat: add PipelineContext and config dataclasses"
```

---

## Task 2: Create discovery.py with recursive video scanning

**Files:**
- Create: `movie_translator/discovery.py`
- Create: `movie_translator/tests/test_discovery.py`

- [ ] **Step 1: Write discovery tests**

```python
# movie_translator/tests/test_discovery.py
"""Tests for recursive video discovery and working directory creation."""

from pathlib import Path

from movie_translator.discovery import VIDEO_EXTENSIONS, create_work_dir, find_videos


class TestFindVideos:
    def test_single_file(self, tmp_path):
        mkv = tmp_path / 'video.mkv'
        mkv.touch()
        assert find_videos(mkv) == [mkv]

    def test_single_file_non_video_returns_empty(self, tmp_path):
        txt = tmp_path / 'readme.txt'
        txt.touch()
        assert find_videos(txt) == []

    def test_flat_directory(self, tmp_path):
        a = tmp_path / 'a.mkv'
        b = tmp_path / 'b.mp4'
        a.touch()
        b.touch()
        result = find_videos(tmp_path)
        assert result == [a, b]

    def test_nested_anime_structure(self, tmp_path):
        # Anime/Show/Season 1/ep01.mkv
        s1 = tmp_path / 'Show' / 'Season 1'
        s1.mkdir(parents=True)
        ep1 = s1 / 'ep01.mkv'
        ep2 = s1 / 'ep02.mkv'
        ep1.touch()
        ep2.touch()

        s2 = tmp_path / 'Show' / 'Season 2'
        s2.mkdir(parents=True)
        ep3 = s2 / 'ep01.mkv'
        ep3.touch()

        result = find_videos(tmp_path)
        assert len(result) == 3
        assert ep1 in result
        assert ep2 in result
        assert ep3 in result

    def test_skips_hidden_directories(self, tmp_path):
        hidden = tmp_path / '.translate_temp'
        hidden.mkdir()
        (hidden / 'temp.mkv').touch()
        visible = tmp_path / 'ep01.mkv'
        visible.touch()
        assert find_videos(tmp_path) == [visible]

    def test_sorted_output(self, tmp_path):
        c = tmp_path / 'c.mkv'
        a = tmp_path / 'a.mkv'
        b = tmp_path / 'b.mkv'
        c.touch()
        a.touch()
        b.touch()
        assert find_videos(tmp_path) == [a, b, c]

    def test_empty_directory(self, tmp_path):
        assert find_videos(tmp_path) == []

    def test_nonexistent_path(self, tmp_path):
        assert find_videos(tmp_path / 'nope') == []


class TestCreateWorkDir:
    def test_creates_work_dir_with_subdirs(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = create_work_dir(video, tmp_path)
        assert work.exists()
        assert (work / 'candidates').exists()
        assert (work / 'reference').exists()

    def test_preserves_relative_structure(self, tmp_path):
        video = tmp_path / 'Show' / 'S1' / 'ep01.mkv'
        video.parent.mkdir(parents=True)
        video.touch()
        work = create_work_dir(video, tmp_path)
        assert '.translate_temp' in str(work)
        assert 'Show' in str(work)
        assert 'S1' in str(work)
        assert 'ep01' in str(work)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest movie_translator/tests/test_discovery.py -v`

Expected: FAIL — `cannot import name 'find_videos'`

- [ ] **Step 3: Implement discovery.py**

```python
# movie_translator/discovery.py
"""Recursive video file discovery and working directory creation."""

from pathlib import Path

VIDEO_EXTENSIONS = {'.mkv', '.mp4'}


def find_videos(input_path: Path) -> list[Path]:
    """Find all video files recursively from any input.

    - If input_path is a file: return [input_path] if it's a video, else []
    - If directory: recursively find all .mkv/.mp4 files, sorted
    - Skips hidden directories (starting with '.')
    - Returns [] for nonexistent paths
    """
    if not input_path.exists():
        return []

    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [input_path]
        return []

    videos: list[Path] = []
    for path in sorted(input_path.rglob('*')):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            # Skip files inside hidden directories
            if any(part.startswith('.') for part in path.relative_to(input_path).parts):
                continue
            videos.append(path)
    return videos


def create_work_dir(video_path: Path, root_input: Path) -> Path:
    """Create temp working directory preserving relative structure.

    For video at ~/Anime/Show/S1/ep01.mkv with root ~/Anime:
    returns ~/Anime/.translate_temp/Show/S1/ep01/
    """
    try:
        relative = video_path.parent.relative_to(root_input)
    except ValueError:
        relative = Path()

    temp_root = root_input / '.translate_temp'
    work_dir = temp_root / relative / video_path.stem

    (work_dir / 'candidates').mkdir(parents=True, exist_ok=True)
    (work_dir / 'reference').mkdir(parents=True, exist_ok=True)

    return work_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest movie_translator/tests/test_discovery.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/discovery.py movie_translator/tests/test_discovery.py
git commit -m "feat: add recursive video discovery module"
```

---

## Task 3: Create stages package with IdentifyStage

**Files:**
- Create: `movie_translator/stages/__init__.py`
- Create: `movie_translator/stages/identify.py`
- Create: `movie_translator/stages/tests/__init__.py`
- Create: `movie_translator/stages/tests/test_identify.py`

- [ ] **Step 1: Create stages package init and test init**

```python
# movie_translator/stages/__init__.py
"""Pipeline stages."""

from .identify import IdentifyStage

__all__ = ['IdentifyStage']
```

```python
# movie_translator/stages/tests/__init__.py
```

- [ ] **Step 2: Write IdentifyStage test**

```python
# movie_translator/stages/tests/test_identify.py
from pathlib import Path
from unittest.mock import patch

from movie_translator.context import PipelineConfig, PipelineContext
from movie_translator.stages.identify import IdentifyStage


class TestIdentifyStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        return PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )

    def test_sets_identity(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        class FakeIdentity:
            title = 'Test'

        with patch('movie_translator.stages.identify.identify_media', return_value=FakeIdentity()):
            result = IdentifyStage().run(ctx)

        assert result.identity is not None
        assert result.identity.title == 'Test'

    def test_returns_same_context_object(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.identify.identify_media', return_value=object()):
            result = IdentifyStage().run(ctx)

        assert result is ctx
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_identify.py -v`

Expected: FAIL — `cannot import name 'IdentifyStage'`

- [ ] **Step 4: Implement IdentifyStage**

```python
# movie_translator/stages/identify.py
"""Identify stage — extract media identity from video file."""

from ..context import PipelineContext
from ..identifier import identify_media
from ..logging import logger


class IdentifyStage:
    name = 'identify'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Identifying: {ctx.video_path.name}')
        ctx.identity = identify_media(ctx.video_path)
        return ctx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_identify.py -v`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/stages/
git commit -m "feat: add stages package with IdentifyStage"
```

---

## Task 4: ExtractReferenceStage

**Files:**
- Create: `movie_translator/stages/extract_ref.py`
- Create: `movie_translator/stages/tests/test_extract_ref.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_extract_ref.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from movie_translator.context import OriginalTrack, PipelineConfig, PipelineContext
from movie_translator.stages.extract_ref import ExtractReferenceStage


class TestExtractReferenceStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'reference').mkdir()
        return PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )

    def test_extracts_embedded_english_track(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ref_file = ctx.work_dir / 'reference' / 'ep01_reference.srt'
        ref_file.touch()

        mock_extractor = MagicMock()
        mock_extractor.get_track_info.return_value = {'tracks': [{'type': 'subtitles'}]}
        mock_extractor.find_english_track.return_value = {
            'id': 2,
            'codec': 'subrip',
            'properties': {'language': 'eng'},
            'subtitle_index': 0,
        }
        mock_extractor.get_subtitle_extension.return_value = '.srt'
        mock_extractor.extract_subtitle.return_value = None

        with patch('movie_translator.stages.extract_ref.SubtitleExtractor', return_value=mock_extractor):
            result = ExtractReferenceStage().run(ctx)

        assert result.reference_path is not None
        assert result.original_english_track is not None
        assert result.original_english_track.stream_index == 2
        assert result.original_english_track.codec == 'subrip'

    def test_no_english_track_sets_none(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        mock_extractor = MagicMock()
        mock_extractor.get_track_info.return_value = {'tracks': []}
        mock_extractor.find_english_track.return_value = None

        with patch('movie_translator.stages.extract_ref.SubtitleExtractor', return_value=mock_extractor), \
             patch('movie_translator.stages.extract_ref.is_vision_ocr_available', return_value=False):
            result = ExtractReferenceStage().run(ctx)

        assert result.reference_path is None
        assert result.original_english_track is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_extract_ref.py -v`

Expected: FAIL

- [ ] **Step 3: Implement ExtractReferenceStage**

```python
# movie_translator/stages/extract_ref.py
"""Extract reference subtitle and record original English track info."""

from ..context import OriginalTrack, PipelineContext
from ..logging import logger
from ..ocr import is_vision_ocr_available, probe_for_burned_in_subtitles, extract_burned_in_subtitles
from ..subtitles import SubtitleExtractor


class ExtractReferenceStage:
    name = 'extract_reference'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        extractor = SubtitleExtractor()
        ref_dir = ctx.work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        track_info = extractor.get_track_info(ctx.video_path)
        eng_track = extractor.find_english_track(track_info) if track_info else None

        if eng_track:
            # Record original track info for preservation in mux stage
            ctx.original_english_track = OriginalTrack(
                stream_index=eng_track['id'],
                subtitle_index=eng_track.get('subtitle_index', 0),
                codec=eng_track.get('codec', 'unknown'),
                language=eng_track.get('properties', {}).get('language', 'eng'),
            )

            subtitle_ext = extractor.get_subtitle_extension(eng_track)
            ref_path = ref_dir / f'{ctx.video_path.stem}_reference{subtitle_ext}'
            try:
                extractor.extract_subtitle(
                    ctx.video_path, eng_track['id'], ref_path,
                    eng_track.get('subtitle_index', 0),
                )
                ctx.reference_path = ref_path
                logger.info(f'Extracted reference: {ref_path.name}')
            except Exception as e:
                logger.warning(f'Failed to extract reference: {e}')

        # Fall back to OCR if no embedded track
        if ctx.reference_path is None and is_vision_ocr_available():
            if probe_for_burned_in_subtitles(ctx.video_path):
                try:
                    result = extract_burned_in_subtitles(ctx.video_path, ref_dir)
                    if result:
                        ctx.reference_path = result.srt_path
                        ctx.ocr_results = result.ocr_results
                        logger.info(f'Extracted OCR reference: {result.srt_path.name}')
                except Exception as e:
                    logger.warning(f'OCR extraction failed: {e}')

        return ctx
```

- [ ] **Step 4: Update stages/__init__.py**

```python
# movie_translator/stages/__init__.py
"""Pipeline stages."""

from .extract_ref import ExtractReferenceStage
from .identify import IdentifyStage

__all__ = ['IdentifyStage', 'ExtractReferenceStage']
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_extract_ref.py -v`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/stages/
git commit -m "feat: add ExtractReferenceStage with original track recording"
```

---

## Task 5: FetchSubtitlesStage

**Files:**
- Create: `movie_translator/stages/fetch.py`
- Create: `movie_translator/stages/tests/test_fetch.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_fetch.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from movie_translator.context import FetchedSubtitle, PipelineConfig, PipelineContext
from movie_translator.stages.fetch import FetchSubtitlesStage


class TestFetchSubtitlesStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        (work / 'candidates').mkdir(parents=True)
        return PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
            identity=MagicMock(),
        )

    def test_sets_fetched_subtitles(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        pol_path = tmp_path / 'pol.ass'
        pol_path.touch()

        mock_fetcher = MagicMock()
        mock_match = MagicMock(language='pol', source='animesub', subtitle_id='123', format='ass')
        mock_fetcher.search_all.return_value = [mock_match]
        mock_fetcher.download_candidate.return_value = pol_path

        with patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher), \
             patch('movie_translator.stages.fetch.SubtitleValidator') as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_candidates.return_value = [(mock_match, pol_path, 0.95)]
            result = FetchSubtitlesStage().run(ctx)

        assert result.fetched_subtitles is not None
        assert 'pol' in result.fetched_subtitles
        assert result.fetched_subtitles['pol'].source == 'animesub'

    def test_fetch_disabled_skips(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.config.enable_fetch = False
        result = FetchSubtitlesStage().run(ctx)
        assert result.fetched_subtitles is None

    def test_no_matches_sets_empty_dict(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        mock_fetcher = MagicMock()
        mock_fetcher.search_all.return_value = []

        with patch('movie_translator.stages.fetch.SubtitleFetcher', return_value=mock_fetcher):
            result = FetchSubtitlesStage().run(ctx)

        assert result.fetched_subtitles == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_fetch.py -v`

Expected: FAIL

- [ ] **Step 3: Implement FetchSubtitlesStage**

This is the largest stage — it extracts the search, download, and validation logic from `pipeline.py:_search_and_validate()`.

```python
# movie_translator/stages/fetch.py
"""Fetch subtitles from online providers."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..context import FetchedSubtitle, PipelineContext
from ..logging import logger
from ..subtitle_fetch import SubtitleFetcher, SubtitleValidator
from ..subtitle_fetch.providers.animesub import AnimeSubProvider
from ..subtitle_fetch.providers.napiprojekt import NapiProjektProvider
from ..subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from ..subtitle_fetch.providers.podnapisi import PodnapisiProvider
from ..subtitle_fetch.types import SubtitleMatch


class FetchSubtitlesStage:
    name = 'fetch'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.config.enable_fetch:
            return ctx

        fetcher = self._build_fetcher(ctx.video_path)
        if fetcher is None:
            return ctx

        try:
            all_matches = fetcher.search_all(ctx.identity, ['eng', 'pol'])
        except Exception as e:
            logger.warning(f'Subtitle search failed: {e}')
            ctx.fetched_subtitles = {}
            return ctx

        if not all_matches:
            logger.info('No subtitles found from any provider')
            ctx.fetched_subtitles = {}
            return ctx

        logger.info(f'Found {len(all_matches)} subtitle candidate(s)')

        # Download all candidates
        candidates_dir = ctx.work_dir / 'candidates'
        candidates_dir.mkdir(parents=True, exist_ok=True)
        downloaded = self._download_all(fetcher, all_matches, candidates_dir)

        if not downloaded:
            logger.warning('All candidate downloads failed')
            ctx.fetched_subtitles = {}
            return ctx

        # Validate and select best per language
        ctx.fetched_subtitles = self._validate_and_select(
            downloaded, ctx.reference_path,
        )
        return ctx

    def _build_fetcher(self, video_path):
        providers: list = [AnimeSubProvider()]
        providers.append(PodnapisiProvider())
        napi = NapiProjektProvider()
        napi.set_video_path(video_path)
        providers.append(napi)
        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))
        return SubtitleFetcher(providers)

    def _download_all(self, fetcher, matches, candidates_dir):
        downloaded = []

        def _download(i_match):
            i, match = i_match
            filename = f'{match.source}_{match.language}_{i}.{match.format}'
            output_path = candidates_dir / filename
            fetcher.download_candidate(match, output_path)
            return match, output_path

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_download, (i, m)): m for i, m in enumerate(matches)}
            for future in as_completed(futures):
                try:
                    downloaded.append(future.result())
                except Exception as e:
                    match = futures[future]
                    logger.warning(f'Failed to download candidate {match.subtitle_id}: {e}')

        logger.info(f'Downloaded {len(downloaded)} candidate(s)')
        return downloaded

    def _validate_and_select(self, downloaded, reference_path):
        result: dict[str, FetchedSubtitle] = {}

        if reference_path is not None:
            try:
                validator = SubtitleValidator(reference_path)
                validated = validator.validate_candidates(downloaded, min_threshold=0.3)
            except Exception as e:
                logger.warning(f'Validation failed, falling back to provider scoring: {e}')
                validated = None
        else:
            validated = None

        if validated is not None:
            if validated:
                logger.info(f'{len(validated)} candidate(s) passed validation')
                for match, path, score in validated:
                    if match.language not in result:
                        result[match.language] = FetchedSubtitle(path=path, source=match.source)
                        logger.info(
                            f'Best {match.language}: {match.release_name} '
                            f'(score: {score:.3f}, source: {match.source})'
                        )
            else:
                logger.warning('No candidates passed validation threshold')
        else:
            for match, path in downloaded:
                if match.language not in result:
                    result[match.language] = FetchedSubtitle(path=path, source=match.source)
                    logger.info(
                        f'Best {match.language} (unvalidated): {match.release_name} '
                        f'(source: {match.source})'
                    )

        return result

```

- [ ] **Step 4: Update stages/__init__.py**

Add `from .fetch import FetchSubtitlesStage` and update `__all__`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_fetch.py -v`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/stages/
git commit -m "feat: add FetchSubtitlesStage"
```

---

## Task 6: ExtractEnglishStage

**Files:**
- Create: `movie_translator/stages/extract_english.py`
- Create: `movie_translator/stages/tests/test_extract_english.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_extract_english.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from movie_translator.context import FetchedSubtitle, PipelineConfig, PipelineContext
from movie_translator.stages.extract_english import ExtractEnglishStage
from movie_translator.types import DialogueLine


class TestExtractEnglishStage:
    def _make_ctx(self, tmp_path, fetched_eng=None, reference=None):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)
        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        if fetched_eng:
            ctx.fetched_subtitles = {'eng': FetchedSubtitle(path=fetched_eng, source='opensubtitles')}
        else:
            ctx.fetched_subtitles = {}
        ctx.reference_path = reference
        return ctx

    def test_prefers_fetched_english(self, tmp_path):
        fetched = tmp_path / 'fetched.srt'
        fetched.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ref = tmp_path / 'ref.srt'
        ref.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = self._make_ctx(tmp_path, fetched_eng=fetched, reference=ref)

        lines = [DialogueLine(1000, 2000, 'Hello')]
        with patch('movie_translator.stages.extract_english.SubtitleProcessor') as MockProc:
            MockProc.extract_dialogue_lines.return_value = lines
            result = ExtractEnglishStage().run(ctx)

        assert result.english_source == fetched
        assert result.dialogue_lines == lines

    def test_falls_back_to_reference(self, tmp_path):
        ref = tmp_path / 'ref.srt'
        ref.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        ctx = self._make_ctx(tmp_path, reference=ref)

        lines = [DialogueLine(1000, 2000, 'Hello')]
        with patch('movie_translator.stages.extract_english.SubtitleProcessor') as MockProc:
            MockProc.extract_dialogue_lines.return_value = lines
            result = ExtractEnglishStage().run(ctx)

        assert result.english_source == ref

    def test_no_source_raises(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        import pytest
        with pytest.raises(RuntimeError, match='No English subtitle source'):
            ExtractEnglishStage().run(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_extract_english.py -v`

Expected: FAIL

- [ ] **Step 3: Implement ExtractEnglishStage**

```python
# movie_translator/stages/extract_english.py
"""Determine English subtitle source and extract dialogue lines."""

from ..context import PipelineContext
from ..logging import logger
from ..subtitles import SubtitleExtractor, SubtitleProcessor
from ..ocr import is_vision_ocr_available, probe_for_burned_in_subtitles, extract_burned_in_subtitles


class ExtractEnglishStage:
    name = 'extract'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Priority: fetched English > reference > embedded > OCR
        fetched_eng = None
        if ctx.fetched_subtitles:
            fetched_eng_sub = ctx.fetched_subtitles.get('eng')
            if fetched_eng_sub:
                fetched_eng = fetched_eng_sub.path

        if fetched_eng:
            ctx.english_source = fetched_eng
        elif ctx.reference_path:
            ctx.english_source = ctx.reference_path
        else:
            ctx.english_source = self._extract_from_video(ctx)

        if ctx.english_source is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
        if not ctx.dialogue_lines:
            raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')

        logger.info(f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)')
        return ctx

    def _extract_from_video(self, ctx: PipelineContext):
        extractor = SubtitleExtractor()
        track_info = extractor.get_track_info(ctx.video_path)
        if not track_info:
            return None

        eng_track = extractor.find_english_track(track_info)
        if eng_track:
            subtitle_ext = extractor.get_subtitle_extension(eng_track)
            output = ctx.work_dir / f'{ctx.video_path.stem}_extracted{subtitle_ext}'
            subtitle_index = eng_track.get('subtitle_index', 0)
            extractor.extract_subtitle(ctx.video_path, eng_track['id'], output, subtitle_index)
            return output

        # OCR fallback
        if is_vision_ocr_available() and probe_for_burned_in_subtitles(ctx.video_path):
            result = extract_burned_in_subtitles(ctx.video_path, ctx.work_dir)
            if result:
                ctx.ocr_results = result.ocr_results
                return result.srt_path

        return None
```

- [ ] **Step 4: Update stages/__init__.py, run tests, commit**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_extract_english.py -v`

```bash
git add movie_translator/stages/
git commit -m "feat: add ExtractEnglishStage"
```

---

## Task 7: TranslateStage

**Files:**
- Create: `movie_translator/stages/translate.py`
- Create: `movie_translator/stages/tests/test_translate.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_translate.py
from pathlib import Path
from unittest.mock import patch, MagicMock

from movie_translator.context import FontInfo, PipelineConfig, PipelineContext
from movie_translator.stages.translate import TranslateStage
from movie_translator.types import DialogueLine


class TestTranslateStage:
    def _make_ctx(self, tmp_path):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        eng_src = tmp_path / 'eng.srt'
        eng_src.touch()
        ctx = PipelineContext(
            video_path=video,
            work_dir=tmp_path / 'work',
            config=PipelineConfig(),
        )
        ctx.english_source = eng_src
        ctx.dialogue_lines = [
            DialogueLine(1000, 3000, 'Hello'),
            DialogueLine(4000, 6000, 'Goodbye'),
        ]
        return ctx

    def test_sets_translated_lines_and_font_info(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        translated = [
            DialogueLine(1000, 3000, 'Cześć'),
            DialogueLine(4000, 6000, 'Do widzenia'),
        ]

        with patch('movie_translator.stages.translate.translate_dialogue_lines', return_value=translated), \
             patch('movie_translator.stages.translate.check_embedded_fonts_support_polish', return_value=False), \
             patch('movie_translator.stages.translate.get_ass_font_names', return_value={'Arial'}), \
             patch('movie_translator.stages.translate.find_system_font_for_polish', return_value=None):
            result = TranslateStage().run(ctx)

        assert result.translated_lines == translated
        assert result.font_info is not None
        assert result.font_info.supports_polish is False

    def test_raises_on_empty_translation(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        import pytest
        with patch('movie_translator.stages.translate.translate_dialogue_lines', return_value=[]), \
             patch('movie_translator.stages.translate.check_embedded_fonts_support_polish', return_value=False), \
             patch('movie_translator.stages.translate.get_ass_font_names', return_value=set()), \
             patch('movie_translator.stages.translate.find_system_font_for_polish', return_value=None):
            with pytest.raises(RuntimeError, match='Translation failed'):
                TranslateStage().run(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_translate.py -v`

Expected: FAIL

- [ ] **Step 3: Implement TranslateStage**

```python
# movie_translator/stages/translate.py
"""AI translation and font checking stage."""

from concurrent.futures import ThreadPoolExecutor

from ..context import FontInfo, PipelineContext
from ..fonts import (
    check_embedded_fonts_support_polish,
    find_system_font_for_polish,
    get_ass_font_names,
)
from ..logging import logger
from ..translation import translate_dialogue_lines


class TranslateStage:
    name = 'translate'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Translating {len(ctx.dialogue_lines)} lines...')

        def _check_fonts():
            supports = check_embedded_fonts_support_polish(ctx.video_path, ctx.english_source)
            if supports:
                return FontInfo(supports_polish=True)
            is_mkv = ctx.video_path.suffix.lower() == '.mkv'
            if is_mkv:
                names = get_ass_font_names(ctx.english_source)
                result = find_system_font_for_polish(names)
                if result:
                    fp, fam = result
                    fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                    return FontInfo(
                        supports_polish=False,
                        font_attachments=[fp],
                        fallback_font_family=fallback,
                    )
            return FontInfo(supports_polish=False)

        with ThreadPoolExecutor(max_workers=2) as pool:
            font_future = pool.submit(_check_fonts)
            translate_future = pool.submit(
                translate_dialogue_lines,
                ctx.dialogue_lines,
                ctx.config.device,
                ctx.config.batch_size,
                ctx.config.model,
            )

            ctx.font_info = font_future.result()
            translated = translate_future.result()

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
```

- [ ] **Step 4: Update stages/__init__.py, run tests, commit**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_translate.py -v`

```bash
git add movie_translator/stages/
git commit -m "feat: add TranslateStage with concurrent font check"
```

---

## Task 8: CreateTracksStage

**Files:**
- Create: `movie_translator/stages/create_tracks.py`
- Create: `movie_translator/stages/tests/test_create_tracks.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_create_tracks.py
from pathlib import Path
from unittest.mock import patch

from movie_translator.context import (
    FetchedSubtitle,
    FontInfo,
    PipelineConfig,
    PipelineContext,
)
from movie_translator.stages.create_tracks import CreateTracksStage
from movie_translator.types import DialogueLine


class TestCreateTracksStage:
    def _make_ctx(self, tmp_path, fetched_pol=None):
        video = tmp_path / 'ep01.mkv'
        video.touch()
        eng_src = tmp_path / 'eng.srt'
        eng_src.write_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n')
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)

        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(),
        )
        ctx.english_source = eng_src
        ctx.dialogue_lines = [DialogueLine(1000, 2000, 'Hello')]
        ctx.translated_lines = [DialogueLine(1000, 2000, 'Cześć')]
        ctx.font_info = FontInfo(supports_polish=True)

        if fetched_pol:
            ctx.fetched_subtitles = {'pol': fetched_pol}
        else:
            ctx.fetched_subtitles = {}
        return ctx

    def test_creates_ai_polish_track(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as MockProc:
            result = CreateTracksStage().run(ctx)

        assert result.subtitle_tracks is not None
        titles = [t.title for t in result.subtitle_tracks]
        assert 'Polish (AI)' in titles

    def test_fetched_polish_includes_source(self, tmp_path):
        pol_file = tmp_path / 'pol.ass'
        pol_file.write_text('[Script Info]\n\n[Events]\n')
        ctx = self._make_ctx(tmp_path, fetched_pol=FetchedSubtitle(pol_file, 'animesub'))

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as MockProc:
            result = CreateTracksStage().run(ctx)

        titles = [t.title for t in result.subtitle_tracks]
        assert 'Polish (animesub)' in titles
        assert 'Polish (AI)' in titles

    def test_fetched_polish_is_default(self, tmp_path):
        pol_file = tmp_path / 'pol.ass'
        pol_file.write_text('[Script Info]\n\n[Events]\n')
        ctx = self._make_ctx(tmp_path, fetched_pol=FetchedSubtitle(pol_file, 'podnapisi'))

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as MockProc:
            result = CreateTracksStage().run(ctx)

        defaults = [t for t in result.subtitle_tracks if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].title == 'Polish (podnapisi)'

    def test_ai_is_default_when_no_fetched(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.create_tracks.SubtitleProcessor') as MockProc:
            result = CreateTracksStage().run(ctx)

        defaults = [t for t in result.subtitle_tracks if t.is_default]
        assert len(defaults) == 1
        assert defaults[0].title == 'Polish (AI)'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_create_tracks.py -v`

Expected: FAIL

- [ ] **Step 3: Implement CreateTracksStage**

```python
# movie_translator/stages/create_tracks.py
"""Create subtitle track files and build the track list."""

from ..context import PipelineContext
from ..logging import logger
from ..subtitles import SubtitleProcessor
from ..types import SubtitleFile


class CreateTracksStage:
    name = 'create_tracks'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        is_mkv = ctx.video_path.suffix.lower() == '.mkv'
        replace_chars = False

        if not ctx.font_info.supports_polish:
            if is_mkv and ctx.font_info.font_attachments:
                logger.info(f'Will embed font "{ctx.font_info.font_attachments[0].name}"')
            elif is_mkv:
                logger.warning('No system font with Polish support, replacing characters')
                replace_chars = True
            else:
                replace_chars = True

        # Create AI Polish subtitle file
        ai_polish_ass = ctx.work_dir / f'{ctx.video_path.stem}_polish_ai.ass'
        SubtitleProcessor.create_polish_subtitles(
            ctx.english_source, ctx.translated_lines, ai_polish_ass, replace_chars,
        )
        if ctx.font_info.fallback_font_family:
            SubtitleProcessor.override_font_name(ai_polish_ass, ctx.font_info.fallback_font_family)

        # Build track list
        fetched_pol = ctx.fetched_subtitles.get('pol') if ctx.fetched_subtitles else None
        tracks: list[SubtitleFile] = []

        if fetched_pol:
            pol_title = f'Polish ({fetched_pol.source})'
            tracks.append(SubtitleFile(fetched_pol.path, 'pol', pol_title, is_default=True))
            if ctx.font_info.fallback_font_family:
                SubtitleProcessor.override_font_name(
                    fetched_pol.path, ctx.font_info.fallback_font_family,
                )

        tracks.append(SubtitleFile(
            ai_polish_ass, 'pol', 'Polish (AI)', is_default=not bool(fetched_pol),
        ))

        ctx.subtitle_tracks = tracks
        return ctx
```

- [ ] **Step 4: Update stages/__init__.py, run tests, commit**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_create_tracks.py -v`

```bash
git add movie_translator/stages/
git commit -m "feat: add CreateTracksStage with source-attributed naming"
```

---

## Task 9: Update muxing to preserve original English track

**Files:**
- Modify: `movie_translator/ffmpeg.py`
- Modify: `movie_translator/video/operations.py`
- Modify: `movie_translator/video/tests/test_operations.py`

- [ ] **Step 1: Write test for original track preservation**

Add to `movie_translator/video/tests/test_operations.py`:

```python
class TestOriginalTrackPreservation:
    def test_mkvmerge_preserves_original_english(self, create_test_mkv, create_ass_file, tmp_path):
        from movie_translator.ffmpeg import get_mkvmerge
        if get_mkvmerge() is None:
            pytest.skip('mkvmerge not installed')

        mkv_file = create_test_mkv(language='eng', track_name='English')
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        subs = [SubtitleFile(polish_ass, 'pol', 'Polish (AI)', is_default=True)]
        ops = VideoOperations()
        ops.create_clean_video(
            mkv_file, subs, output_path,
            original_sub_index=0,
            original_sub_title='English (Original)',
        )

        from movie_translator.ffmpeg import get_video_info
        info = get_video_info(output_path)
        sub_streams = [s for s in info['streams'] if s['codec_type'] == 'subtitle']
        assert len(sub_streams) == 2  # original English + Polish (AI)
        titles = [s.get('tags', {}).get('title', '') for s in sub_streams]
        assert 'English (Original)' in titles

    def test_no_original_track_works(self, create_test_mkv, create_ass_file, tmp_path):
        from movie_translator.ffmpeg import get_mkvmerge
        if get_mkvmerge() is None:
            pytest.skip('mkvmerge not installed')

        mkv_file = create_test_mkv()
        polish_ass = create_ass_file('polish.ass')
        output_path = tmp_path / 'output.mkv'

        subs = [SubtitleFile(polish_ass, 'pol', 'Polish (AI)', is_default=True)]
        ops = VideoOperations()
        ops.create_clean_video(mkv_file, subs, output_path)

        from movie_translator.ffmpeg import get_video_info
        info = get_video_info(output_path)
        sub_streams = [s for s in info['streams'] if s['codec_type'] == 'subtitle']
        assert len(sub_streams) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/video/tests/test_operations.py::TestOriginalTrackPreservation -v`

Expected: FAIL — `create_clean_video() got unexpected keyword argument 'original_sub_index'`

- [ ] **Step 3: Update video/operations.py**

```python
# In VideoOperations.create_clean_video(), add parameters:
def create_clean_video(
    self,
    original_video: Path,
    subtitle_files: list[SubtitleFile],
    output_video: Path,
    font_attachments: list[Path] | None = None,
    original_sub_index: int | None = None,
    original_sub_title: str | None = None,
) -> None:
    logger.info(f'🎬 Creating clean video: {output_video.name}')
    track_desc = ', '.join(f'{s.title} ({s.language})' for s in subtitle_files)
    if original_sub_title:
        track_desc = f'{original_sub_title}, {track_desc}'
    logger.info(f'   - Adding: {track_desc}')

    mux_video_with_subtitles(
        original_video,
        subtitle_files,
        output_video,
        font_attachments=font_attachments,
        original_sub_index=original_sub_index,
        original_sub_title=original_sub_title,
    )

    logger.info('   - Clean video merge successful')

    if output_video.exists() and output_video.stat().st_size > 0:
        size_mb = output_video.stat().st_size / 1024 / 1024
        logger.info(f'   - Output size: {size_mb:.1f} MB')
```

- [ ] **Step 4: Update ffmpeg.py mux functions**

Update `mux_video_with_subtitles` signature:

```python
def mux_video_with_subtitles(
    video_path: Path,
    subtitle_files: list[SubtitleFile],
    output_path: Path,
    font_attachments: list[Path] | None = None,
    original_sub_index: int | None = None,
    original_sub_title: str | None = None,
) -> None:
```

Pass through to both `_mux_with_mkvmerge` and `_mux_with_ffmpeg`.

In `_mux_with_mkvmerge`: replace `--no-subtitles` with `--subtitle-tracks <track_id>` when `original_sub_index` is set, and add `--track-name <tid>:<title>`:

```python
def _mux_with_mkvmerge(
    mkvmerge, video_path, subtitle_files, output_path,
    font_attachments=None, original_sub_index=None, original_sub_title=None,
):
    cmd = [mkvmerge, '-o', str(output_path)]

    if original_sub_index is not None:
        cmd.extend(['--subtitle-tracks', str(original_sub_index)])
        if original_sub_title:
            cmd.extend(['--track-name', f'{original_sub_index}:{original_sub_title}'])
        cmd.extend(['--default-track-flag', f'{original_sub_index}:0'])
    else:
        cmd.append('--no-subtitles')

    cmd.append(str(video_path))
    # ... rest unchanged (add subtitle files, fonts)
```

In `_mux_with_ffmpeg`: map the specific subtitle stream when `original_sub_index` is set:

```python
def _mux_with_ffmpeg(
    video_path, subtitle_files, output_path,
    font_attachments=None, original_sub_index=None, original_sub_title=None,
):
    # ... existing setup ...

    cmd.extend(['-map', '0:v'])
    cmd.extend(['-map', '0:a'])
    cmd.extend(['-map', '0:t?'])

    # Preserve original subtitle track if specified
    orig_sub_offset = 0
    if original_sub_index is not None:
        cmd.extend(['-map', f'0:s:{original_sub_index}'])
        orig_sub_offset = 1

    for i in range(1, len(subtitle_files) + 1):
        cmd.extend(['-map', f'{i}:0'])

    # ... codec settings ...

    # Metadata for preserved original track
    if original_sub_index is not None:
        if original_sub_title:
            cmd.extend(['-metadata:s:s:0', f'title={original_sub_title}'])
        cmd.extend(['-disposition:s:0', '0'])

    # Metadata for our added tracks (offset by orig_sub_offset)
    for i, sub in enumerate(subtitle_files):
        idx = i + orig_sub_offset
        cmd.extend([f'-metadata:s:s:{idx}', f'language={sub.language}'])
        cmd.extend([f'-metadata:s:s:{idx}', f'title={sub.title}'])
        disposition = 'default' if sub.is_default else '0'
        cmd.extend([f'-disposition:s:{idx}', disposition])
    # ...
```

- [ ] **Step 5: Run all video operations tests**

Run: `.venv/bin/python -m pytest movie_translator/video/tests/test_operations.py -v`

Expected: All PASS (existing + new tests)

- [ ] **Step 6: Commit**

```bash
git add movie_translator/ffmpeg.py movie_translator/video/operations.py movie_translator/video/tests/test_operations.py
git commit -m "feat: support preserving original English subtitle track in mux"
```

---

## Task 10: MuxStage

**Files:**
- Create: `movie_translator/stages/mux.py`
- Create: `movie_translator/stages/tests/test_mux.py`
- Modify: `movie_translator/stages/__init__.py`

- [ ] **Step 1: Write test**

```python
# movie_translator/stages/tests/test_mux.py
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from movie_translator.context import (
    FontInfo,
    OriginalTrack,
    PipelineConfig,
    PipelineContext,
)
from movie_translator.stages.mux import MuxStage
from movie_translator.types import SubtitleFile


class TestMuxStage:
    def _make_ctx(self, tmp_path, dry_run=False):
        video = tmp_path / 'ep01.mkv'
        video.write_text('fake video')
        work = tmp_path / 'work'
        work.mkdir(exist_ok=True)

        pol_ass = tmp_path / 'pol.ass'
        pol_ass.touch()

        ctx = PipelineContext(
            video_path=video,
            work_dir=work,
            config=PipelineConfig(dry_run=dry_run),
        )
        ctx.subtitle_tracks = [SubtitleFile(pol_ass, 'pol', 'Polish (AI)', is_default=True)]
        ctx.font_info = FontInfo(supports_polish=True)
        ctx.original_english_track = OriginalTrack(
            stream_index=2, subtitle_index=0, codec='subrip', language='eng',
        )
        return ctx

    def test_passes_original_track_to_mux(self, tmp_path):
        ctx = self._make_ctx(tmp_path)

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            # Simulate output file creation
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            assert call_kwargs.kwargs.get('original_sub_index') == 0
            assert call_kwargs.kwargs.get('original_sub_title') == 'English (Original)'

    def test_dry_run_does_not_replace_original(self, tmp_path):
        ctx = self._make_ctx(tmp_path, dry_run=True)

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

        # Original should still contain 'fake video'
        assert ctx.video_path.read_text() == 'fake video'

    def test_no_original_track_passes_none(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ctx.original_english_track = None

        with patch('movie_translator.stages.mux.VideoOperations') as MockOps:
            mock_ops = MockOps.return_value
            mock_ops.create_clean_video.return_value = None
            mock_ops.verify_result.return_value = None
            temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp.mkv'
            temp_video.write_text('output')
            MuxStage().run(ctx)

            call_kwargs = mock_ops.create_clean_video.call_args
            assert call_kwargs.kwargs.get('original_sub_index') is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/test_mux.py -v`

Expected: FAIL

- [ ] **Step 3: Implement MuxStage**

```python
# movie_translator/stages/mux.py
"""Final video muxing stage — combines video with subtitle tracks."""

import shutil

from ..context import PipelineContext
from ..inpainting import remove_burned_in_subtitles
from ..logging import logger
from ..video import VideoOperations


class MuxStage:
    name = 'mux'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Use inpainted video if OCR was used
        source_video = ctx.video_path
        if ctx.ocr_results and ctx.inpainted_video is None:
            logger.info('Removing burned-in subtitles...')
            inpainted = ctx.work_dir / f'{ctx.video_path.stem}_inpainted{ctx.video_path.suffix}'
            remove_burned_in_subtitles(
                ctx.video_path, inpainted, ctx.ocr_results, ctx.config.device,
            )
            ctx.inpainted_video = inpainted
            source_video = inpainted
        elif ctx.inpainted_video:
            source_video = ctx.inpainted_video

        # Determine original track preservation
        original_sub_index = None
        original_sub_title = None
        if ctx.original_english_track:
            original_sub_index = ctx.original_english_track.subtitle_index
            original_sub_title = 'English (Original)'

        # Mux
        temp_video = ctx.work_dir / f'{ctx.video_path.stem}_temp{ctx.video_path.suffix}'
        ops = VideoOperations()
        ops.create_clean_video(
            source_video,
            ctx.subtitle_tracks,
            temp_video,
            font_attachments=ctx.font_info.font_attachments or None,
            original_sub_index=original_sub_index,
            original_sub_title=original_sub_title,
        )
        ops.verify_result(temp_video, expected_tracks=ctx.subtitle_tracks)

        if not ctx.config.dry_run:
            self._replace_original(ctx.video_path, temp_video)

        return ctx

    def _replace_original(self, video_path, temp_video):
        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)
        try:
            shutil.move(str(temp_video), str(video_path))
            ops = VideoOperations()
            ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
```

- [ ] **Step 4: Update stages/__init__.py with all stages, run tests, commit**

```python
# movie_translator/stages/__init__.py
"""Pipeline stages."""

from .create_tracks import CreateTracksStage
from .extract_english import ExtractEnglishStage
from .extract_ref import ExtractReferenceStage
from .fetch import FetchSubtitlesStage
from .identify import IdentifyStage
from .mux import MuxStage
from .translate import TranslateStage

__all__ = [
    'IdentifyStage',
    'ExtractReferenceStage',
    'FetchSubtitlesStage',
    'ExtractEnglishStage',
    'TranslateStage',
    'CreateTracksStage',
    'MuxStage',
]
```

Run: `.venv/bin/python -m pytest movie_translator/stages/tests/ -v`

```bash
git add movie_translator/stages/
git commit -m "feat: add MuxStage with original English track preservation"
```

---

## Task 11: Rewrite pipeline.py as thin orchestrator

**Files:**
- Modify: `movie_translator/pipeline.py`

- [ ] **Step 1: Replace pipeline.py contents**

```python
# movie_translator/pipeline.py
"""Thin pipeline orchestrator — chains stages sequentially."""

from pathlib import Path

from .context import PipelineConfig, PipelineContext
from .logging import logger
from .stages import (
    CreateTracksStage,
    ExtractEnglishStage,
    ExtractReferenceStage,
    FetchSubtitlesStage,
    IdentifyStage,
    MuxStage,
    TranslateStage,
)


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        tracker=None,
    ):
        self.config = PipelineConfig(
            device=device,
            batch_size=batch_size,
            model=model,
            enable_fetch=enable_fetch,
        )
        self.tracker = tracker
        self.stages = [
            IdentifyStage(),
            ExtractReferenceStage(),
            FetchSubtitlesStage(),
            ExtractEnglishStage(),
            TranslateStage(),
            CreateTracksStage(),
            MuxStage(),
        ]

    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self.config.dry_run = dry_run
        ctx = PipelineContext(video_path=video_path, work_dir=work_dir, config=self.config)

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                ctx = stage.run(ctx)
            return True
        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False
```

- [ ] **Step 2: Verify import works**

Run: `.venv/bin/python -c "from movie_translator.pipeline import TranslationPipeline; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add movie_translator/pipeline.py
git commit -m "refactor: replace monolithic pipeline with stage orchestrator"
```

---

## Task 12: Update main.py to use discovery module

**Files:**
- Modify: `movie_translator/main.py`

- [ ] **Step 1: Rewrite main.py**

Replace the file discovery functions and update the main loop. Keep `check_dependencies()`, `parse_args()`, `show_summary()`, and the `main()` function. Remove `find_video_files()`, `find_video_files_with_temp_dirs()`, `create_working_dirs()`, and `VIDEO_EXTENSIONS`.

```python
# movie_translator/main.py
import logging
import shutil
import sys
from pathlib import Path

from .discovery import create_work_dir, find_videos
from .logging import console, logger, set_verbose
from .pipeline import TranslationPipeline
from .progress import ProgressTracker
from .subtitles import SubtitleExtractor


def check_dependencies() -> bool:
    """Check all required dependencies. Returns True if all satisfied."""
    import importlib.util

    from .ffmpeg import get_ffmpeg_version

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print(f'[red]❌ Python 3.10+ required, found {version.major}.{version.minor}[/red]')
        return False

    try:
        get_ffmpeg_version()
    except Exception:
        console.print('[red]❌ FFmpeg not available. Run ./setup.sh first.[/red]')
        return False

    required_packages = ['pysubs2', 'torch', 'transformers']
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            console.print(f'[red]❌ Missing package: {package}. Run ./setup.sh first.[/red]')
            return False

    return True


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original video'
    )
    parser.add_argument('input', help='Video file or directory containing video files')
    parser.add_argument(
        '--device',
        choices=['cpu', 'mps'],
        default='mps' if sys.platform == 'darwin' else 'cpu',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model', choices=['allegro'], default='allegro')
    parser.add_argument('--no-fetch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-artifacts', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def show_summary(results: list[tuple[str, str]], dry_run: bool = False) -> None:
    successful = sum(1 for _, status in results if status == 'success')
    failed = sum(1 for _, status in results if status == 'failed')
    skipped = sum(1 for _, status in results if status == 'skipped')

    parts = []
    if successful > 0:
        parts.append(f'[green]✓ {successful} translated[/green]')
    if skipped > 0:
        parts.append(f'[blue]⏭ {skipped} skipped[/blue]')
    if failed > 0:
        parts.append(f'[red]✗ {failed} failed[/red]')

    console.print(' | '.join(parts))

    if dry_run and successful > 0:
        console.print('[yellow]Dry run - originals not modified[/yellow]')


def main():
    args = parse_args()
    set_verbose(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f'[red]❌ Not found: {input_path}[/red]')
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    video_files = find_videos(input_path)
    if not video_files:
        console.print(f'[red]❌ No video files found in {input_path}[/red]')
        sys.exit(1)

    # Determine root directory for work dir creation
    root_dir = input_path if input_path.is_dir() else input_path.parent

    if args.dry_run:
        console.print('[yellow]Dry run mode - originals will not be modified[/yellow]')

    logging.getLogger('transformers').setLevel(logging.ERROR)
    extractor = SubtitleExtractor()

    with ProgressTracker(len(video_files), console=console) as tracker:
        pipeline = TranslationPipeline(
            device=args.device,
            batch_size=args.batch_size,
            model=args.model,
            enable_fetch=not args.no_fetch,
            tracker=tracker,
        )

        for video_path in video_files:
            relative_name = str(video_path.relative_to(root_dir)) if root_dir != video_path.parent else video_path.name
            tracker.start_file(relative_name)
            work_dir = create_work_dir(video_path, root_dir)
            success = False

            try:
                if extractor.has_polish_subtitles(video_path):
                    tracker.complete_file('skipped')
                    success = True
                elif pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run):
                    tracker.complete_file('success')
                    success = True
                else:
                    tracker.complete_file('failed')
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                tracker.complete_file('failed')

            if success and not args.keep_artifacts and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    # Clean up empty parent dirs up to .translate_temp
                    parent = work_dir.parent
                    temp_root = root_dir / '.translate_temp'
                    while parent != temp_root and parent != root_dir:
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                    if temp_root.exists() and not any(temp_root.iterdir()):
                        temp_root.rmdir()
                except OSError as e:
                    logger.debug(f'Failed to clean up {work_dir}: {e}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Update pyproject.toml if the CLI arg name changed**

The arg changed from `input_dir` to `input` (since it can now be a single file). Check `pyproject.toml` — the entry point `movie-translator = "movie_translator.main:main"` is unchanged, only the argparse positional name changed internally.

- [ ] **Step 3: Verify CLI still works**

Run: `.venv/bin/movie-translator --help`

Expected: Help text showing `input` as positional arg

- [ ] **Step 4: Commit**

```bash
git add movie_translator/main.py
git commit -m "refactor: slim main.py to pure CLI, use discovery module"
```

---

## Task 13: Update integration tests

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Update integration tests**

The integration tests need updates for:
1. Replace `find_video_files_with_temp_dirs` references with `find_videos` + `create_work_dir`
2. Pipeline API is the same (`TranslationPipeline.process_video_file()`)
3. Update `TestFindVideoFiles` to test `find_videos` and `create_work_dir` from `discovery.py`

Read the current `tests/test_integration.py`, then update:
- Change imports from `movie_translator.main` to `movie_translator.discovery`
- Replace `find_video_files_with_temp_dirs` calls with `find_videos`
- Update assertions to match new behavior (recursive, sorted)
- Remove tests that tested `find_video_files_with_temp_dirs` temp dir creation logic — those are now in `test_discovery.py`

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest -v`

Expected: All tests pass (may need minor fixes to align with new API)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: update integration tests for new pipeline and discovery"
```

---

## Task 14: Final cleanup and verification

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -v`

Expected: All tests pass

- [ ] **Step 2: Run linter**

Run: `.venv/bin/ruff check movie_translator/`

Expected: No errors

- [ ] **Step 3: Test end-to-end with a real video**

Run: `.venv/bin/movie-translator /tmp/translate_test --dry-run --keep-artifacts -v`

Expected: Pipeline completes successfully with new stage-based output and track names showing sources

- [ ] **Step 4: Verify subtitle track names in output**

Run:
```bash
ffprobe -v error -show_streams -select_streams s \
  "/tmp/translate_test/.translate_temp/test_anime/[EMBER] Undead Unluck - 01/[EMBER] Undead Unluck - 01_temp.mkv" \
  2>/dev/null | grep -E 'TAG:title|TAG:language'
```

Expected output should include:
- `TAG:title=English (Original)`
- `TAG:title=Polish (animesub)` (or similar source name)
- `TAG:title=Polish (AI)`

- [ ] **Step 5: Commit any remaining fixes**

```bash
git add -u
git commit -m "refactor: pipeline refactor complete — stages, discovery, track naming"
```
