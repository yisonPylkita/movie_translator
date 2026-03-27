# Subtitle Fetch & Media Identification Design

## Goal

Add the ability to download existing Polish and English subtitles from online databases before falling back to OCR extraction and AI translation. This avoids unnecessary translation work when human-quality subtitles already exist.

## Priority Chain

The pipeline should prefer pre-existing subtitles over self-generated ones:

1. **Fetch** Polish + English from subtitle databases
2. **Extract** embedded subtitle tracks from the video container
3. **OCR** burned-in subtitles (slowest, lowest quality)

Translation is only needed when Polish subtitles aren't available from any source.

## Architecture

Two new modules plus pipeline integration:

```
identifier/         -- Figure out WHAT we're processing
subtitle_fetch/     -- Download subtitles from external databases
pipeline.py         -- Updated decision logic
```

---

## Module 1: Media Identification (`identifier/`)

### Purpose

Determine the title, episode, year, and other metadata for a video file using multiple signals.

### Signals (priority order)

| Signal | Source | Provides |
|--------|--------|----------|
| Container metadata | ffprobe tags (title, episode, description) | Most authoritative when present |
| Filename parsing | guessit library on video filename | title, season, episode, year, release group |
| Folder name parsing | guessit on parent directory name | Fallback context (series name often in folder) |
| File hash | OpenSubtitles hash (first+last 64KB) | Exact-match key for database lookups |

### OpenSubtitles Hash Algorithm

Read first 64KB + last 64KB of the file. Sum all 8-byte little-endian chunks and add the file size. Return as 16-char lowercase hex. No external dependency needed.

### Output Type

```python
class MediaIdentity(NamedTuple):
    title: str                    # Best-guess title (combined from signals)
    year: int | None              # Release year
    season: int | None            # Season number
    episode: int | None           # Episode number
    media_type: str               # 'movie' or 'episode'
    oshash: str                   # OpenSubtitles file hash (16 hex chars)
    file_size: int                # Bytes (needed for OpenSubtitles API)
    raw_filename: str             # Original filename for fallback text search
```

### Combining Signals

1. Parse filename with guessit. Parse parent folder name with guessit.
2. Read container metadata via ffprobe (already have `get_video_info`).
3. Compute OpenSubtitles hash.
4. Merge: container metadata fields override filename fields when present. Folder name fills gaps (e.g., series title often only in folder).

### Files

- `identifier/__init__.py` -- exports `identify_media`
- `identifier/hasher.py` -- OpenSubtitles hash computation
- `identifier/metadata.py` -- ffprobe container metadata extraction
- `identifier/parser.py` -- guessit wrapper for filename + folder
- `identifier/types.py` -- `MediaIdentity` type

### Dependencies

- `guessit>=3.8` (new dependency, filename parsing)

---

## Module 2: Subtitle Fetch (`subtitle_fetch/`)

### Purpose

Download subtitles from external databases given a `MediaIdentity` and desired languages.

### Provider Protocol

```python
class SubtitleMatch(NamedTuple):
    language: str           # ISO 639-2B code ('eng', 'pol')
    source: str             # Provider name (e.g., 'opensubtitles')
    subtitle_id: str        # Provider-specific identifier
    release_name: str       # Subtitle release name
    format: str             # File format ('srt', 'ass', 'sub')
    score: float            # Match confidence 0.0-1.0
    hash_match: bool        # True if matched by file hash (highest confidence)

class SubtitleProvider(Protocol):
    name: str

    def search(
        self, identity: MediaIdentity, languages: list[str]
    ) -> list[SubtitleMatch]:
        """Search for subtitles. Returns matches sorted by score descending."""
        ...

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        """Download subtitle to output_path. Returns the actual written path."""
        ...
```

### OpenSubtitles Backend

**API**: REST v2 at `https://api.opensubtitles.com/api/v1`

**Authentication**:
- API key via `OPENSUBTITLES_API_KEY` env var (required for all requests).
- Username/password via `OPENSUBTITLES_USERNAME` / `OPENSUBTITLES_PASSWORD` env vars (required for downloads — the `/login` endpoint returns a JWT token used to authorize `/download` calls).

**Search strategy**:
1. Search by `moviehash` + `moviebytesize` (exact file match, score=1.0)
2. If no hash results: search by `query` (title) + `season_number` + `episode_number` (score=0.7)
3. Filter by requested `languages`

**Rate limiting**: Respect `X-RateLimit-*` headers. Back off on 429 responses.

**Download flow**: Call `/download` endpoint with subtitle file ID, receive a temporary download link, fetch the subtitle file (often gzipped).

**HTTP client**: Use `urllib.request` from stdlib to avoid adding dependencies.

### Subtitle Fetcher (Orchestrator)

```python
class SubtitleFetcher:
    def __init__(self, providers: list[SubtitleProvider]): ...

    def fetch_subtitles(
        self,
        identity: MediaIdentity,
        languages: list[str],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Try all providers. Return {language_code: subtitle_file_path} for best matches."""
```

**Ranking logic**:
1. Hash-matched results rank above query-matched
2. Among same match type, prefer higher score
3. Prefer .ass over .srt (richer formatting for anime)
4. First provider to return a hash match wins (no need to query others)

### Files

- `subtitle_fetch/__init__.py` -- exports `SubtitleFetcher`, `SubtitleProvider`
- `subtitle_fetch/types.py` -- `SubtitleMatch` type
- `subtitle_fetch/fetcher.py` -- `SubtitleFetcher` orchestrator
- `subtitle_fetch/providers/__init__.py`
- `subtitle_fetch/providers/opensubtitles.py` -- OpenSubtitles REST API v2
- `subtitle_fetch/providers/base.py` -- `SubtitleProvider` protocol

### Dependencies

No new dependencies (uses stdlib `urllib.request` + `json`).

---

## Module 3: Pipeline Integration

### Updated Flow in `TranslationPipeline.process_video_file()`

```
OLD:
  1. Extract subtitles (embedded or OCR)
  2. Parse dialogue lines
  3. Translate to Polish
  4. Create subtitle files
  5. Inpaint burned-in subtitles (if OCR was used)
  6. Mux final video

NEW:
  1. Identify media                          [NEW]
  2. Try fetch Polish + English subtitles    [NEW]
  3. Decide subtitle source                  [NEW]
     a. Polish fetched + English fetched → use both, skip to step 7
     b. English fetched only → use fetched, translate to Polish, skip to step 7
     c. Polish fetched only → use fetched Polish, extract English (embedded/OCR)
     d. Nothing fetched → current behavior (embedded/OCR + translate)
  4. Extract subtitles (embedded or OCR)     [only if step 3c or 3d]
  5. Parse dialogue lines                    [only if translating]
  6. Translate to Polish                     [only if no Polish from fetch]
  7. Inpaint burned-in subtitles             [if OCR detected any]
  8. Mux final video with Polish + English   [same as before]
```

### Key decisions

- Fetched subtitles are always preferred over extracted/translated ones.
- If fetch returns English but not Polish, we still translate (fetched English is likely better quality than OCR'd English).
- If the video has burned-in subtitles AND we fetched clean English, we still inpaint (to remove the baked-in text before overlaying our clean subtitles).
- OCR is only triggered when no English subtitles are available from any source (fetch or embedded tracks).

### CLI Changes

New flags on `movie-translator`:
- `--no-fetch` -- disable online subtitle fetching (use current behavior)
- No separate `--fetch-only` flag needed; the pipeline naturally skips OCR/translation when fetch succeeds.

### Error Handling

- No API key set: log info message, skip fetch entirely, proceed with current behavior.
- API key set but no username/password: search works, but downloads will fail — log warning with setup instructions, skip fetch.
- Network error / timeout: log warning, skip fetch, proceed.
- Rate limited (429): respect `Retry-After`, retry once. If still limited, skip.
- API returns no results: normal case, fall back to next source.

---

## Testing Strategy

### Unit Tests (fast, no network)

- `identifier/tests/test_hasher.py` -- hash computation against known values
- `identifier/tests/test_parser.py` -- guessit wrapper with representative anime/movie filenames
- `identifier/tests/test_metadata.py` -- ffprobe metadata extraction (mocked)
- `subtitle_fetch/tests/test_opensubtitles.py` -- API client with mocked HTTP responses
- `subtitle_fetch/tests/test_fetcher.py` -- orchestrator ranking logic

### Integration Tests (slow, requires API key + network)

- End-to-end: identify a known video file + search OpenSubtitles
- Marked with `@pytest.mark.integration`

---

## File Structure Summary

```
movie_translator/
  identifier/
    __init__.py          # exports identify_media()
    types.py             # MediaIdentity
    hasher.py            # OpenSubtitles hash
    metadata.py          # ffprobe tag extraction
    parser.py            # guessit wrapper
    tests/
      __init__.py
      test_hasher.py
      test_parser.py
      test_metadata.py
  subtitle_fetch/
    __init__.py          # exports SubtitleFetcher
    types.py             # SubtitleMatch
    fetcher.py           # Orchestrator
    providers/
      __init__.py
      base.py            # SubtitleProvider protocol
      opensubtitles.py   # OpenSubtitles REST v2
    tests/
      __init__.py
      test_opensubtitles.py
      test_fetcher.py
  pipeline.py            # Updated decision logic
  main.py                # New CLI flags
```

## New Dependencies

- `guessit>=3.8` (filename parsing)

No other new dependencies. HTTP via stdlib, everything else already available.
