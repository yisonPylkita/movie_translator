# Pipeline Refactor Design

## Goal

Refactor the movie_translator pipeline for extensibility: support arbitrarily nested directory structures, break the monolithic pipeline into discrete testable stages, properly attribute subtitle sources in track names, and preserve original English subtitles in output.

## Non-Goals

- Adding new target languages beyond Polish
- Config file system
- CLI filtering (--show, --season)
- Changing the translation backend

---

## 1. Directory Scanning & Video Discovery

### Current Problem

`find_video_files()` in `main.py` scans only 1 level deep and mixes file discovery with working directory creation.

### New Design

New module `movie_translator/discovery.py`:

```python
def find_videos(input_path: Path) -> list[Path]:
    """Find all video files recursively from any input.

    - If input_path is a file: return [input_path]
    - If directory: recursively find all .mkv/.mp4 files, sorted
    - Skips hidden directories (starting with '.')
    """

def create_work_dir(video_path: Path, root_input: Path) -> Path:
    """Create temp working directory preserving relative structure.

    e.g., for video at ~/Anime/Show/S1/ep01.mkv with root ~/Anime:
    returns ~/Anime/.translate_temp/Show/S1/ep01/

    Creates candidates/ and reference/ subdirectories.
    """
```

`main.py` becomes pure CLI: parse args, call `find_videos()`, iterate, report results. No more `find_video_files_with_temp_dirs` or `create_working_dirs` in main.

The existing `has_polish_subtitles()` skip check stays in `main.py`'s loop — it's a pre-pipeline gate, not a pipeline stage.

---

## 2. Pipeline Context

A dataclass that accumulates state as it flows through stages:

```python
@dataclass
class PipelineContext:
    # Inputs (set before pipeline runs)
    video_path: Path
    work_dir: Path
    config: PipelineConfig

    # Stage outputs (set progressively)
    identity: MediaIdentity | None = None
    reference_path: Path | None = None
    fetched_subtitles: dict[str, FetchedSubtitle] | None = None  # lang -> (path, source)
    english_source: Path | None = None
    original_english_track: OriginalTrack | None = None  # track info for preservation
    dialogue_lines: list[DialogueLine] | None = None
    translated_lines: list[DialogueLine] | None = None
    font_info: FontInfo | None = None
    subtitle_tracks: list[SubtitleFile] | None = None
    ocr_results: list[OCRResult] | None = None
    inpainted_video: Path | None = None


@dataclass
class PipelineConfig:
    device: str = "mps"
    batch_size: int = 16
    model: str = "allegro"
    enable_fetch: bool = True
    dry_run: bool = False


@dataclass
class FetchedSubtitle:
    path: Path
    source: str  # provider name, e.g. "animesub"


@dataclass
class FontInfo:
    supports_polish: bool
    font_attachments: list[Path]
    fallback_font_family: str | None


@dataclass
class OriginalTrack:
    stream_index: int
    codec: str  # "subrip", "ass", etc.
    language: str
```

---

## 3. Stage Interface & Pipeline Orchestrator

### Stage Protocol

```python
class Stage(Protocol):
    name: str

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute stage, returning updated context."""
        ...
```

### Pipeline Orchestrator

`pipeline.py` becomes ~50 lines of glue:

```python
class TranslationPipeline:
    def __init__(self, config: PipelineConfig, tracker=None):
        self.stages = [
            IdentifyStage(),
            ExtractReferenceStage(),
            FetchSubtitlesStage(),
            ExtractEnglishStage(),
            TranslateStage(),
            CreateTracksStage(),
            MuxStage(),
        ]

    def process(self, video_path: Path, work_dir: Path) -> bool:
        ctx = PipelineContext(video_path, work_dir, self.config)
        for stage in self.stages:
            self._update_progress(stage.name)
            ctx = stage.run(ctx)
        return True
```

### Stages Directory

```
movie_translator/stages/
├── __init__.py
├── identify.py         # IdentifyStage
├── extract_ref.py      # ExtractReferenceStage
├── fetch.py            # FetchSubtitlesStage
├── extract_english.py  # ExtractEnglishStage
├── translate.py        # TranslateStage
├── create_tracks.py    # CreateTracksStage
└── mux.py              # MuxStage
```

---

## 4. Stage Responsibilities

### IdentifyStage

Calls `identify_media()`. Sets `ctx.identity`.

### ExtractReferenceStage

Extracts embedded English subtitle for validation. Sets `ctx.reference_path`. Also records `ctx.original_english_track` with the stream index and codec so the Mux stage can preserve it.

### FetchSubtitlesStage

Builds the fetcher with all configured providers, searches in parallel, downloads candidates, validates against reference, selects best per language. Sets `ctx.fetched_subtitles` as `{"pol": FetchedSubtitle(path, "animesub"), "eng": FetchedSubtitle(path, "opensubtitles")}`.

### ExtractEnglishStage

Determines the English subtitle source (priority: fetched English > reference > embedded > OCR). Extracts dialogue lines via `SubtitleProcessor.extract_dialogue_lines()`. Sets `ctx.english_source` and `ctx.dialogue_lines`.

### TranslateStage

Runs AI translation and font checking in parallel (ThreadPoolExecutor). Sets `ctx.translated_lines` and `ctx.font_info`.

### CreateTracksStage

Creates ASS subtitle files and builds the track list with source-attributed names:

1. If fetched Polish exists: `SubtitleFile(path, "pol", "Polish (animesub)", is_default=True)`
2. AI Polish: `SubtitleFile(path, "pol", "Polish (AI)", is_default=not fetched_pol)`

Applies font overrides to Polish tracks if needed. Sets `ctx.subtitle_tracks`.

Note: The original English track is NOT part of `ctx.subtitle_tracks` — it's preserved directly from the source video by the MuxStage using `ctx.original_english_track`.

### MuxStage

Muxes the final video:
- Preserves the original English subtitle track (by stream index from `ctx.original_english_track`)
- Strips all other original subtitle tracks
- Adds our generated tracks
- Attaches fonts if needed
- Uses mkvmerge for MKV, ffmpeg for MP4
- Handles inpainting source swap if `ctx.inpainted_video` is set
- Replaces original file (unless dry_run)

---

## 5. Original English Subtitle Preservation

### Current Behavior

All original subtitle tracks are stripped. A regenerated "English Dialogue" clean track is added.

### New Behavior

The original English subtitle track is preserved as-is (no re-encoding, no filtering) with the title `"English (Original)"`. This keeps the original formatting, signs, songs — everything the source had.

The AI-generated tracks are additions on top.

### Implementation

For **mkvmerge**: Instead of `--no-subtitles`, use `--subtitle-tracks <track_id>` to keep only the English track, then add our generated tracks. Set `--track-name` to `"English (Original)"` for the preserved track.

For **ffmpeg** fallback: Map the specific subtitle stream with `-map 0:s:<index>`, set metadata title to `"English (Original)"`.

### Edge Cases

- **No embedded English track**: No original to preserve. Output only has Polish (AI) + optionally fetched Polish.
- **Multiple English tracks** (dialogue + signs): Preserve only the dialogue track selected by `SubtitleExtractor.find_english_track()`.
- **Image-based subtitles** (PGS/VOBSUB): Still preserve — they're the original, even if large.

---

## 6. Track Naming Convention

All tracks in the output indicate their origin:

| Track | Name Pattern | Example |
|-------|-------------|---------|
| Fetched Polish | `Polish ({source})` | `Polish (animesub)` |
| AI translated | `Polish (AI)` | `Polish (AI)` |
| Original English | `English (Original)` | `English (Original)` |

Default track priority:
1. Fetched Polish (if available)
2. Polish (AI) (if no fetched Polish)

All other tracks have default=0.

---

## 7. Module Structure Changes

### New Files

| File | Purpose |
|------|---------|
| `discovery.py` | Recursive video finding + work dir creation |
| `context.py` | PipelineContext, PipelineConfig, supporting dataclasses |
| `stages/__init__.py` | Exports all stages |
| `stages/identify.py` | IdentifyStage |
| `stages/extract_ref.py` | ExtractReferenceStage |
| `stages/fetch.py` | FetchSubtitlesStage |
| `stages/extract_english.py` | ExtractEnglishStage |
| `stages/translate.py` | TranslateStage |
| `stages/create_tracks.py` | CreateTracksStage |
| `stages/mux.py` | MuxStage |

### Modified Files

| File | Change |
|------|--------|
| `main.py` | Slim down to pure CLI; use `discovery.find_videos()` and new pipeline |
| `pipeline.py` | Replace 443-line monolith with thin stage orchestrator (~50 lines) |
| `ffmpeg.py` | Update `mux_video_with_subtitles` to support preserving a specific original subtitle track |
| `video/operations.py` | Update `create_clean_video` signature to accept original track preservation |

### Unchanged Modules

These modules are already well-structured and stay as-is:

- `identifier/` — media identification
- `subtitles/` — extraction and processing
- `subtitle_fetch/` — fetching and validation (providers, fetcher, validator)
- `translation/` — AI translation
- `ocr/` — burned-in subtitle extraction
- `inpainting/` — subtitle removal
- `fonts.py` — font detection
- `types.py` — core types (add new dataclasses to `context.py` instead)
- `logging.py` — logger setup
- `progress.py` — TUI progress

### Deleted Code

- `find_video_files()` and `find_video_files_with_temp_dirs()` from `main.py` (replaced by `discovery.py`)
- `create_working_dirs()` from `main.py` (replaced by `discovery.py`)
- The bulk of `TranslationPipeline.process_video_file()` (logic moves to stages)
- `TranslationPipeline._build_fetcher()`, `_extract_reference()`, `_search_and_validate()`, `_extract_subtitles()`, `_extract_burned_in_subtitles()`, `_replace_original()` (all move to stages)

---

## 8. Testing Strategy

Each stage gets its own test file under `movie_translator/stages/tests/`:

```
stages/tests/
├── test_identify.py
├── test_extract_ref.py
├── test_fetch.py
├── test_extract_english.py
├── test_translate.py
├── test_create_tracks.py
└── test_mux.py
```

**Stage tests** mock the underlying domain modules (identifier, subtitles, subtitle_fetch, etc.) and verify:
- Correct context fields are set
- Edge cases (missing tracks, failed fetch, OCR fallback)
- Error propagation

**Discovery tests** in `movie_translator/tests/test_discovery.py`:
- Single file input
- Flat directory
- Nested anime/season/episode structure
- Hidden directory skipping
- Mixed file types

**Integration tests** (`tests/test_integration.py`): Update existing tests to use the new pipeline API.

---

## 9. Migration Path

The refactor can be done incrementally:

1. **Create `context.py` and `stages/` directory** with the stage interface
2. **Extract each stage** from `pipeline.py` one at a time, keeping the old pipeline working until all stages are extracted
3. **Create `discovery.py`** and update `main.py`
4. **Update muxing** to preserve original English track
5. **Delete old code** from pipeline.py once all stages work
6. **Update tests** to cover new structure

Each step can be tested independently. The old pipeline continues working until the switchover.
