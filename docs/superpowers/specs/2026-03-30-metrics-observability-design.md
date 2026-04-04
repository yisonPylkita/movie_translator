# Metrics & Observability Framework

**Date:** 2026-03-30
**Status:** Approved

## Problem

The pipeline has no structured performance data. We introduce refactors and optimizations but have no way to measure their impact. We need full transparency into what takes how long, per stage and sub-operation, and the ability to compare performance across code versions.

## Goals

1. Instrument every meaningful pipeline operation with structured timing spans.
2. Produce a JSON report per run with per-video entries, media identity, and git context.
3. Provide a CLI tool to compare two reports and show deltas.
4. Store baseline reports in git alongside the tooling for historical reference.
5. Zero overhead when metrics collection is disabled (production default).

## Non-goals

- Real-time streaming / JSONL output (may add later).
- Distributed tracing or OpenTelemetry integration.
- Automatic regression detection or CI gating.

## Architecture

### Event-based with context manager API

The instrumentation uses an event-based architecture. Stages emit timing events to a collector via context managers. The collector dispatches events to pluggable listeners (report builder, TUI progress bridge, etc.).

Key design choices:

- **Context managers as primary API** — solves the paired start/end problem. `__exit__` always fires, even on exceptions, so spans are never left open.
- **Event bus with listeners** — decouples instrumentation from reporting. The same events can feed both the JSON report and the Rich TUI progress display.
- **Auto-nesting via `ContextVar`** — child spans automatically inherit their parent's name as a prefix. Stages use local names (`"search_all"`, `"animesub"`) and the full dotted path (`"fetch.search_all.animesub"`) is built automatically.
- **Thread-safe** — `ContextVar` propagates correctly into `ThreadPoolExecutor` worker threads, so parallel spans (e.g. `translate.batch` and `translate.check_fonts`) don't collide.
- **NullCollector** — when `--metrics` is not passed, `ctx.metrics` is a no-op implementation with zero overhead.

### Data flow

```
Stage code                   MetricsCollector              Listeners
──────────                   ────────────────              ─────────
with ctx.metrics.span(name)
  │
  ├─ __enter__: record start time, push prefix
  │
  ├─ stage work happens...
  │   span.detail(key, value)
  │
  └─ __exit__: pop prefix, compute duration
       │
       └─ emit(SpanEvent) ──────────────────> ReportBuilder.on_event()
                                              ProgressBridge.on_event()
```

## Module structure

```
movie_translator/metrics/
├── __init__.py          # Public API: MetricsCollector, NullCollector, Span
├── collector.py         # MetricsCollector, NullCollector, Span, ContextVar prefix stack
├── events.py            # SpanEvent dataclass
├── listeners.py         # ReportBuilder, ProgressBridge
├── report.py            # Report schema: load, save, serialize, git metadata
├── compare.py           # Comparison logic + CLI table output
└── __main__.py          # CLI: python -m movie_translator.metrics compare ...
```

Baselines:

```
movie_translator/metrics/results/
├── 2026-03-30_a1b2c3d.json
├── 2026-04-02_f8e9d0c.json
└── ...
```

## Core components

### SpanEvent

```python
@dataclass
class SpanEvent:
    name: str            # Dotted path, e.g. "fetch.search_all.animesub"
    duration_ms: float   # Wall-clock milliseconds
    details: dict        # Stage-specific metadata, optional
```

### MetricsCollector

```python
class MetricsCollector:
    _listeners: list[Callable[[SpanEvent], None]]
    _prefix: ContextVar[str]  # Thread-local span name stack

    def span(self, name: str) -> Span:
        """Context manager. Builds full dotted name from ContextVar prefix."""

    def emit(self, event: SpanEvent):
        """Dispatch event to all registered listeners."""

    def add_listener(self, fn: Callable[[SpanEvent], None]):
        """Register an event consumer."""
```

### Span

```python
class Span:
    def __enter__(self) -> Span:
        """Record start time, push name onto ContextVar prefix stack."""

    def __exit__(self, *exc) -> bool:
        """Pop prefix, compute duration, emit SpanEvent. Never suppresses exceptions."""

    def detail(self, key: str, value: Any):
        """Attach stage-specific metadata to this span."""
```

### NullCollector

```python
class NullCollector:
    """No-op implementation. All methods are instant no-ops. Zero overhead."""

    def span(self, name: str) -> NullSpan: ...
    def emit(self, event: SpanEvent): pass
    def add_listener(self, fn): pass
```

`NullSpan.__enter__` returns itself, `__exit__` does nothing, `detail()` does nothing.

### ReportBuilder (listener)

Accumulates `SpanEvent`s into a per-video entry list. Exposes `start_video(path, identity, hash, ...)` and `end_video()` methods called by the pipeline orchestrator to demarcate per-video sections. Events emitted between `start_video` and `end_video` are grouped into that video's entries. After all videos are processed, `save()` serializes to the report JSON.

### ProgressBridge (listener)

Forwards span start/completion events to the existing `ProgressTracker` TUI. Replaces the current manual `tracker.set_stage()` calls. Top-level stage spans (where the name has no dots) map to TUI stage transitions.

## Report schema

```json
{
  "version": 1,
  "git_commit": "a1b2c3d",
  "dirty": false,
  "timestamp": "2026-03-30T14:22:01Z",
  "config": {
    "device": "mps",
    "batch_size": 16,
    "model": "allegro",
    "enable_fetch": true,
    "enable_inpaint": false
  },
  "videos": [
    {
      "path": "Konosuba/Season 1/01.mkv",
      "hash": "a1b2c3d4e5f6g7h8",
      "file_size_bytes": 524288000,
      "duration_ms": 1440000,
      "identity": {
        "title": "Kono Subarashii Sekai ni Shukufuku wo!",
        "parsed_title": "Konosuba",
        "media_type": "episode",
        "season": 1,
        "episode": 1,
        "year": 2016,
        "is_anime": true,
        "release_group": "Erai-raws",
        "imdb_id": "tt5463826",
        "tmdb_id": 65249
      },
      "total_duration_ms": 108200,
      "entries": [
        {
          "name": "identify",
          "duration_ms": 210
        },
        {
          "name": "identify.parse_filename",
          "duration_ms": 2
        },
        {
          "name": "translate.batch",
          "duration_ms": 85000,
          "details": {
            "input_lines": 342,
            "input_chars": 18400,
            "output_lines": 342,
            "output_chars": 21200,
            "batches": 22,
            "batch_size": 16
          }
        }
      ]
    }
  ]
}
```

Schema notes:

- `version` — integer, incremented on breaking schema changes. The compare tool checks this.
- `git_commit` — short hash from `git rev-parse --short HEAD`.
- `dirty` — `true` if `git status --porcelain` produces output (uncommitted changes).
- `config` — the `PipelineConfig` fields, so you know what settings produced these numbers.
- `videos[].hash` — the OpenSubtitles hash (`oshash`) already computed by the identify stage.
- `videos[].total_duration_ms` — wall-clock time for the entire video, measured by the pipeline orchestrator. Not the sum of entries (parallel spans mean the sum exceeds wall clock).
- `entries` — flat list, only present for operations that actually ran. No placeholders for skipped operations. Presence/absence of entries reveals the pipeline profile.
- `entries[].details` — optional dict, only present when there's stage-specific data to report.

## Pipeline integration

### Wiring in pipeline.py

```python
# Automatic stage-level timing — no changes needed in stage code
for stage in self.stages:
    with ctx.metrics.span(stage.name):
        ctx = stage.run(ctx)
```

### Wiring in main.py

```python
# CLI flag: --metrics
if args.metrics:
    collector = MetricsCollector()
    report_builder = ReportBuilder()
    collector.add_listener(report_builder.on_event)
else:
    collector = NullCollector()

# Pass collector into PipelineContext
ctx = PipelineContext(video_path=..., work_dir=..., config=..., metrics=collector)

# After all videos processed:
if args.metrics:
    report_builder.save(work_dir / "metrics.json", config=pipeline.config)
```

### Stage instrumentation example

```python
# Inside FetchSubtitlesStage.run():
with ctx.metrics.span("search_all"):
    for provider in providers:
        with ctx.metrics.span(provider.name) as s:
            results = provider.search(ctx.identity, languages)
            s.detail("candidates", len(results))

with ctx.metrics.span("download_all") as s:
    downloaded = self._download_all(fetcher, all_matches, candidates_dir)
    s.detail("downloaded", len(downloaded))
    s.detail("failed", len(all_matches) - len(downloaded))

with ctx.metrics.span("validate_and_select") as s:
    validated = validator.validate_candidates(downloaded)
    s.detail("passed", len(validated))
    s.detail("rejected", len(downloaded) - len(validated))
    if validated:
        s.detail("best_score", validated[0][2])
```

Note: the stage doesn't wrap itself in `span("fetch")` — the pipeline orchestrator does that. So `span("search_all")` inside the fetch stage automatically becomes `"fetch.search_all"` via the `ContextVar` prefix.

## Operation catalog

Every span that should be instrumented, with its details:

### identify

| Span | Details |
|------|---------|
| `identify.parse_filename` | |
| `identify.extract_container_metadata` | |
| `identify.compute_oshash` | |
| `identify.lookup_tmdb` | `hit: bool` |

### extract_reference

| Span | Details |
|------|---------|
| `extract_reference.get_track_info` | |
| `extract_reference.extract_pgs_track` | `subtitle_count: int` |
| `extract_reference.extract_subtitle` | |
| `extract_reference.probe_burned_in` | `detected: bool` |
| `extract_reference.extract_burned_in` | `frames: int` |

### fetch

| Span | Details |
|------|---------|
| `fetch.search_all` | |
| `fetch.search_all.<provider>` | `candidates: int` |
| `fetch.download_all` | `downloaded: int, failed: int` |
| `fetch.validate_and_select` | `passed: int, rejected: int, best_score: float` |
| `fetch.align` | `method: str, offset_ms: int` |

### extract_english (named `extract` in stage.name)

| Span | Details |
|------|---------|
| `extract.select_source` | `source: "fetched"\|"reference"\|"embedded"\|"ocr"` |
| `extract.extract_subtitle` | |
| `extract.extract_pgs_track` | `subtitle_count: int` |
| `extract.probe_burned_in` | `detected: bool` |
| `extract.extract_burned_in` | `frames: int` |
| `extract.extract_dialogue_lines` | `lines: int, chars: int` |

### translate

| Span | Details |
|------|---------|
| `translate.load_model` | `cached: bool` |
| `translate.batch` | `input_lines: int, input_chars: int, output_lines: int, output_chars: int, batches: int, batch_size: int` |
| `translate.check_fonts` | `supports_polish: bool, fallback_font: str\|null` |

### create_tracks

| Span | Details |
|------|---------|
| `create_tracks.create_polish_subtitles` | |
| `create_tracks.override_font` | |
| `create_tracks.build_track_list` | `tracks: int` |

### mux

| Span | Details |
|------|---------|
| `mux.inpaint` | `frames: int` |
| `mux.create_clean_video` | `tracks: int, font_attachments: int` |
| `mux.verify_result` | |
| `mux.replace_original` | |

## Comparison CLI

### Invocation

```bash
python -m movie_translator.metrics compare <before.json> <after.json>
```

### Video matching

Videos are matched across reports by:
1. **Hash match** — same `hash` field (identical file).
2. **Identity match** — same `media_type` + `title` + `season` + `episode` (same content, possibly different encode).

Videos with different pipeline profiles (e.g. one has `extract_reference.extract_pgs_track` entries and the other doesn't) are flagged and excluded from aggregation, because they exercised different code paths.

### Aggregation

When multiple videos match across both reports, durations are averaged across matched pairs per span name. Only span names present in both reports are compared.

### Output format

```
Comparing: a1b2c3d (2026-03-28) -> e4f5g6h (2026-03-30)
Config: device=mps, model=allegro, batch_size=16
Dirty: before=no, after=no

Videos matched: 3/3 (0 excluded for different profiles)

Stage                                   Before      After      Delta
--------------------------------------------------------------------------
identify                                  210ms      180ms       -14%
identify.extract_container_metadata       150ms      130ms       -13%
fetch                                    5200ms     4800ms        -8%
fetch.search_all.animesub                1200ms      900ms       -25%
fetch.download_all                       3400ms     3100ms        -9%
translate                               89500ms    66500ms       -26%
translate.load_model                     4500ms     4500ms         0%
translate.batch                         85000ms    62000ms       -27%
mux                                     12200ms    11800ms        -3%
mux.create_clean_video                  12000ms    11600ms        -3%
--------------------------------------------------------------------------
Total (wall clock)                     108200ms    83500ms       -23%

Summary: 23% faster overall. Biggest win: translate.batch (-27%, -23000ms).
3 videos compared. 0 excluded.
```

Design notes:

- Parent spans (e.g. `translate`) and their children (e.g. `translate.batch`) both appear. This shows where time shifted within a stage.
- Delta column shows percentage change. Negative = faster (improvement).
- Summary line gives a natural-language verdict readable by both humans and LLMs.
- If `dirty` is `true` on either side, the output includes a warning.

## File naming and storage

Report files in `movie_translator/metrics/results/`:

```
<YYYY-MM-DD>_<short-git-hash>.json
```

Examples:
- `2026-03-30_a1b2c3d.json`
- `2026-04-02_f8e9d0c.json`

If the repo is dirty, the filename still uses the current HEAD hash but the `dirty: true` flag inside the report makes this explicit.

## Testing

- **Unit tests** for `MetricsCollector`: verify span nesting produces correct dotted names, details are attached, events reach listeners, `NullCollector` is truly no-op.
- **Unit tests** for `ReportBuilder`: verify serialization matches the schema.
- **Unit tests** for `compare.py`: verify video matching, aggregation, and table output with known inputs.
- **Integration test**: run a small pipeline segment with metrics enabled, verify the report file is written and parseable.
