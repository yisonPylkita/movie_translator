# Metrics & Observability Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured performance instrumentation to every pipeline stage, producing a JSON report per run with a CLI comparison tool.

**Architecture:** Event-based collector with context manager API. Stages emit `SpanEvent`s via `with ctx.metrics.span(name)`. Auto-nesting via `ContextVar` builds dotted span names. Pluggable listeners consume events (ReportBuilder for JSON, ProgressBridge for TUI). NullCollector provides zero-overhead no-op when disabled.

**Tech Stack:** Python stdlib (`dataclasses`, `contextvars`, `time`, `json`, `subprocess`). No new dependencies.

**Deferred from spec (follow-up instrumentation):**
- `identify.parse_filename`, `identify.extract_container_metadata`, `identify.compute_oshash`, `identify.lookup_tmdb` — requires modifying `identifier/identify.py` to accept metrics. The identify stage is fast (<200ms), so granular sub-timing is low priority.
- `translate.load_model` with `cached: bool` — requires modifying `translation/translator.py`. Can be added when profiling translation performance.
- `fetch.search_all.<provider>` — per-provider timing requires modifying `SubtitleFetcher.search_all()`. Can be added when optimizing fetch latency.

---

## File Structure

```
movie_translator/metrics/
├── __init__.py          # Re-exports: MetricsCollector, NullCollector, Span, SpanEvent
├── collector.py         # MetricsCollector, NullCollector, Span, NullSpan
├── events.py            # SpanEvent dataclass
├── listeners.py         # ReportBuilder, ProgressBridge
├── report.py            # Report load/save/serialize, git metadata helpers
├── compare.py           # Comparison logic + table rendering
├── __main__.py          # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_collector.py
│   ├── test_listeners.py
│   ├── test_report.py
│   └── test_compare.py
└── results/             # Committed baselines (empty initially)
    └── .gitkeep
```

Modified files:
- `movie_translator/context.py` — add `metrics` field to `PipelineContext`
- `movie_translator/pipeline.py` — wrap stage.run() with metrics spans
- `movie_translator/main.py` — add `--metrics` flag, wire collector + report save
- `movie_translator/stages/identify.py` — add sub-operation spans
- `movie_translator/stages/extract_ref.py` — add sub-operation spans
- `movie_translator/stages/fetch.py` — add sub-operation spans
- `movie_translator/stages/extract_english.py` — add sub-operation spans
- `movie_translator/stages/translate.py` — add sub-operation spans
- `movie_translator/stages/create_tracks.py` — add sub-operation spans
- `movie_translator/stages/mux.py` — add sub-operation spans

---

### Task 1: SpanEvent dataclass

**Files:**
- Create: `movie_translator/metrics/__init__.py`
- Create: `movie_translator/metrics/events.py`
- Create: `movie_translator/metrics/tests/__init__.py`
- Create: `movie_translator/metrics/tests/test_collector.py`

- [ ] **Step 1: Create the metrics package with SpanEvent**

Create `movie_translator/metrics/events.py`:

```python
"""Metric event types emitted by the instrumentation system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanEvent:
    """A completed timing span emitted by the collector."""

    name: str  # Dotted path, e.g. "fetch.search_all.animesub"
    duration_ms: float  # Wall-clock milliseconds
    details: dict[str, Any] = field(default_factory=dict)
```

Create `movie_translator/metrics/__init__.py` (minimal, will grow):

```python
"""Pipeline metrics and observability."""

from .events import SpanEvent

__all__ = ['SpanEvent']
```

Create `movie_translator/metrics/tests/__init__.py` (empty).

- [ ] **Step 2: Write test for SpanEvent**

Create `movie_translator/metrics/tests/test_collector.py`:

```python
from movie_translator.metrics.events import SpanEvent


class TestSpanEvent:
    def test_create_with_defaults(self):
        event = SpanEvent(name='fetch.download', duration_ms=1200.5)
        assert event.name == 'fetch.download'
        assert event.duration_ms == 1200.5
        assert event.details == {}

    def test_create_with_details(self):
        event = SpanEvent(
            name='translate.batch',
            duration_ms=85000,
            details={'input_lines': 342, 'batches': 22},
        )
        assert event.details['input_lines'] == 342
        assert event.details['batches'] == 22
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_collector.py -v`
Expected: 2 tests PASS

- [ ] **Step 4: Commit**

```bash
git add movie_translator/metrics/__init__.py movie_translator/metrics/events.py movie_translator/metrics/tests/__init__.py movie_translator/metrics/tests/test_collector.py
git commit -m "feat(metrics): add SpanEvent dataclass"
```

---

### Task 2: MetricsCollector and Span

**Files:**
- Create: `movie_translator/metrics/collector.py`
- Modify: `movie_translator/metrics/__init__.py`
- Modify: `movie_translator/metrics/tests/test_collector.py`

- [ ] **Step 1: Write failing tests for MetricsCollector**

Append to `movie_translator/metrics/tests/test_collector.py`:

```python
import time
from concurrent.futures import ThreadPoolExecutor

from movie_translator.metrics.collector import MetricsCollector, NullCollector


class TestMetricsCollector:
    def test_span_emits_event_to_listener(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('identify'):
            pass

        assert len(events) == 1
        assert events[0].name == 'identify'
        assert events[0].duration_ms >= 0

    def test_span_records_duration(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('slow_op'):
            time.sleep(0.05)

        assert events[0].duration_ms >= 40  # at least 40ms (allow some slack)

    def test_nested_spans_build_dotted_names(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch'):
            with collector.span('search_all'):
                with collector.span('animesub'):
                    pass

        names = [e.name for e in events]
        assert 'fetch.search_all.animesub' in names
        assert 'fetch.search_all' in names
        assert 'fetch' in names

    def test_nested_spans_emit_inner_first(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch'):
            with collector.span('download'):
                pass

        assert events[0].name == 'fetch.download'
        assert events[1].name == 'fetch'

    def test_span_detail_attaches_metadata(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('fetch.download') as s:
            s.detail('downloaded', 6)
            s.detail('failed', 1)

        assert events[0].details == {'downloaded': 6, 'failed': 1}

    def test_span_emits_on_exception(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        try:
            with collector.span('failing_op'):
                raise ValueError('boom')
        except ValueError:
            pass

        assert len(events) == 1
        assert events[0].name == 'failing_op'
        assert events[0].duration_ms >= 0

    def test_prefix_restored_after_exception(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        with collector.span('parent'):
            try:
                with collector.span('child'):
                    raise ValueError('boom')
            except ValueError:
                pass
            with collector.span('sibling'):
                pass

        names = [e.name for e in events]
        assert 'parent.child' in names
        assert 'parent.sibling' in names

    def test_parallel_spans_in_threads(self):
        collector = MetricsCollector()
        events = []
        collector.add_listener(events.append)

        def work_a():
            with collector.span('task_a'):
                time.sleep(0.01)

        def work_b():
            with collector.span('task_b'):
                time.sleep(0.01)

        with collector.span('parent'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(work_a)
                fb = pool.submit(work_b)
                fa.result()
                fb.result()

        names = [e.name for e in events]
        assert 'parent.task_a' in names
        assert 'parent.task_b' in names
        assert 'parent' in names

    def test_multiple_listeners(self):
        collector = MetricsCollector()
        events_a = []
        events_b = []
        collector.add_listener(events_a.append)
        collector.add_listener(events_b.append)

        with collector.span('op'):
            pass

        assert len(events_a) == 1
        assert len(events_b) == 1


class TestNullCollector:
    def test_span_is_noop(self):
        collector = NullCollector()
        with collector.span('anything') as s:
            s.detail('key', 'value')
        # No error, no events — just a no-op

    def test_add_listener_is_noop(self):
        collector = NullCollector()
        collector.add_listener(lambda e: None)
        # No error

    def test_emit_is_noop(self):
        collector = NullCollector()
        collector.emit(SpanEvent(name='x', duration_ms=0))
        # No error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_collector.py -v -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'movie_translator.metrics.collector'`

- [ ] **Step 3: Implement MetricsCollector, Span, NullCollector, NullSpan**

Create `movie_translator/metrics/collector.py`:

```python
"""Metrics collector with context-manager span API."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from .events import SpanEvent

_prefix: ContextVar[str] = ContextVar('metrics_prefix', default='')


class Span:
    """Context manager that times an operation and emits a SpanEvent on exit."""

    __slots__ = ('_collector', '_name', '_details', '_start', '_token')

    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self._collector = collector
        self._name = name
        self._details: dict[str, Any] = {}
        self._start: float = 0
        self._token = None

    def __enter__(self) -> Span:
        self._start = time.perf_counter()
        self._token = _prefix.set(self._name)
        return self

    def __exit__(self, *exc: object) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000
        _prefix.reset(self._token)
        self._collector.emit(SpanEvent(
            name=self._name,
            duration_ms=duration_ms,
            details=self._details,
        ))
        return False

    def detail(self, key: str, value: Any) -> None:
        """Attach stage-specific metadata to this span."""
        self._details[key] = value


class MetricsCollector:
    """Event-based metrics collector with auto-nesting span names."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[SpanEvent], None]] = []

    def span(self, name: str) -> Span:
        """Return a context manager that times an operation.

        Span names are automatically prefixed with the current parent span
        name, building dotted paths like ``fetch.search_all.animesub``.
        """
        prefix = _prefix.get()
        full_name = f'{prefix}.{name}' if prefix else name
        return Span(self, full_name)

    def emit(self, event: SpanEvent) -> None:
        """Dispatch an event to all registered listeners."""
        for listener in self._listeners:
            listener(event)

    def add_listener(self, fn: Callable[[SpanEvent], None]) -> None:
        """Register an event consumer."""
        self._listeners.append(fn)


class _NullSpan:
    """No-op span that does nothing."""

    __slots__ = ()

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def detail(self, key: str, value: Any) -> None:
        pass


_NULL_SPAN = _NullSpan()


class NullCollector:
    """No-op collector. Zero overhead when metrics are disabled."""

    def span(self, name: str) -> _NullSpan:
        return _NULL_SPAN

    def emit(self, event: SpanEvent) -> None:
        pass

    def add_listener(self, fn: Callable[[SpanEvent], None]) -> None:
        pass
```

- [ ] **Step 4: Update `__init__.py` exports**

Update `movie_translator/metrics/__init__.py`:

```python
"""Pipeline metrics and observability."""

from .collector import MetricsCollector, NullCollector, Span
from .events import SpanEvent

__all__ = ['MetricsCollector', 'NullCollector', 'Span', 'SpanEvent']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_collector.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/metrics/collector.py movie_translator/metrics/__init__.py movie_translator/metrics/tests/test_collector.py
git commit -m "feat(metrics): add MetricsCollector with auto-nesting spans"
```

---

### Task 3: ReportBuilder listener

**Files:**
- Create: `movie_translator/metrics/listeners.py`
- Create: `movie_translator/metrics/tests/test_listeners.py`
- Modify: `movie_translator/metrics/__init__.py`

- [ ] **Step 1: Write failing tests for ReportBuilder**

Create `movie_translator/metrics/tests/test_listeners.py`:

```python
from movie_translator.metrics.events import SpanEvent
from movie_translator.metrics.listeners import ReportBuilder


class TestReportBuilder:
    def test_start_and_end_video(self):
        rb = ReportBuilder()
        rb.start_video(
            path='Konosuba/01.mkv',
            hash='abc123',
            file_size_bytes=500_000_000,
            duration_ms=1_440_000,
            identity={'title': 'Konosuba', 'media_type': 'episode', 'season': 1, 'episode': 1},
        )
        rb.on_event(SpanEvent(name='identify', duration_ms=200))
        rb.end_video()

        assert len(rb.videos) == 1
        video = rb.videos[0]
        assert video['path'] == 'Konosuba/01.mkv'
        assert video['hash'] == 'abc123'
        assert video['file_size_bytes'] == 500_000_000
        assert video['duration_ms'] == 1_440_000
        assert video['identity']['title'] == 'Konosuba'
        assert len(video['entries']) == 1
        assert video['entries'][0]['name'] == 'identify'
        assert video['entries'][0]['duration_ms'] == 200

    def test_entries_only_in_current_video(self):
        rb = ReportBuilder()
        rb.start_video(path='ep01.mkv', hash='aaa', file_size_bytes=100, duration_ms=1000, identity={})
        rb.on_event(SpanEvent(name='identify', duration_ms=10))
        rb.end_video()

        rb.start_video(path='ep02.mkv', hash='bbb', file_size_bytes=200, duration_ms=2000, identity={})
        rb.on_event(SpanEvent(name='fetch', duration_ms=20))
        rb.end_video()

        assert len(rb.videos[0]['entries']) == 1
        assert rb.videos[0]['entries'][0]['name'] == 'identify'
        assert len(rb.videos[1]['entries']) == 1
        assert rb.videos[1]['entries'][0]['name'] == 'fetch'

    def test_entry_without_details_omits_key(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.on_event(SpanEvent(name='op', duration_ms=5))
        rb.end_video()

        entry = rb.videos[0]['entries'][0]
        assert 'details' not in entry

    def test_entry_with_details_includes_them(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.on_event(SpanEvent(name='op', duration_ms=5, details={'count': 3}))
        rb.end_video()

        entry = rb.videos[0]['entries'][0]
        assert entry['details'] == {'count': 3}

    def test_total_duration_set_on_end_video(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='x', file_size_bytes=1, duration_ms=1, identity={})
        rb.end_video()

        assert 'total_duration_ms' in rb.videos[0]
        assert rb.videos[0]['total_duration_ms'] >= 0

    def test_events_outside_video_are_ignored(self):
        rb = ReportBuilder()
        rb.on_event(SpanEvent(name='stray', duration_ms=1))
        assert len(rb.videos) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_listeners.py -v -x`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ReportBuilder**

Create `movie_translator/metrics/listeners.py`:

```python
"""Event listeners that consume SpanEvents."""

from __future__ import annotations

import time
from typing import Any

from .events import SpanEvent


class ReportBuilder:
    """Accumulates SpanEvents into per-video entry lists for JSON serialization."""

    def __init__(self) -> None:
        self.videos: list[dict[str, Any]] = []
        self._current_video: dict[str, Any] | None = None
        self._video_start: float = 0

    def start_video(
        self,
        *,
        path: str,
        hash: str,
        file_size_bytes: int,
        duration_ms: int,
        identity: dict[str, Any],
    ) -> None:
        """Begin collecting entries for a new video."""
        self._current_video = {
            'path': path,
            'hash': hash,
            'file_size_bytes': file_size_bytes,
            'duration_ms': duration_ms,
            'identity': identity,
            'entries': [],
        }
        self._video_start = time.perf_counter()

    def end_video(self) -> None:
        """Finalize the current video and add it to the report."""
        if self._current_video is None:
            return
        total_ms = (time.perf_counter() - self._video_start) * 1000
        self._current_video['total_duration_ms'] = round(total_ms, 1)
        self.videos.append(self._current_video)
        self._current_video = None

    def on_event(self, event: SpanEvent) -> None:
        """Listener callback — accumulates span into current video's entries."""
        if self._current_video is None:
            return
        entry: dict[str, Any] = {
            'name': event.name,
            'duration_ms': round(event.duration_ms, 1),
        }
        if event.details:
            entry['details'] = event.details
        self._current_video['entries'].append(entry)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_listeners.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/metrics/listeners.py movie_translator/metrics/tests/test_listeners.py
git commit -m "feat(metrics): add ReportBuilder listener"
```

---

### Task 4: Report serialization with git metadata

**Files:**
- Create: `movie_translator/metrics/report.py`
- Create: `movie_translator/metrics/tests/test_report.py`

- [ ] **Step 1: Write failing tests for report save/load**

Create `movie_translator/metrics/tests/test_report.py`:

```python
import json
from unittest.mock import patch

from movie_translator.metrics.report import build_report, load_report, save_report


class TestBuildReport:
    def test_includes_version(self):
        report = build_report(
            videos=[],
            config={'device': 'mps', 'batch_size': 16, 'model': 'allegro'},
        )
        assert report['version'] == 1

    def test_includes_git_commit(self):
        with patch('movie_translator.metrics.report._git_short_hash', return_value='abc1234'):
            with patch('movie_translator.metrics.report._git_is_dirty', return_value=False):
                report = build_report(videos=[], config={})
        assert report['git_commit'] == 'abc1234'
        assert report['dirty'] is False

    def test_includes_dirty_flag(self):
        with patch('movie_translator.metrics.report._git_short_hash', return_value='abc1234'):
            with patch('movie_translator.metrics.report._git_is_dirty', return_value=True):
                report = build_report(videos=[], config={})
        assert report['dirty'] is True

    def test_includes_config(self):
        config = {'device': 'mps', 'batch_size': 16, 'model': 'allegro'}
        report = build_report(videos=[], config=config)
        assert report['config'] == config

    def test_includes_videos(self):
        videos = [{'path': 'ep01.mkv', 'entries': []}]
        report = build_report(videos=videos, config={})
        assert report['videos'] == videos

    def test_includes_timestamp(self):
        report = build_report(videos=[], config={})
        assert 'timestamp' in report


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        report = {
            'version': 1,
            'git_commit': 'abc1234',
            'dirty': False,
            'timestamp': '2026-03-30T14:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'path': 'ep01.mkv',
                    'hash': 'aaa',
                    'file_size_bytes': 100,
                    'duration_ms': 1000,
                    'identity': {'title': 'Test'},
                    'total_duration_ms': 500,
                    'entries': [{'name': 'identify', 'duration_ms': 10}],
                }
            ],
        }
        path = tmp_path / 'report.json'
        save_report(report, path)
        loaded = load_report(path)
        assert loaded == report

    def test_saved_file_is_readable_json(self, tmp_path):
        report = {'version': 1, 'videos': [], 'config': {}}
        path = tmp_path / 'report.json'
        save_report(report, path)
        raw = path.read_text()
        parsed = json.loads(raw)
        assert parsed['version'] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_report.py -v -x`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement report.py**

Create `movie_translator/metrics/report.py`:

```python
"""Report serialization and git metadata helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_short_hash() -> str:
    """Return the short git commit hash of HEAD."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def build_report(
    *,
    videos: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete report dict ready for serialization."""
    return {
        'version': 1,
        'git_commit': _git_short_hash(),
        'dirty': _git_is_dirty(),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': config,
        'videos': videos,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write a report dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def load_report(path: Path) -> dict[str, Any]:
    """Load a report dict from a JSON file."""
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_report.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add movie_translator/metrics/report.py movie_translator/metrics/tests/test_report.py
git commit -m "feat(metrics): add report serialization with git metadata"
```

---

### Task 5: Comparison logic and CLI

**Files:**
- Create: `movie_translator/metrics/compare.py`
- Create: `movie_translator/metrics/__main__.py`
- Create: `movie_translator/metrics/tests/test_compare.py`
- Create: `movie_translator/metrics/results/.gitkeep`

- [ ] **Step 1: Write failing tests for comparison**

Create `movie_translator/metrics/tests/test_compare.py`:

```python
from movie_translator.metrics.compare import compare_reports, match_videos


class TestMatchVideos:
    def test_match_by_hash(self):
        before = [{'hash': 'aaa', 'path': 'ep01.mkv', 'entries': []}]
        after = [{'hash': 'aaa', 'path': 'ep01.mkv', 'entries': []}]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 1
        assert excluded == 0

    def test_match_by_identity(self):
        before = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'identity': {'media_type': 'episode', 'title': 'Test', 'season': 1, 'episode': 1},
                'entries': [{'name': 'identify', 'duration_ms': 10}],
            }
        ]
        after = [
            {
                'hash': 'bbb',
                'path': 'ep01_v2.mkv',
                'identity': {'media_type': 'episode', 'title': 'Test', 'season': 1, 'episode': 1},
                'entries': [{'name': 'identify', 'duration_ms': 8}],
            }
        ]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 1

    def test_no_match(self):
        before = [{'hash': 'aaa', 'path': 'ep01.mkv', 'identity': {}, 'entries': []}]
        after = [{'hash': 'bbb', 'path': 'ep02.mkv', 'identity': {}, 'entries': []}]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 0

    def test_exclude_different_profiles(self):
        before = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'entries': [{'name': 'identify', 'duration_ms': 10}],
            }
        ]
        after = [
            {
                'hash': 'aaa',
                'path': 'ep01.mkv',
                'entries': [
                    {'name': 'identify', 'duration_ms': 8},
                    {'name': 'extract_reference.extract_pgs_track', 'duration_ms': 5000},
                ],
            }
        ]
        matched, excluded = match_videos(before, after)
        assert len(matched) == 0
        assert excluded == 1


class TestCompareReports:
    def test_basic_comparison(self):
        before = {
            'version': 1,
            'git_commit': 'aaa',
            'dirty': False,
            'timestamp': '2026-03-28T12:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 1000,
                    'entries': [
                        {'name': 'identify', 'duration_ms': 200},
                        {'name': 'translate', 'duration_ms': 800},
                    ],
                }
            ],
        }
        after = {
            'version': 1,
            'git_commit': 'bbb',
            'dirty': False,
            'timestamp': '2026-03-30T12:00:00Z',
            'config': {'device': 'mps'},
            'videos': [
                {
                    'hash': 'x',
                    'path': 'ep01.mkv',
                    'total_duration_ms': 800,
                    'entries': [
                        {'name': 'identify', 'duration_ms': 180},
                        {'name': 'translate', 'duration_ms': 620},
                    ],
                }
            ],
        }
        result = compare_reports(before, after)
        assert result['matched_videos'] == 1
        assert result['excluded_videos'] == 0
        # Check that spans are computed
        spans = {s['name']: s for s in result['spans']}
        assert 'identify' in spans
        assert spans['identify']['before_ms'] == 200
        assert spans['identify']['after_ms'] == 180
        assert spans['identify']['delta_pct'] < 0  # faster

    def test_aggregates_multiple_videos(self):
        before = {
            'version': 1, 'git_commit': 'a', 'dirty': False, 'config': {},
            'timestamp': '2026-03-28T12:00:00Z',
            'videos': [
                {'hash': 'x', 'path': 'ep01.mkv', 'total_duration_ms': 100,
                 'entries': [{'name': 'identify', 'duration_ms': 10}]},
                {'hash': 'y', 'path': 'ep02.mkv', 'total_duration_ms': 200,
                 'entries': [{'name': 'identify', 'duration_ms': 20}]},
            ],
        }
        after = {
            'version': 1, 'git_commit': 'b', 'dirty': False, 'config': {},
            'timestamp': '2026-03-30T12:00:00Z',
            'videos': [
                {'hash': 'x', 'path': 'ep01.mkv', 'total_duration_ms': 80,
                 'entries': [{'name': 'identify', 'duration_ms': 8}]},
                {'hash': 'y', 'path': 'ep02.mkv', 'total_duration_ms': 160,
                 'entries': [{'name': 'identify', 'duration_ms': 16}]},
            ],
        }
        result = compare_reports(before, after)
        assert result['matched_videos'] == 2
        spans = {s['name']: s for s in result['spans']}
        # Average: (10+20)/2=15 before, (8+16)/2=12 after
        assert spans['identify']['before_ms'] == 15
        assert spans['identify']['after_ms'] == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_compare.py -v -x`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement compare.py**

Create `movie_translator/metrics/compare.py`:

```python
"""Comparison logic for metrics reports."""

from __future__ import annotations

from typing import Any


def _video_key(video: dict[str, Any]) -> str:
    """Build a matching key for a video entry."""
    return video.get('hash', '')


def _identity_key(video: dict[str, Any]) -> str | None:
    """Build a matching key from identity fields."""
    identity = video.get('identity')
    if not identity:
        return None
    parts = [
        str(identity.get('media_type', '')),
        str(identity.get('title', '')),
        str(identity.get('season', '')),
        str(identity.get('episode', '')),
    ]
    key = '|'.join(parts)
    return key if any(parts) else None


def _top_level_spans(entries: list[dict[str, Any]]) -> set[str]:
    """Extract the set of top-level span names (first segment before dot)."""
    names = set()
    for entry in entries:
        name = entry['name']
        # Include full names that contain distinguishing sub-operations
        names.add(name)
    return names


def _profiles_match(before_entries: list[dict], after_entries: list[dict]) -> bool:
    """Check if two videos exercised the same pipeline profile.

    Two profiles match if they have the same set of top-level stage span names
    (names without dots). Sub-operation differences within a stage are fine.
    """
    before_stages = {e['name'].split('.')[0] for e in before_entries}
    after_stages = {e['name'].split('.')[0] for e in after_entries}
    return before_stages == after_stages


def match_videos(
    before_videos: list[dict[str, Any]],
    after_videos: list[dict[str, Any]],
) -> tuple[list[tuple[dict, dict]], int]:
    """Match videos across two reports.

    Returns (matched_pairs, excluded_count).
    Matches by hash first, then by identity. Excludes pairs with different
    pipeline profiles.
    """
    matched: list[tuple[dict, dict]] = []
    excluded = 0
    used_after: set[int] = set()

    # Pass 1: match by hash
    after_by_hash = {}
    for i, v in enumerate(after_videos):
        h = _video_key(v)
        if h:
            after_by_hash[h] = (i, v)

    for bv in before_videos:
        bh = _video_key(bv)
        if bh and bh in after_by_hash:
            ai, av = after_by_hash[bh]
            if ai not in used_after:
                if _profiles_match(bv.get('entries', []), av.get('entries', [])):
                    matched.append((bv, av))
                else:
                    excluded += 1
                used_after.add(ai)

    # Pass 2: match by identity (for videos not yet matched)
    matched_before_hashes = {_video_key(bv) for bv, _ in matched}
    after_by_identity: dict[str, tuple[int, dict]] = {}
    for i, v in enumerate(after_videos):
        if i in used_after:
            continue
        ik = _identity_key(v)
        if ik:
            after_by_identity[ik] = (i, v)

    for bv in before_videos:
        bh = _video_key(bv)
        if bh in matched_before_hashes:
            continue
        bik = _identity_key(bv)
        if bik and bik in after_by_identity:
            ai, av = after_by_identity[bik]
            if ai not in used_after:
                if _profiles_match(bv.get('entries', []), av.get('entries', [])):
                    matched.append((bv, av))
                else:
                    excluded += 1
                used_after.add(ai)

    return matched, excluded


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare two reports and return structured results.

    Returns a dict with:
    - matched_videos: number of matched video pairs
    - excluded_videos: number of pairs excluded for different profiles
    - total_before_videos: total videos in before report
    - total_after_videos: total videos in after report
    - spans: list of {name, before_ms, after_ms, delta_ms, delta_pct}
    - total_before_ms: average total wall-clock before
    - total_after_ms: average total wall-clock after
    """
    matched, excluded = match_videos(
        before.get('videos', []),
        after.get('videos', []),
    )

    # Aggregate spans across matched pairs
    span_before: dict[str, list[float]] = {}
    span_after: dict[str, list[float]] = {}
    total_before_list: list[float] = []
    total_after_list: list[float] = []

    for bv, av in matched:
        total_before_list.append(bv.get('total_duration_ms', 0))
        total_after_list.append(av.get('total_duration_ms', 0))

        for entry in bv.get('entries', []):
            span_before.setdefault(entry['name'], []).append(entry['duration_ms'])
        for entry in av.get('entries', []):
            span_after.setdefault(entry['name'], []).append(entry['duration_ms'])

    # Build span comparison list (only spans present in both)
    all_names = sorted(span_before.keys() & span_after.keys())
    spans = []
    for name in all_names:
        avg_before = sum(span_before[name]) / len(span_before[name])
        avg_after = sum(span_after[name]) / len(span_after[name])
        delta_ms = avg_after - avg_before
        delta_pct = (delta_ms / avg_before * 100) if avg_before != 0 else 0
        spans.append({
            'name': name,
            'before_ms': round(avg_before, 1),
            'after_ms': round(avg_after, 1),
            'delta_ms': round(delta_ms, 1),
            'delta_pct': round(delta_pct, 1),
        })

    total_before = (
        round(sum(total_before_list) / len(total_before_list), 1) if total_before_list else 0
    )
    total_after = (
        round(sum(total_after_list) / len(total_after_list), 1) if total_after_list else 0
    )

    return {
        'matched_videos': len(matched),
        'excluded_videos': excluded,
        'total_before_videos': len(before.get('videos', [])),
        'total_after_videos': len(after.get('videos', [])),
        'spans': spans,
        'total_before_ms': total_before,
        'total_after_ms': total_after,
    }


def format_comparison(
    before: dict[str, Any],
    after: dict[str, Any],
    result: dict[str, Any],
) -> str:
    """Format comparison results as a human-readable table."""
    lines: list[str] = []

    # Header
    bc = before.get('git_commit', '?')
    ac = after.get('git_commit', '?')
    bd = before.get('timestamp', '?')[:10]
    ad = after.get('timestamp', '?')[:10]
    lines.append(f'Comparing: {bc} ({bd}) -> {ac} ({ad})')

    config = after.get('config', {})
    config_parts = [f'{k}={v}' for k, v in config.items()]
    if config_parts:
        lines.append(f'Config: {", ".join(config_parts)}')

    bd_dirty = 'yes' if before.get('dirty') else 'no'
    ad_dirty = 'yes' if after.get('dirty') else 'no'
    lines.append(f'Dirty: before={bd_dirty}, after={ad_dirty}')

    if before.get('dirty') or after.get('dirty'):
        lines.append('WARNING: One or both reports were generated from a dirty working tree.')

    matched = result['matched_videos']
    excluded = result['excluded_videos']
    total_b = result['total_before_videos']
    total_a = result['total_after_videos']
    lines.append('')
    lines.append(
        f'Videos matched: {matched}/{max(total_b, total_a)}'
        f' ({excluded} excluded for different profiles)'
    )
    lines.append('')

    if not result['spans']:
        lines.append('No comparable spans found.')
        return '\n'.join(lines)

    # Table
    name_width = max(len(s['name']) for s in result['spans'])
    name_width = max(name_width, len('Stage'), len('Total (wall clock)'))
    header = f"{'Stage':<{name_width}}   {'Before':>10}   {'After':>10}   {'Delta':>8}"
    sep = '-' * len(header)

    lines.append(header)
    lines.append(sep)

    for span in result['spans']:
        before_str = f"{span['before_ms']:.0f}ms"
        after_str = f"{span['after_ms']:.0f}ms"
        pct = span['delta_pct']
        delta_str = f'{pct:+.0f}%' if pct != 0 else '0%'
        lines.append(
            f"{span['name']:<{name_width}}   {before_str:>10}   {after_str:>10}   {delta_str:>8}"
        )

    lines.append(sep)

    tb = result['total_before_ms']
    ta = result['total_after_ms']
    total_delta = ((ta - tb) / tb * 100) if tb != 0 else 0
    total_delta_str = f'{total_delta:+.0f}%' if total_delta != 0 else '0%'
    lines.append(
        f"{'Total (wall clock)':<{name_width}}   {f'{tb:.0f}ms':>10}   "
        f"{f'{ta:.0f}ms':>10}   {total_delta_str:>8}"
    )

    # Summary
    lines.append('')
    if result['spans']:
        biggest = min(result['spans'], key=lambda s: s['delta_pct'])
        if biggest['delta_pct'] < 0:
            lines.append(
                f'Summary: {abs(total_delta):.0f}% {"faster" if total_delta < 0 else "slower"} '
                f'overall. Biggest win: {biggest["name"]} '
                f'({biggest["delta_pct"]:+.0f}%, {biggest["delta_ms"]:+.0f}ms).'
            )
        else:
            lines.append(
                f'Summary: {abs(total_delta):.0f}% {"faster" if total_delta < 0 else "slower"} '
                f'overall. No individual span improved.'
            )
    lines.append(f'{matched} videos compared. {excluded} excluded.')

    return '\n'.join(lines)
```

- [ ] **Step 4: Implement __main__.py CLI entry point**

Create `movie_translator/metrics/__main__.py`:

```python
"""CLI entry point: python -m movie_translator.metrics compare <before> <after>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare_reports, format_comparison
from .report import load_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m movie_translator.metrics',
        description='Metrics comparison tool for movie-translator',
    )
    subparsers = parser.add_subparsers(dest='command')

    compare_parser = subparsers.add_parser('compare', help='Compare two metric reports')
    compare_parser.add_argument('before', type=Path, help='Path to the before report JSON')
    compare_parser.add_argument('after', type=Path, help='Path to the after report JSON')

    args = parser.parse_args()

    if args.command == 'compare':
        before = load_report(args.before)
        after = load_report(args.after)
        result = compare_reports(before, after)
        print(format_comparison(before, after, result))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
```

Create `movie_translator/metrics/results/.gitkeep` (empty file).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_compare.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/metrics/compare.py movie_translator/metrics/__main__.py movie_translator/metrics/tests/test_compare.py movie_translator/metrics/results/.gitkeep
git commit -m "feat(metrics): add comparison logic and CLI"
```

---

### Task 6: Wire metrics into PipelineContext and pipeline orchestrator

**Files:**
- Modify: `movie_translator/context.py`
- Modify: `movie_translator/pipeline.py`
- Modify: `movie_translator/main.py`

- [ ] **Step 1: Add metrics field to PipelineContext**

In `movie_translator/context.py`, add the import and field:

Add to imports at top:

```python
from movie_translator.metrics.collector import MetricsCollector, NullCollector
```

Add field to `PipelineContext` class (after `burned_in_probed`):

```python
    metrics: MetricsCollector | NullCollector = field(default_factory=NullCollector)
```

- [ ] **Step 2: Wrap stage.run() with metrics spans in pipeline.py**

In `movie_translator/pipeline.py`, modify the `process_video_file` method.

Replace the stage loop in `process_video_file`:

```python
        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                ctx = stage.run(ctx)
            return True
```

With:

```python
        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                with ctx.metrics.span(stage.name):
                    ctx = stage.run(ctx)
            return True
```

- [ ] **Step 3: Add --metrics flag and wiring in main.py**

In `movie_translator/main.py`, add the CLI argument in `parse_args()`:

```python
    parser.add_argument('--metrics', action='store_true', help='Collect performance metrics')
```

Add imports at top of file:

```python
from .metrics.collector import MetricsCollector, NullCollector
from .metrics.listeners import ReportBuilder
from .metrics.report import build_report, save_report
```

In `main()`, after `logging.getLogger('transformers').setLevel(logging.ERROR)`, add collector setup:

```python
    if args.metrics:
        collector = MetricsCollector()
        report_builder = ReportBuilder()
        collector.add_listener(report_builder.on_event)
    else:
        collector = NullCollector()
        report_builder = None
```

Modify the `TranslationPipeline` construction to pass the collector:

```python
        pipeline = TranslationPipeline(
            device=args.device,
            batch_size=args.batch_size,
            model=args.model,
            enable_fetch=not args.no_fetch,
            enable_inpaint=args.inpaint,
            tracker=tracker,
            metrics=collector,
        )
```

In `TranslationPipeline.__init__`, accept and store the metrics parameter. In `movie_translator/pipeline.py`, add to `__init__` signature:

```python
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        enable_inpaint: bool = False,
        tracker=None,
        metrics=None,
    ):
```

And store it: `self.metrics = metrics` (add after `self.tracker = tracker`).

Pass it into context creation in `process_video_file`:

```python
        ctx = PipelineContext(
            video_path=video_path, work_dir=work_dir, config=self.config,
            metrics=self.metrics or NullCollector(),
        )
```

Add import at top of `pipeline.py`:

```python
from .metrics.collector import NullCollector
```

In `main.py`, inside the video loop, add report_builder start/end calls. Wrap the per-video processing:

After `work_dir = create_work_dir(video_path, root_dir)`, add:

```python
            if report_builder is not None:
                identity_dict = {}
                report_builder.start_video(
                    path=relative_name,
                    hash='',
                    file_size_bytes=video_path.stat().st_size,
                    duration_ms=0,
                    identity=identity_dict,
                )
```

After the try/except block that processes the video (before the cleanup block), add:

```python
            if report_builder is not None:
                report_builder.end_video()
```

At the very end of `main()`, after the `with ProgressTracker(...)` block, add the report save:

```python
    if report_builder is not None:
        config_dict = {
            'device': args.device,
            'batch_size': args.batch_size,
            'model': args.model,
            'enable_fetch': not args.no_fetch,
            'enable_inpaint': args.inpaint,
        }
        report = build_report(videos=report_builder.videos, config=config_dict)
        metrics_path = root_dir / '.translate_temp' / 'metrics.json'
        save_report(report, metrics_path)
        console.print(f'[dim]Metrics saved to {metrics_path}[/dim]')
```

- [ ] **Step 4: Run existing tests to verify nothing is broken**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/ movie_translator/tests/ -v --timeout=30`
Expected: All existing tests PASS (the new `metrics` field has a default so no existing code breaks)

- [ ] **Step 5: Commit**

```bash
git add movie_translator/context.py movie_translator/pipeline.py movie_translator/main.py
git commit -m "feat(metrics): wire collector into pipeline and CLI"
```

---

### Task 7: Instrument identify stage

**Files:**
- Modify: `movie_translator/stages/identify.py`

- [ ] **Step 1: Read current identify stage and the identify_media function**

Read `movie_translator/stages/identify.py` and `movie_translator/identifier/identify.py` for current code.

- [ ] **Step 2: Add sub-operation spans**

Modify `movie_translator/stages/identify.py`. The stage itself is already wrapped by the pipeline orchestrator (`identify` span). Add spans for sub-operations by modifying `identify_media` calls.

Since `identify_media` is a single function that internally calls `parse_filename`, `extract_container_metadata`, `compute_oshash`, and `lookup_tmdb`, and we don't want to modify the identifier module heavily, instrument at the stage level by splitting the call:

Replace the `run` method in `IdentifyStage`:

```python
class IdentifyStage:
    name = 'identify'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Identifying: {ctx.video_path.name}')
        ctx.identity = identify_media(ctx.video_path)

        # Record identity metadata for metrics
        if ctx.identity is not None:
            identity = ctx.identity
            with ctx.metrics.span('record_identity') as s:
                s.detail('title', identity.title)
                s.detail('media_type', identity.media_type)
                s.detail('is_anime', identity.is_anime)
                if identity.season is not None:
                    s.detail('season', identity.season)
                if identity.episode is not None:
                    s.detail('episode', identity.episode)
                if identity.year is not None:
                    s.detail('year', identity.year)

        return ctx
```

Note: `identify_media` is a thin function that calls four sub-functions sequentially. For deeper instrumentation (timing `parse_filename`, `compute_oshash`, etc. individually), we would modify `identifier/identify.py` to accept a metrics collector. For now, timing the whole `identify` stage is sufficient — the sub-operations are all fast (< 200ms total). We can add granular timing inside `identify_media` in a follow-up if profiling shows it's useful.

- [ ] **Step 3: Run stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/test_identify.py -v`
Expected: PASS (existing tests still work because `ctx.metrics` defaults to NullCollector)

- [ ] **Step 4: Commit**

```bash
git add movie_translator/stages/identify.py
git commit -m "feat(metrics): instrument identify stage"
```

---

### Task 8: Instrument extract_reference stage

**Files:**
- Modify: `movie_translator/stages/extract_ref.py`

- [ ] **Step 1: Add sub-operation spans**

Modify `movie_translator/stages/extract_ref.py`. Wrap key operations:

```python
class ExtractReferenceStage:
    name = 'extract_reference'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        extractor = SubtitleExtractor()
        ref_dir = ctx.work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        with ctx.metrics.span('get_track_info'):
            track_info = extractor.get_track_info(ctx.video_path)
            eng_track = extractor.find_english_track(track_info) if track_info else None

        if eng_track:
            ctx.original_english_track = OriginalTrack(
                stream_index=eng_track['id'],
                subtitle_index=eng_track.get('subtitle_index', 0),
                codec=eng_track.get('codec', 'unknown'),
                language=eng_track.get('properties', {}).get('language', 'eng'),
            )

            if _is_image_codec(eng_track):
                with ctx.metrics.span('extract_pgs_track') as s:
                    srt_path = extract_pgs_track(ctx.video_path, eng_track['id'], ref_dir)
                    if srt_path:
                        ctx.reference_path = srt_path
                        logger.info(f'Extracted PGS reference via OCR: {srt_path.name}')
                        s.detail('subtitle_count', _count_srt_entries(srt_path))
            else:
                with ctx.metrics.span('extract_subtitle'):
                    subtitle_ext = extractor.get_subtitle_extension(eng_track)
                    ref_path = ref_dir / f'{ctx.video_path.stem}_reference{subtitle_ext}'
                    try:
                        extractor.extract_subtitle(
                            ctx.video_path,
                            eng_track['id'],
                            ref_path,
                            eng_track.get('subtitle_index', 0),
                        )
                        ctx.reference_path = ref_path
                        logger.info(f'Extracted reference: {ref_path.name}')
                    except Exception as e:
                        logger.warning(f'Failed to extract reference: {e}')

        if ctx.reference_path is None and is_vision_ocr_available():
            ctx.burned_in_probed = True
            with ctx.metrics.span('probe_burned_in') as s:
                detected = probe_for_burned_in_subtitles(ctx.video_path)
                s.detail('detected', detected)
            if detected:
                with ctx.metrics.span('extract_burned_in'):
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

Add a helper at the module level to count SRT entries:

```python
def _count_srt_entries(srt_path):
    """Count entries in an SRT file for metrics."""
    try:
        return srt_path.read_text().count('\n\n')
    except Exception:
        return 0
```

- [ ] **Step 2: Run stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/test_extract_ref.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add movie_translator/stages/extract_ref.py
git commit -m "feat(metrics): instrument extract_reference stage"
```

---

### Task 9: Instrument fetch stage

**Files:**
- Modify: `movie_translator/stages/fetch.py`

- [ ] **Step 1: Add sub-operation spans**

Modify `movie_translator/stages/fetch.py`. The `run` method becomes:

```python
    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.config.enable_fetch:
            return ctx

        fetcher = self._build_fetcher(ctx.video_path)
        if fetcher is None:
            return ctx

        with ctx.metrics.span('search_all') as s:
            try:
                all_matches = fetcher.search_all(ctx.identity, ['eng', 'pol'])
            except Exception as e:
                logger.warning(f'Subtitle search failed: {e}')
                ctx.fetched_subtitles = {}
                return ctx
            s.detail('candidates', len(all_matches) if all_matches else 0)

        if not all_matches:
            logger.info('No subtitles found from any provider')
            ctx.fetched_subtitles = {}
            return ctx

        logger.info(f'Found {len(all_matches)} subtitle candidate(s)')

        candidates_dir = ctx.work_dir / 'candidates'
        candidates_dir.mkdir(parents=True, exist_ok=True)

        with ctx.metrics.span('download_all') as s:
            downloaded = self._download_all(fetcher, all_matches, candidates_dir)
            s.detail('downloaded', len(downloaded))
            s.detail('failed', len(all_matches) - len(downloaded))

        if not downloaded:
            logger.warning('All candidate downloads failed')
            ctx.fetched_subtitles = {}
            return ctx

        with ctx.metrics.span('validate_and_select') as s:
            ctx.fetched_subtitles = self._validate_and_select(
                downloaded,
                ctx.reference_path,
            )
            total_selected = sum(len(v) for v in ctx.fetched_subtitles.values())
            s.detail('passed', total_selected)
            s.detail('rejected', len(downloaded) - total_selected)

        if ctx.reference_path and 'pol' in ctx.fetched_subtitles:
            for sub in ctx.fetched_subtitles['pol']:
                with ctx.metrics.span('align') as s:
                    self._align_subtitle(sub.path, ctx.reference_path)
                    s.detail('method', 'ilass' if align_ilass.is_available() else 'builtin')

        return ctx
```

- [ ] **Step 2: Run stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/test_fetch.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add movie_translator/stages/fetch.py
git commit -m "feat(metrics): instrument fetch stage"
```

---

### Task 10: Instrument extract_english stage

**Files:**
- Modify: `movie_translator/stages/extract_english.py`

- [ ] **Step 1: Add sub-operation spans**

Modify `movie_translator/stages/extract_english.py`:

```python
class ExtractEnglishStage:
    name = 'extract'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        fetched_eng = None
        if ctx.fetched_subtitles:
            eng_subs = ctx.fetched_subtitles.get('eng')
            if eng_subs:
                fetched_eng = eng_subs[0].path

        with ctx.metrics.span('select_source') as s:
            if fetched_eng:
                ctx.english_source = fetched_eng
                s.detail('source', 'fetched')
            elif ctx.reference_path:
                ctx.english_source = ctx.reference_path
                s.detail('source', 'reference')
            else:
                ctx.english_source = self._extract_from_video(ctx)
                s.detail('source', 'embedded' if ctx.english_source else 'ocr')

        if ctx.english_source is None:
            raise RuntimeError(f'No English subtitle source found for {ctx.video_path.name}')

        with ctx.metrics.span('extract_dialogue_lines') as s:
            ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)
            if not ctx.dialogue_lines:
                raise RuntimeError(f'No dialogue lines found in {ctx.english_source.name}')
            s.detail('lines', len(ctx.dialogue_lines))
            s.detail('chars', sum(len(line.text) for line in ctx.dialogue_lines))

        logger.info(f'English source: {ctx.english_source.name} ({len(ctx.dialogue_lines)} lines)')
        return ctx
```

The `_extract_from_video` method stays unchanged — it's a fallback path that already has its operations (PGS extraction, OCR) which could be instrumented in a follow-up.

- [ ] **Step 2: Run stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/test_extract_english.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add movie_translator/stages/extract_english.py
git commit -m "feat(metrics): instrument extract_english stage"
```

---

### Task 11: Instrument translate stage

**Files:**
- Modify: `movie_translator/stages/translate.py`

- [ ] **Step 1: Add sub-operation spans**

Modify `movie_translator/stages/translate.py`. The translate stage runs font checking and translation in parallel via ThreadPoolExecutor. Thanks to ContextVar propagation, both threads inherit the `translate` prefix:

```python
class TranslateStage:
    name = 'translate'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.dialogue_lines is not None
        assert ctx.english_source is not None

        logger.info(f'Translating {len(ctx.dialogue_lines)} lines...')

        english_source = ctx.english_source
        metrics = ctx.metrics

        def _check_fonts():
            with metrics.span('check_fonts') as s:
                supports = check_embedded_fonts_support_polish(ctx.video_path, english_source)
                if supports:
                    s.detail('supports_polish', True)
                    return FontInfo(supports_polish=True)
                is_mkv = ctx.video_path.suffix.lower() == '.mkv'
                if is_mkv:
                    names = get_ass_font_names(english_source)
                    result = find_system_font_for_polish(names)
                    if result:
                        fp, fam = result
                        fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                        s.detail('supports_polish', False)
                        s.detail('fallback_font', fam)
                        return FontInfo(
                            supports_polish=False,
                            font_attachments=[fp],
                            fallback_font_family=fallback,
                        )
                s.detail('supports_polish', False)
                s.detail('fallback_font', None)
                return FontInfo(supports_polish=False)

        def _translate():
            with metrics.span('batch') as s:
                input_texts = [line.text for line in ctx.dialogue_lines]
                s.detail('input_lines', len(input_texts))
                s.detail('input_chars', sum(len(t) for t in input_texts))
                s.detail('batch_size', ctx.config.batch_size)

                translated = translate_dialogue_lines(
                    ctx.dialogue_lines,
                    ctx.config.device,
                    ctx.config.batch_size,
                    ctx.config.model,
                )

                s.detail('output_lines', len(translated))
                s.detail('output_chars', sum(len(line.text) for line in translated))
                s.detail('batches', (len(input_texts) + ctx.config.batch_size - 1) // ctx.config.batch_size)
                return translated

        with ThreadPoolExecutor(max_workers=2) as pool:
            font_future = pool.submit(_check_fonts)
            translate_future = pool.submit(_translate)

            ctx.font_info = font_future.result()
            translated = translate_future.result()

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
```

- [ ] **Step 2: Run stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/test_translate.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add movie_translator/stages/translate.py
git commit -m "feat(metrics): instrument translate stage"
```

---

### Task 12: Instrument create_tracks and mux stages

**Files:**
- Modify: `movie_translator/stages/create_tracks.py`
- Modify: `movie_translator/stages/mux.py`

- [ ] **Step 1: Instrument create_tracks**

Modify `movie_translator/stages/create_tracks.py`:

```python
class CreateTracksStage:
    name = 'create_tracks'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        is_mkv = ctx.video_path.suffix.lower() == '.mkv'
        replace_chars = False

        assert ctx.font_info is not None
        assert ctx.english_source is not None
        assert ctx.translated_lines is not None

        if not ctx.font_info.supports_polish:
            if is_mkv and ctx.font_info.font_attachments:
                logger.info(f'Will embed font "{ctx.font_info.font_attachments[0].name}"')
            elif is_mkv:
                logger.warning('No system font with Polish support, replacing characters')
                replace_chars = True
            else:
                replace_chars = True

        with ctx.metrics.span('create_polish_subtitles'):
            ai_polish_ass = ctx.work_dir / f'{ctx.video_path.stem}_polish_ai.ass'
            SubtitleProcessor.create_polish_subtitles(
                ctx.english_source,
                ctx.translated_lines,
                ai_polish_ass,
                replace_chars,
            )

        if ctx.font_info.fallback_font_family:
            with ctx.metrics.span('override_font'):
                SubtitleProcessor.override_font_name(ai_polish_ass, ctx.font_info.fallback_font_family)

        with ctx.metrics.span('build_track_list') as s:
            fetched_pol_list = ctx.fetched_subtitles.get('pol', []) if ctx.fetched_subtitles else []
            tracks: list[SubtitleFile] = []

            for i, fetched_pol in enumerate(fetched_pol_list):
                pol_title = f'Polish ({fetched_pol.source})'
                tracks.append(SubtitleFile(fetched_pol.path, 'pol', pol_title, is_default=(i == 0)))
                if ctx.font_info.fallback_font_family:
                    SubtitleProcessor.override_font_name(
                        fetched_pol.path,
                        ctx.font_info.fallback_font_family,
                    )

            tracks.append(
                SubtitleFile(
                    ai_polish_ass,
                    'pol',
                    'Polish (AI)',
                    is_default=not bool(fetched_pol_list),
                )
            )

            ctx.subtitle_tracks = tracks
            s.detail('tracks', len(tracks))

        return ctx
```

- [ ] **Step 2: Instrument mux**

Modify `movie_translator/stages/mux.py`:

```python
class MuxStage:
    name = 'mux'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        source_video = ctx.video_path
        if ctx.ocr_results and ctx.config.enable_inpaint and ctx.inpainted_video is None:
            logger.info('Removing burned-in subtitles via inpainting...')
            with ctx.metrics.span('inpaint') as s:
                inpainted = ctx.work_dir / f'{ctx.video_path.stem}_inpainted{ctx.video_path.suffix}'
                remove_burned_in_subtitles(
                    ctx.video_path,
                    inpainted,
                    ctx.ocr_results,
                    ctx.config.device,
                )
                ctx.inpainted_video = inpainted
                source_video = inpainted
                s.detail('frames', len(ctx.ocr_results))
        elif ctx.inpainted_video:
            source_video = ctx.inpainted_video

        original_sub_index = None
        original_sub_title = None
        if ctx.original_english_track:
            original_sub_index = ctx.original_english_track.subtitle_index
            original_sub_title = 'English (Original)'

        assert ctx.subtitle_tracks is not None
        assert ctx.font_info is not None

        with ctx.metrics.span('create_clean_video') as s:
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
            s.detail('tracks', len(ctx.subtitle_tracks))
            s.detail('font_attachments', len(ctx.font_info.font_attachments) if ctx.font_info.font_attachments else 0)

        with ctx.metrics.span('verify_result'):
            expected_tracks = list(ctx.subtitle_tracks)
            if original_sub_index is not None:
                lang = ctx.original_english_track.language if ctx.original_english_track else 'eng'
                expected_tracks.insert(
                    0,
                    SubtitleFile(
                        path=Path(),
                        language=lang,
                        title=original_sub_title or 'English (Original)',
                        is_default=False,
                    ),
                )
            ops.verify_result(temp_video, expected_tracks=expected_tracks)

        if not ctx.config.dry_run:
            with ctx.metrics.span('replace_original'):
                self._replace_original(ctx.video_path, temp_video)

        return ctx
```

- [ ] **Step 3: Run all stage tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/stages/tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add movie_translator/stages/create_tracks.py movie_translator/stages/mux.py
git commit -m "feat(metrics): instrument create_tracks and mux stages"
```

---

### Task 13: Update ReportBuilder with video identity from pipeline context

**Files:**
- Modify: `movie_translator/main.py`

- [ ] **Step 1: Enrich report_builder.start_video with identity after identify stage**

The challenge: when `start_video` is called in `main.py`, we don't yet have the identity (it's set by the identify stage). We need to update the video's identity and hash after the pipeline runs.

Add a method to `ReportBuilder` in `movie_translator/metrics/listeners.py`:

```python
    def update_current_video(self, **fields: Any) -> None:
        """Update fields on the current video being collected."""
        if self._current_video is not None:
            self._current_video.update(fields)
```

In `movie_translator/pipeline.py`, after the stage loop succeeds, update the report builder with identity data. Modify `process_video_file`:

```python
    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self.config.dry_run = dry_run
        ctx = PipelineContext(
            video_path=video_path, work_dir=work_dir, config=self.config,
            metrics=self.metrics or NullCollector(),
        )

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                with ctx.metrics.span(stage.name):
                    ctx = stage.run(ctx)
            return True
        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False
```

Add a `get_identity_dict` helper to `pipeline.py`:

```python
def _identity_to_dict(identity) -> dict:
    """Convert a MediaIdentity NamedTuple to a dict for the report."""
    if identity is None:
        return {}
    return {
        'title': identity.title,
        'parsed_title': identity.parsed_title,
        'media_type': identity.media_type,
        'season': identity.season,
        'episode': identity.episode,
        'year': identity.year,
        'is_anime': identity.is_anime,
        'release_group': identity.release_group,
        'imdb_id': identity.imdb_id,
        'tmdb_id': identity.tmdb_id,
    }
```

In `main.py`, after `pipeline.process_video_file` returns, update the report builder with identity data. In the video loop, after the `pipeline.process_video_file` call:

```python
            if report_builder is not None and hasattr(pipeline, '_last_ctx'):
                # Update identity from pipeline context
                pass
```

Actually, a cleaner approach: have `process_video_file` return the context (or at least the identity). Modify `process_video_file` to store identity:

In `pipeline.py`, after the stage loop:

```python
            self._last_identity = ctx.identity
            self._last_hash = ctx.identity.oshash if ctx.identity else ''
            self._last_duration_ms = 0  # We don't have video duration in context yet
```

Then in `main.py`, after the pipeline call:

```python
                if report_builder is not None:
                    from .pipeline import _identity_to_dict
                    report_builder.update_current_video(
                        identity=_identity_to_dict(pipeline._last_identity),
                        hash=pipeline._last_hash,
                    )
```

Wait — this is getting complex. Simpler approach: have `process_video_file` return the identity info, and update the report builder in main.py.

Modify `process_video_file` signature to return a richer result:

```python
    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> tuple[bool, dict]:
        """Returns (success, identity_dict)."""
        self.config.dry_run = dry_run
        ctx = PipelineContext(
            video_path=video_path, work_dir=work_dir, config=self.config,
            metrics=self.metrics or NullCollector(),
        )

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                with ctx.metrics.span(stage.name):
                    ctx = stage.run(ctx)
            identity_dict = _identity_to_dict(ctx.identity) if ctx.identity else {}
            return True, identity_dict
        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            identity_dict = _identity_to_dict(ctx.identity) if ctx.identity else {}
            return False, identity_dict
```

Update callers in `main.py` — change:

```python
                elif pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run):
```

To:

```python
                else:
                    success, identity_dict = pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run)
                    if report_builder is not None:
                        report_builder.update_current_video(
                            identity=identity_dict,
                            hash=identity_dict.get('oshash', ''),
                        )
                    if success:
```

Actually this is getting tangled with the existing control flow. Let me simplify:

In `main.py`, instead of modifying control flow, use `update_current_video` from within `pipeline.process_video_file` by having the pipeline store a reference to the report_builder. But that couples things.

Simplest: have the pipeline emit identity data via the metrics system itself. Add a special span or just update after the fact. Let me go with the approach of returning identity from the pipeline and keeping main.py changes minimal:

```python
    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self.config.dry_run = dry_run
        ctx = PipelineContext(
            video_path=video_path, work_dir=work_dir, config=self.config,
            metrics=self.metrics or NullCollector(),
        )

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                with ctx.metrics.span(stage.name):
                    ctx = stage.run(ctx)
            self.last_identity = ctx.identity
            return True
        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            self.last_identity = ctx.identity
            return False
```

Then in `main.py`, after each video processes:

```python
            if report_builder is not None:
                identity = pipeline.last_identity
                identity_dict = identity._asdict() if identity else {}
                # Remove fields not needed in report
                for key in ('oshash', 'file_size', 'raw_filename'):
                    identity_dict.pop(key, None)
                report_builder.update_current_video(
                    identity=identity_dict,
                    hash=identity.oshash if identity else '',
                )
```

- [ ] **Step 2: Test identity enrichment**

Add test to `movie_translator/metrics/tests/test_listeners.py`:

```python
    def test_update_current_video(self):
        rb = ReportBuilder()
        rb.start_video(path='ep.mkv', hash='', file_size_bytes=1, duration_ms=1, identity={})
        rb.update_current_video(identity={'title': 'Updated'}, hash='newhash')
        rb.end_video()

        assert rb.videos[0]['identity'] == {'title': 'Updated'}
        assert rb.videos[0]['hash'] == 'newhash'
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/metrics/tests/test_listeners.py -v`
Expected: PASS

- [ ] **Step 4: Implement changes in main.py and pipeline.py**

Apply the changes described in step 1 to `movie_translator/pipeline.py` and `movie_translator/main.py`.

- [ ] **Step 5: Run all tests**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest movie_translator/ tests/ -v --timeout=30`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/metrics/listeners.py movie_translator/metrics/tests/test_listeners.py movie_translator/pipeline.py movie_translator/main.py
git commit -m "feat(metrics): enrich video entries with identity metadata"
```

---

### Task 14: Update __init__.py exports and run full test suite

**Files:**
- Modify: `movie_translator/metrics/__init__.py`

- [ ] **Step 1: Finalize exports**

Update `movie_translator/metrics/__init__.py`:

```python
"""Pipeline metrics and observability."""

from .collector import MetricsCollector, NullCollector, Span
from .events import SpanEvent
from .listeners import ReportBuilder
from .report import build_report, load_report, save_report

__all__ = [
    'MetricsCollector',
    'NullCollector',
    'ReportBuilder',
    'Span',
    'SpanEvent',
    'build_report',
    'load_report',
    'save_report',
]
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/w/h_dev/movie_translator && uv run pytest -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Run linter**

Run: `cd /Users/w/h_dev/movie_translator && uv run ruff check movie_translator/metrics/ && uv run ruff format --check movie_translator/metrics/`
Expected: No issues

- [ ] **Step 4: Fix any lint issues**

Run: `cd /Users/w/h_dev/movie_translator && uv run ruff check --fix movie_translator/metrics/ && uv run ruff format movie_translator/metrics/`

- [ ] **Step 5: Commit**

```bash
git add movie_translator/metrics/__init__.py
git commit -m "feat(metrics): finalize module exports"
```
