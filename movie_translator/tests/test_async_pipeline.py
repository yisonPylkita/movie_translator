"""Tests for the async pipeline orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from movie_translator.async_pipeline import _make_file_tag, process_file, run_all
from movie_translator.context import FontInfo, PipelineConfig, PipelineContext
from movie_translator.gpu_queue import GpuQueue, GpuTask
from movie_translator.metrics.collector import NullCollector
from movie_translator.types import DialogueLine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StubGpuTask(GpuTask):
    model_type: str = field(init=False, default='stub')
    value: Any = None

    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> Any:
        return self.value


class FakeStage:
    """A stage that records calls and applies side effects to ctx."""

    def __init__(self, name: str, side_effect=None):
        self.name = name
        self._side_effect = side_effect
        self.call_count = 0

    def run(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        if self._side_effect:
            self._side_effect(ctx)
        return ctx


class FakeTranslateStage:
    """Fake translate stage with check_fonts and set_tracker."""

    name = 'translate'

    def __init__(self):
        self.call_count = 0
        self._tracker = None

    def set_tracker(self, tracker):
        self._tracker = tracker

    def check_fonts(self, ctx: PipelineContext) -> FontInfo:
        return FontInfo(supports_polish=True)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        return ctx


def _make_test_config(**overrides) -> PipelineConfig:
    defaults: dict = {
        'device': 'cpu',
        'batch_size': 4,
        'model': 'allegro',
        'enable_fetch': False,
        'enable_inpaint': False,
        'dry_run': True,
        'workers': 2,
        'external_subs_dir': None,
        'model_cache': None,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_fake_tracker():
    """Create a mock tracker with all required methods."""
    tracker = MagicMock()
    tracker.set_stage = MagicMock()
    tracker.set_gpu_status = MagicMock()
    tracker.set_stage_progress = MagicMock()
    tracker.start_file = MagicMock()
    tracker.complete_file = MagicMock()
    return tracker


def _dialogue_lines():
    return [
        DialogueLine(start_ms=0, end_ms=1000, text='Hello'),
        DialogueLine(start_ms=1000, end_ms=2000, text='World'),
    ]


# ---------------------------------------------------------------------------
# Tests: _make_file_tag
# ---------------------------------------------------------------------------


class TestMakeFileTag:
    def test_short_name(self):
        assert _make_file_tag(Path('/videos/ep01.mkv')) == 'ep01'

    def test_truncates_long_name(self):
        tag = _make_file_tag(Path('/videos/this_is_a_very_long_episode_name.mkv'))
        assert len(tag) == 20
        assert tag.endswith('...')

    def test_exactly_20_chars(self):
        tag = _make_file_tag(Path('/videos/12345678901234567890.mkv'))
        assert tag == '12345678901234567890'
        assert len(tag) == 20


# ---------------------------------------------------------------------------
# Tests: process_file
# ---------------------------------------------------------------------------


class TestProcessFile:
    async def test_runs_all_stages(self, tmp_path):
        """All stages should be invoked in order for a simple happy path."""
        video_path = tmp_path / 'test.mkv'
        video_path.touch()
        work_dir = tmp_path / 'work'
        work_dir.mkdir()

        lines = _dialogue_lines()
        english_source = tmp_path / 'english.srt'
        english_source.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello\n')

        def set_identify(ctx):
            ctx.identity = {'title': 'Test'}

        def set_reference(ctx):
            ctx.reference_path = english_source

        def set_english(ctx):
            ctx.english_source = english_source
            ctx.dialogue_lines = lines

        def set_tracks(ctx):
            ctx.subtitle_tracks = []

        stages = {
            'identify': FakeStage('identify', side_effect=set_identify),
            'extract_ref': FakeStage('extract_reference', side_effect=set_reference),
            'fetch': FakeStage('fetch'),
            'extract_english': FakeStage('extract', side_effect=set_english),
            'translate': FakeTranslateStage(),
            'create_tracks': FakeStage('create_tracks', side_effect=set_tracks),
            'mux': FakeStage('mux'),
        }

        config = _make_test_config()
        tracker = _make_fake_tracker()

        # GPU queue that returns translated lines for TranslateTask
        gpu_queue = GpuQueue()
        gpu_queue.start()

        # We need to mock gpu_queue.submit to return translated lines
        translated = [
            DialogueLine(start_ms=0, end_ms=1000, text='Czesc'),
            DialogueLine(start_ms=1000, end_ms=2000, text='Swiat'),
        ]
        original_submit = gpu_queue.submit

        async def fake_submit(task):
            from movie_translator.gpu_queue import TranslateTask as TT

            if isinstance(task, TT):
                return translated
            return await original_submit(task)

        gpu_queue.submit = fake_submit  # ty: ignore[invalid-assignment]

        result = await process_file(
            video_path=video_path,
            work_dir=work_dir,
            config=config,
            stages=stages,  # ty: ignore[invalid-argument-type]
            gpu_queue=gpu_queue,
            tracker=tracker,
            metrics=NullCollector(),
        )

        assert result is True
        assert stages['identify'].call_count == 1
        assert stages['extract_ref'].call_count == 1
        assert stages['fetch'].call_count == 1
        assert stages['extract_english'].call_count == 1
        assert stages['create_tracks'].call_count == 1
        assert stages['mux'].call_count == 1

        await gpu_queue.shutdown()

    async def test_returns_false_on_failure(self, tmp_path):
        """A stage that raises should cause process_file to return False."""
        video_path = tmp_path / 'test.mkv'
        video_path.touch()
        work_dir = tmp_path / 'work'
        work_dir.mkdir()

        def blow_up(ctx):
            raise RuntimeError('stage failed')

        stages = {
            'identify': FakeStage('identify', side_effect=blow_up),
            'extract_ref': FakeStage('extract_reference'),
            'fetch': FakeStage('fetch'),
            'extract_english': FakeStage('extract'),
            'translate': FakeTranslateStage(),
            'create_tracks': FakeStage('create_tracks'),
            'mux': FakeStage('mux'),
        }

        config = _make_test_config()
        tracker = _make_fake_tracker()
        gpu_queue = GpuQueue()
        gpu_queue.start()

        result = await process_file(
            video_path=video_path,
            work_dir=work_dir,
            config=config,
            stages=stages,  # ty: ignore[invalid-argument-type]
            gpu_queue=gpu_queue,
            tracker=tracker,
        )

        assert result is False
        await gpu_queue.shutdown()


# ---------------------------------------------------------------------------
# Tests: run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    async def test_processes_multiple_files(self, tmp_path, monkeypatch):
        """All files should be processed when using run_all."""
        video_files = []
        for i in range(3):
            vp = tmp_path / f'ep{i:02d}.mkv'
            vp.touch()
            video_files.append(vp)

        config = _make_test_config(workers=2, enable_fetch=False)
        tracker = _make_fake_tracker()

        # Mock has_polish_subtitles to return False
        monkeypatch.setattr(
            'movie_translator.async_pipeline.SubtitleExtractor.has_polish_subtitles',
            lambda self, path: False,
        )

        # Track which files were processed
        processed_files = []

        # Mock process_file to just record the call and succeed
        async def mock_process_file(
            video_path, work_dir, config, stages, gpu_queue, tracker, metrics=None, display_name=''
        ):
            processed_files.append(video_path)
            return True

        monkeypatch.setattr('movie_translator.async_pipeline.process_file', mock_process_file)

        gpu_queue = GpuQueue()
        gpu_queue.start()

        results = await run_all(
            video_files=video_files,
            root_dir=tmp_path,
            config=config,
            metrics=NullCollector(),
            gpu_queue=gpu_queue,
            tracker=tracker,
        )

        await gpu_queue.shutdown()

        assert len(results) == 3
        assert all(status == 'success' for _, status in results)
        assert set(processed_files) == set(video_files)

    async def test_skips_files_with_polish_subs(self, tmp_path, monkeypatch):
        """Files with existing Polish subtitles should be skipped."""
        video_files = []
        for i in range(2):
            vp = tmp_path / f'ep{i:02d}.mkv'
            vp.touch()
            video_files.append(vp)

        config = _make_test_config(workers=2)
        tracker = _make_fake_tracker()

        # First file has Polish subs, second does not
        def mock_has_polish(self, path):
            return path.name == 'ep00.mkv'

        monkeypatch.setattr(
            'movie_translator.async_pipeline.SubtitleExtractor.has_polish_subtitles',
            mock_has_polish,
        )

        processed = []

        async def mock_process_file(
            video_path, work_dir, config, stages, gpu_queue, tracker, metrics=None, display_name=''
        ):
            processed.append(video_path)
            return True

        monkeypatch.setattr('movie_translator.async_pipeline.process_file', mock_process_file)

        gpu_queue = GpuQueue()
        gpu_queue.start()

        results = await run_all(
            video_files=video_files,
            root_dir=tmp_path,
            config=config,
            metrics=NullCollector(),
            gpu_queue=gpu_queue,
            tracker=tracker,
        )

        await gpu_queue.shutdown()

        statuses = {path.name: status for path, status in results}
        assert statuses['ep00.mkv'] == 'skipped'
        assert statuses['ep01.mkv'] == 'success'
        assert len(processed) == 1
        assert processed[0].name == 'ep01.mkv'

    async def test_worker_limit_respected(self, tmp_path, monkeypatch):
        """With workers=1, max concurrency should be 1 — never two files at once."""
        video_files = []
        for i in range(3):
            vp = tmp_path / f'ep{i:02d}.mkv'
            vp.touch()
            video_files.append(vp)

        config = _make_test_config(workers=1)
        tracker = _make_fake_tracker()

        monkeypatch.setattr(
            'movie_translator.async_pipeline.SubtitleExtractor.has_polish_subtitles',
            lambda self, path: False,
        )

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_process_file(
            video_path, work_dir, config, stages, gpu_queue, tracker, metrics=None, display_name=''
        ):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            # Yield control so other coroutines could (incorrectly) run
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return True

        monkeypatch.setattr('movie_translator.async_pipeline.process_file', mock_process_file)

        gpu_queue = GpuQueue()
        gpu_queue.start()

        results = await run_all(
            video_files=video_files,
            root_dir=tmp_path,
            config=config,
            metrics=NullCollector(),
            gpu_queue=gpu_queue,
            tracker=tracker,
        )

        await gpu_queue.shutdown()

        assert len(results) == 3
        assert max_concurrent == 1, f'Expected max 1 concurrent worker, got {max_concurrent}'

    async def test_worker_limit_allows_parallel(self, tmp_path, monkeypatch):
        """With workers=3 and 3 files, all should be able to run concurrently."""
        video_files = []
        for i in range(3):
            vp = tmp_path / f'ep{i:02d}.mkv'
            vp.touch()
            video_files.append(vp)

        config = _make_test_config(workers=3)
        tracker = _make_fake_tracker()

        monkeypatch.setattr(
            'movie_translator.async_pipeline.SubtitleExtractor.has_polish_subtitles',
            lambda self, path: False,
        )

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()
        barrier = asyncio.Barrier(3)

        async def mock_process_file(
            video_path, work_dir, config, stages, gpu_queue, tracker, metrics=None, display_name=''
        ):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            # All 3 must reach this point concurrently
            await barrier.wait()
            async with lock:
                current_concurrent -= 1
            return True

        monkeypatch.setattr('movie_translator.async_pipeline.process_file', mock_process_file)

        gpu_queue = GpuQueue()
        gpu_queue.start()

        results = await run_all(
            video_files=video_files,
            root_dir=tmp_path,
            config=config,
            metrics=NullCollector(),
            gpu_queue=gpu_queue,
            tracker=tracker,
        )

        await gpu_queue.shutdown()

        assert len(results) == 3
        assert max_concurrent == 3, f'Expected 3 concurrent workers, got {max_concurrent}'
