"""Tests for the GPU work queue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from movie_translator.gpu_queue import GpuQueue, GpuTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StubTask(GpuTask):
    model_type: str = field(init=False, default='stub')
    value: Any = 42

    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> Any:
        return self.value


@dataclass
class RecordingTask(GpuTask):
    """Records execution order and model_cache contents."""

    model_type: str = field(init=False, default='recorder')
    label: str = ''
    log: list[str] = field(default_factory=list)

    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> str:
        self.log.append(self.label)
        model_cache.setdefault('seen', []).append(self.label)
        return self.label


@dataclass
class FailingTask(GpuTask):
    model_type: str = field(init=False, default='fail')

    def execute(self, model_cache: dict[str, Any], last_model_type: str | None) -> Any:
        raise RuntimeError('boom')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGpuQueue:
    async def test_submit_returns_result(self):
        q = GpuQueue()
        q.start()
        result = await q.submit(StubTask(value=99))
        assert result == 99
        await q.shutdown()

    async def test_fifo_ordering(self):
        q = GpuQueue()
        log: list[str] = []
        tasks = [RecordingTask(label=f't{i}', log=log) for i in range(5)]
        q.start()
        results = await asyncio.gather(*(q.submit(t) for t in tasks))
        await q.shutdown()
        assert list(results) == ['t0', 't1', 't2', 't3', 't4']
        assert log == ['t0', 't1', 't2', 't3', 't4']

    async def test_error_isolation(self):
        q = GpuQueue()
        q.start()

        with pytest.raises(RuntimeError, match='boom'):
            await q.submit(FailingTask())

        # Queue still works after a failure
        result = await q.submit(StubTask(value='ok'))
        assert result == 'ok'
        await q.shutdown()

    async def test_model_cache_shared_between_tasks(self):
        q = GpuQueue()
        log: list[str] = []
        q.start()
        await q.submit(RecordingTask(label='a', log=log))
        await q.submit(RecordingTask(label='b', log=log))
        await q.shutdown()
        # Both tasks wrote into the same model_cache dict
        assert q._model_cache.get('seen') == ['a', 'b']

    async def test_shutdown_stops_worker(self):
        q = GpuQueue()
        q.start()
        await q.shutdown()
        assert q._worker_task is not None
        assert q._worker_task.done()

    async def test_pending_count(self):
        q = GpuQueue()
        # Don't start worker yet so items accumulate
        for i in range(3):
            await q._queue.put((StubTask(value=i), asyncio.get_event_loop().create_future()))
        assert q.pending == 3
        # Clean up: drain and start/stop
        while not q._queue.empty():
            q._queue.get_nowait()
        q.start()
        await q.shutdown()
