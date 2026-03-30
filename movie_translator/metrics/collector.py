"""Metrics collector with context-manager span API."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar, Token
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
        self._token: Token[str] | None = None

    def __enter__(self) -> Span:
        self._start = time.perf_counter()
        self._token = _prefix.set(self._name)
        return self

    def __exit__(self, *exc: object) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000
        assert self._token is not None
        _prefix.reset(self._token)
        self._collector.emit(
            SpanEvent(
                name=self._name,
                duration_ms=duration_ms,
                details=self._details,
            )
        )
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
