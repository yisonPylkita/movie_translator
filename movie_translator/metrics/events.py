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
