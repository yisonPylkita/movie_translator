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

    def update_current_video(self, **fields: Any) -> None:
        """Update fields on the current video being collected."""
        if self._current_video is not None:
            self._current_video.update(fields)

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
