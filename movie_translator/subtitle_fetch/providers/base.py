from pathlib import Path
from typing import Protocol, runtime_checkable

from ...identifier.types import MediaIdentity
from ..types import SubtitleMatch


@runtime_checkable
class SubtitleProvider(Protocol):
    """Protocol for subtitle download providers."""

    @property
    def name(self) -> str: ...

    def search(
        self, identity: MediaIdentity, languages: list[str]
    ) -> list[SubtitleMatch]:
        """Search for subtitles. Returns matches sorted by score descending."""
        ...

    def download(self, match: SubtitleMatch, output_path: Path) -> Path:
        """Download subtitle file. Returns the path written to."""
        ...
