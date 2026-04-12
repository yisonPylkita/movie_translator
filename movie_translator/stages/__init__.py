"""Pipeline stages."""

from __future__ import annotations

from typing import Protocol

from ..context import PipelineContext
from .create_tracks import CreateTracksStage
from .extract_english import ExtractEnglishStage
from .extract_ref import ExtractReferenceStage
from .fetch import FetchSubtitlesStage
from .identify import IdentifyStage
from .mux import MuxStage
from .translate import TranslateStage


class Stage(Protocol):
    """Common interface for all pipeline stages."""

    name: str

    def run(self, ctx: PipelineContext) -> PipelineContext: ...


__all__ = [
    'CreateTracksStage',
    'ExtractEnglishStage',
    'ExtractReferenceStage',
    'FetchSubtitlesStage',
    'IdentifyStage',
    'MuxStage',
    'Stage',
    'TranslateStage',
]
