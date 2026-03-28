"""Pipeline stages."""

from .create_tracks import CreateTracksStage
from .extract_english import ExtractEnglishStage
from .extract_ref import ExtractReferenceStage
from .fetch import FetchSubtitlesStage
from .identify import IdentifyStage
from .mux import MuxStage
from .translate import TranslateStage

__all__ = [
    'CreateTracksStage',
    'ExtractEnglishStage',
    'ExtractReferenceStage',
    'FetchSubtitlesStage',
    'IdentifyStage',
    'MuxStage',
    'TranslateStage',
]
