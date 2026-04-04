"""Pipeline context and configuration dataclasses.

These dataclasses are populated progressively as the pipeline runs,
giving each stage typed access to the outputs of previous stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from movie_translator.metrics.collector import MetricsCollector, NullCollector
from movie_translator.types import DialogueLine, OCRResult, SubtitleFile

if TYPE_CHECKING:
    from movie_translator.translation.model_cache import ModelCache


@dataclass
class PipelineConfig:
    device: str = 'mps'
    batch_size: int = 16
    model: str = 'allegro'
    enable_fetch: bool = True
    enable_inpaint: bool = False
    dry_run: bool = False
    workers: int = 4
    external_subs_dir: Path | None = None
    model_cache: ModelCache | None = None


@dataclass
class FetchedSubtitle:
    path: Path
    source: str  # provider name, e.g. "animesub"


@dataclass
class FontInfo:
    supports_polish: bool
    font_attachments: list[Path] = field(default_factory=list)
    fallback_font_family: str | None = None


@dataclass
class OriginalTrack:
    stream_index: int
    subtitle_index: int
    codec: str  # "subrip", "ass", etc.
    language: str


@dataclass
class PendingOcr:
    """Typed description of an OCR task deferred to the GPU queue.

    Set by run() on ExtractReferenceStage or ExtractEnglishStage,
    consumed by _handle_pending_ocr() in async_pipeline.
    """

    type: str  # 'pgs' or 'burned_in'
    track_id: int | None = None
    output_dir: Path = field(default_factory=Path)


@dataclass
class PipelineContext:
    # Inputs (set before pipeline runs)
    video_path: Path
    work_dir: Path
    config: PipelineConfig

    # Stage outputs (set progressively by each stage)
    identity: object | None = None  # MediaIdentity (avoid circular import)
    reference_path: Path | None = None
    original_english_track: OriginalTrack | None = None
    fetched_subtitles: dict[str, list[FetchedSubtitle]] | None = None
    english_source: Path | None = None
    dialogue_lines: list[DialogueLine] | None = None
    translated_lines: list[DialogueLine] | None = None
    font_info: FontInfo | None = None
    subtitle_tracks: list[SubtitleFile] | None = None
    ocr_results: list[OCRResult] | None = None
    inpainted_video: Path | None = None
    # Coordination flag: True after any stage has probed for burned-in subtitles.
    # Prevents ExtractEnglishStage from re-probing after ExtractReferenceStage already did.
    burned_in_probed: bool = False
    pending_ocr: PendingOcr | None = None
    metrics: MetricsCollector | NullCollector = field(default_factory=NullCollector)
