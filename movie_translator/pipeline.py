"""Thin pipeline orchestrator — chains stages sequentially."""

from pathlib import Path

from .context import PipelineConfig, PipelineContext
from .logging import logger
from .metrics.collector import NullCollector
from .stages import (
    CreateTracksStage,
    ExtractEnglishStage,
    ExtractReferenceStage,
    FetchSubtitlesStage,
    IdentifyStage,
    MuxStage,
    TranslateStage,
)
from .translation import ModelCache


def _resolve_pending_ocr(ctx: PipelineContext, stage_label: str) -> None:
    """Resolve a pending OCR task synchronously (for the sync pipeline).

    In the async pipeline this is handled by the GPU queue instead.
    """
    if ctx.pending_ocr is None:
        return

    pending = ctx.pending_ocr

    if pending.type == 'pgs':
        from .ocr.pgs_extractor import extract_pgs_track

        result = extract_pgs_track(ctx.video_path, pending.track_id or 0, pending.output_dir)
        if stage_label == 'extract_ref' and result is not None:
            ctx.reference_path = result
        elif stage_label == 'extract_english' and result is not None:
            ctx.english_source = result

    elif pending.type == 'burned_in':
        from .ocr import extract_burned_in_subtitles, probe_for_burned_in_subtitles

        ctx.burned_in_probed = True
        detected = probe_for_burned_in_subtitles(ctx.video_path)
        if detected:
            result = extract_burned_in_subtitles(ctx.video_path, pending.output_dir)
            if result:
                if stage_label == 'extract_ref':
                    ctx.reference_path = result.srt_path
                    ctx.ocr_results = result.ocr_results
                elif stage_label == 'extract_english':
                    ctx.english_source = result.srt_path
                    ctx.ocr_results = result.ocr_results

    # If extract_english deferred OCR and we now have a source, extract dialogue
    if (
        stage_label == 'extract_english'
        and ctx.english_source is not None
        and ctx.dialogue_lines is None
    ):
        from .subtitles import SubtitleProcessor

        ctx.dialogue_lines = SubtitleProcessor.extract_dialogue_lines(ctx.english_source)

    ctx.pending_ocr = None


# Stages that may produce pending_ocr requiring inline resolution
_OCR_STAGE_LABELS = {
    'extract_reference': 'extract_ref',
    'extract': 'extract_english',
}


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        enable_inpaint: bool = False,
        tracker=None,
        metrics=None,
        external_subs_dir: Path | None = None,
    ):
        self.config = PipelineConfig(
            device=device,
            batch_size=batch_size,
            model=model,
            enable_fetch=enable_fetch,
            enable_inpaint=enable_inpaint,
            external_subs_dir=external_subs_dir,
            model_cache=ModelCache(),
        )
        self.tracker = tracker
        self.metrics = metrics
        self.last_identity = None
        self.stages = [
            IdentifyStage(),
            ExtractReferenceStage(),
            FetchSubtitlesStage(),
            ExtractEnglishStage(),
            TranslateStage(),
            CreateTracksStage(),
            MuxStage(),
        ]

    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self.config.dry_run = dry_run
        ctx = PipelineContext(
            video_path=video_path,
            work_dir=work_dir,
            config=self.config,
            metrics=self.metrics or NullCollector(),
        )

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                    set_tracker = getattr(stage, 'set_tracker', None)
                    if set_tracker is not None:
                        set_tracker(self.tracker)
                with ctx.metrics.span(stage.name):
                    ctx = stage.run(ctx)

                # Resolve deferred OCR tasks synchronously
                ocr_label = _OCR_STAGE_LABELS.get(stage.name)
                if ocr_label and ctx.pending_ocr:
                    _resolve_pending_ocr(ctx, ocr_label)

            self.last_identity = ctx.identity
            return True
        except Exception as e:
            self.last_identity = ctx.identity
            logger.error(f'Failed: {video_path.name} - {e}')
            return False
