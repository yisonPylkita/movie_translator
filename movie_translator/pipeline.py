"""Thin pipeline orchestrator — chains stages sequentially."""

from pathlib import Path

from .context import PipelineConfig, PipelineContext
from .logging import logger
from .stages import (
    CreateTracksStage,
    ExtractEnglishStage,
    ExtractReferenceStage,
    FetchSubtitlesStage,
    IdentifyStage,
    MuxStage,
    TranslateStage,
)


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        enable_inpaint: bool = False,
        tracker=None,
    ):
        self.config = PipelineConfig(
            device=device,
            batch_size=batch_size,
            model=model,
            enable_fetch=enable_fetch,
            enable_inpaint=enable_inpaint,
        )
        self.tracker = tracker
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
        ctx = PipelineContext(video_path=video_path, work_dir=work_dir, config=self.config)

        try:
            for stage in self.stages:
                if self.tracker:
                    self.tracker.set_stage(stage.name)
                ctx = stage.run(ctx)
            return True
        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False
