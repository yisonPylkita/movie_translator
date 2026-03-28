"""Identify stage — extract media identity from video file."""

from ..context import PipelineContext
from ..identifier import identify_media
from ..logging import logger


class IdentifyStage:
    name = 'identify'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Identifying: {ctx.video_path.name}')
        ctx.identity = identify_media(ctx.video_path)
        return ctx
