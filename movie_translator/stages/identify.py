"""Identify stage — extract media identity from video file."""

from ..context import PipelineContext
from ..identifier import identify_media
from ..logging import logger


class IdentifyStage:
    name = 'identify'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Identifying: {ctx.video_path.name}')
        ctx.identity = identify_media(ctx.video_path)

        if ctx.identity is not None and hasattr(ctx.identity, 'media_type'):
            identity = ctx.identity
            with ctx.metrics.span('record_identity') as s:
                s.detail('title', identity.title)
                s.detail('media_type', identity.media_type)
                s.detail('is_anime', identity.is_anime)
                if identity.season is not None:
                    s.detail('season', identity.season)
                if identity.episode is not None:
                    s.detail('episode', identity.episode)

        return ctx
