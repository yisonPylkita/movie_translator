"""Create subtitle track files and build the track list."""

from ..context import PipelineContext
from ..logging import logger
from ..subtitles import SubtitleProcessor
from ..types import SubtitleFile

# Reliable fallback font for AI-translated subtitles. Used when no system
# font with Polish support was detected. Must be widely available across
# players and operating systems.
_DEFAULT_AI_FONT = 'Arial'


class CreateTracksStage:
    name = 'create_tracks'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        is_mkv = ctx.video_path.suffix.lower() == '.mkv'
        replace_chars = False

        if not ctx.font_info.supports_polish:
            if is_mkv and ctx.font_info.font_attachments:
                logger.info(f'Will embed font "{ctx.font_info.font_attachments[0].name}"')
            elif is_mkv:
                logger.warning('No system font with Polish support, replacing characters')
                replace_chars = True
            else:
                replace_chars = True

        # Create AI Polish subtitle file
        ai_polish_ass = ctx.work_dir / f'{ctx.video_path.stem}_polish_ai.ass'
        SubtitleProcessor.create_polish_subtitles(
            ctx.english_source,
            ctx.translated_lines,
            ai_polish_ass,
            replace_chars,
        )

        # The AI subtitle inherits styles from the English source, which
        # often uses custom fansub fonts. These may not be available to the
        # player for the AI track (embedded fonts apply to the original track
        # only). Always set a reliable, widely available font.
        ai_font = ctx.font_info.fallback_font_family or _DEFAULT_AI_FONT
        SubtitleProcessor.override_font_name(ai_polish_ass, ai_font)

        # Build track list
        fetched_pol_list = ctx.fetched_subtitles.get('pol', []) if ctx.fetched_subtitles else []
        tracks: list[SubtitleFile] = []

        for i, fetched_pol in enumerate(fetched_pol_list):
            pol_title = f'Polish ({fetched_pol.source})'
            tracks.append(SubtitleFile(fetched_pol.path, 'pol', pol_title, is_default=(i == 0)))
            if ctx.font_info.fallback_font_family:
                SubtitleProcessor.override_font_name(
                    fetched_pol.path,
                    ctx.font_info.fallback_font_family,
                )

        tracks.append(
            SubtitleFile(
                ai_polish_ass,
                'pol',
                'Polish (AI)',
                is_default=not bool(fetched_pol_list),
            )
        )

        ctx.subtitle_tracks = tracks
        return ctx
