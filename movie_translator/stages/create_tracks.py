"""Create subtitle track files and build the track list."""

import json
from pathlib import Path

from ..context import PipelineContext
from ..logging import logger
from ..subtitles import SubtitleProcessor
from ..types import SubtitleFile

# Language code mapping for external subtitle tracks
_LANG_TO_TRACK = {'pl': 'pol', 'en': 'eng'}


def _load_external_subs(
    external_dir: Path,
    identity: object,
) -> list[SubtitleFile]:
    """Load matching external subtitles from a manifest directory.

    Matches by parsed_title + season + episode (identity-based),
    falling back to filename stem matching.
    """
    manifest_path = external_dir / 'manifest.json'
    if not manifest_path.exists():
        logger.warning(f'No manifest.json found in {external_dir}')
        return []

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest.get('entries', [])

    # Get identity fields from current video
    cur_title = (getattr(identity, 'parsed_title', '') or '').lower()
    cur_season = getattr(identity, 'season', None)
    cur_episode = getattr(identity, 'episode', None)

    tracks: list[SubtitleFile] = []

    for entry in entries:
        eid = entry.get('identity', {})
        entry_title = (eid.get('parsed_title') or '').lower()
        entry_season = eid.get('season')
        entry_episode = eid.get('episode')

        # Match on title + season + episode
        title_match = (
            cur_title and entry_title and (cur_title in entry_title or entry_title in cur_title)
        )
        episode_match = (
            cur_season is not None
            and cur_episode is not None
            and cur_season == entry_season
            and cur_episode == entry_episode
        )

        if not (title_match and episode_match):
            # Fallback: try matching on episode number alone (for simple numbering)
            if not (cur_episode is not None and cur_episode == entry_episode):
                continue

        for sub in entry.get('subtitles', []):
            sub_path = external_dir / sub['file']
            if not sub_path.exists():
                logger.warning(f'External subtitle not found: {sub_path}')
                continue

            lang_code = _LANG_TO_TRACK.get(sub['language'], sub['language'])
            method = sub.get('method', 'unknown')
            title = f'{sub["language"].upper()} (external, {method})'
            tracks.append(SubtitleFile(sub_path, lang_code, title, is_default=False))
            logger.info(f'Adding external subtitle: {sub["file"]}')

    return tracks


class CreateTracksStage:
    name = 'create_tracks'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        is_mkv = ctx.video_path.suffix.lower() == '.mkv'
        replace_chars = False

        assert ctx.font_info is not None
        assert ctx.english_source is not None
        assert ctx.translated_lines is not None

        if not ctx.font_info.supports_polish:
            if is_mkv and ctx.font_info.font_attachments:
                logger.info(f'Will embed font "{ctx.font_info.font_attachments[0].name}"')
            elif is_mkv:
                logger.warning('No system font with Polish support, replacing characters')
                replace_chars = True
            else:
                replace_chars = True

        # Create AI Polish subtitle file
        with ctx.metrics.span('create_polish_subtitles'):
            ai_polish_ass = ctx.work_dir / f'{ctx.video_path.stem}_polish_ai.ass'
            SubtitleProcessor.create_polish_subtitles(
                ctx.english_source,
                ctx.translated_lines,
                ai_polish_ass,
                replace_chars,
            )

        if ctx.font_info.fallback_font_family:
            with ctx.metrics.span('override_font'):
                SubtitleProcessor.override_font_name(
                    ai_polish_ass, ctx.font_info.fallback_font_family
                )

        # Build track list
        with ctx.metrics.span('build_track_list') as s:
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

            # Add external subtitles if configured
            if ctx.config.external_subs_dir and ctx.identity:
                external_tracks = _load_external_subs(ctx.config.external_subs_dir, ctx.identity)
                tracks.extend(external_tracks)
                if external_tracks:
                    s.detail('external_tracks', len(external_tracks))

            s.detail('tracks', len(tracks))

        ctx.subtitle_tracks = tracks
        return ctx
