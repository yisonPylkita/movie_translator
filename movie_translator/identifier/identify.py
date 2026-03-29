from pathlib import Path

from ..logging import logger
from .hasher import compute_oshash
from .metadata import extract_container_metadata
from .parser import parse_filename
from .tmdb import lookup_tmdb
from .types import MediaIdentity


def identify_media(video_path: Path) -> MediaIdentity:
    """Identify a video file using filename, container metadata, and file hash.

    Combines multiple signals with priority:
    container metadata > filename > folder name.
    """
    filename = video_path.name
    folder_name = video_path.parent.name

    # Signal 1: Parse filename (and folder as fallback context)
    parsed = parse_filename(filename, folder_name=folder_name)

    # Signal 2: Container metadata (overrides filename when present)
    container = extract_container_metadata(video_path)

    # Signal 3: File hash
    try:
        oshash = compute_oshash(video_path)
    except Exception as e:
        logger.warning(f'Could not compute file hash: {e}')
        oshash = ''

    file_size = video_path.stat().st_size

    # Merge: container overrides filename for "title", but keep parsed title separately
    # (parsed title is cleaner and better for text-based subtitle searches)
    parsed_title = parsed.get('title') or filename
    title = container.get('title') or parsed_title
    season = parsed.get('season')
    episode = parsed.get('episode')
    year = parsed.get('year')
    media_type = parsed.get('media_type', 'movie')
    is_anime = parsed.get('is_anime', False)
    release_group = parsed.get('release_group')

    # If container has episode info, try to use it
    container_episode = container.get('episode')
    if container_episode and not episode:
        try:
            episode = int(container_episode)
        except (ValueError, TypeError):
            pass

    anime_tag = ' [anime]' if is_anime else ''
    logger.info(
        f'Identified: "{title}" (type={media_type}, S{season}E{episode}, year={year}){anime_tag}'
    )

    # Signal 4: TMDB enrichment (optional, requires TMDB_API_KEY)
    imdb_id = None
    tmdb_id = None
    try:
        tmdb_result = lookup_tmdb(parsed_title, year, media_type)
        if tmdb_result:
            tmdb_id = tmdb_result.get('tmdb_id')
            imdb_id = tmdb_result.get('imdb_id')
            logger.debug(f'TMDB enrichment: tmdb_id={tmdb_id}, imdb_id={imdb_id}')
    except Exception as e:
        logger.debug(f'TMDB enrichment skipped: {e}')

    return MediaIdentity(
        title=title,
        parsed_title=parsed_title,
        year=year,
        season=season,
        episode=episode,
        media_type=media_type,
        oshash=oshash,
        file_size=file_size,
        raw_filename=filename,
        imdb_id=imdb_id,
        tmdb_id=tmdb_id,
        is_anime=is_anime,
        release_group=release_group,
    )
