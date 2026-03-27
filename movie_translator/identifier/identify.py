from pathlib import Path

from ..logging import logger
from .hasher import compute_oshash
from .metadata import extract_container_metadata
from .parser import parse_filename
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

    # Merge: container overrides filename
    title = container.get('title') or parsed.get('title') or filename
    season = parsed.get('season')
    episode = parsed.get('episode')
    year = parsed.get('year')
    media_type = parsed.get('media_type', 'movie')

    # If container has episode info, try to use it
    container_episode = container.get('episode')
    if container_episode and not episode:
        try:
            episode = int(container_episode)
        except (ValueError, TypeError):
            pass

    logger.info(f'Identified: "{title}" (type={media_type}, S{season}E{episode}, year={year})')

    return MediaIdentity(
        title=title,
        year=year,
        season=season,
        episode=episode,
        media_type=media_type,
        oshash=oshash,
        file_size=file_size,
        raw_filename=filename,
    )
