from pathlib import Path

from ..ffmpeg import get_video_info
from ..logging import logger


def extract_container_metadata(video_path: str | Path) -> dict:
    """Extract title and episode metadata from video container tags.

    Returns dict with keys: title, episode. Missing fields are None.
    """
    try:
        info = get_video_info(Path(video_path))
    except Exception as e:
        logger.debug(f'Could not read container metadata: {e}')
        return {'title': None, 'episode': None}

    tags = info.get('format', {}).get('tags', {})

    # Common tag names for title and episode across containers
    title = tags.get('title') or tags.get('TITLE')
    episode = tags.get('episode_id') or tags.get('episode_sort') or tags.get('track')

    return {
        'title': title,
        'episode': episode,
    }
