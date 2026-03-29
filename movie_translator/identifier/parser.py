import aniparse
from guessit import guessit


def parse_filename(
    filename: str,
    folder_name: str | None = None,
) -> dict:
    """Parse a video filename (and optional folder name) into structured metadata.

    Tries aniparse first (anime-specialized), falls back to guessit.
    Returns dict with keys: title, year, season, episode, media_type, is_anime, release_group.
    Missing fields are None.
    """
    # Try aniparse first — handles anime bracket patterns much better than guessit
    ani = _parse_with_aniparse(filename)

    # Always also run guessit as the general-purpose fallback
    info = guessit(filename)

    # Decide which title source to use
    ani_title = ani.get('title')
    guessit_title = info.get('title')

    # Prefer aniparse title when it looks like anime (has release_group or brackets in filename)
    is_anime = ani.get('is_anime', False)
    if is_anime and ani_title:
        title = ani_title
    else:
        title = guessit_title or ani_title

    # Merge episode/season — prefer aniparse for anime, guessit otherwise
    if is_anime:
        season = ani.get('season') or info.get('season')
        episode = ani.get('episode') or info.get('episode')
    else:
        season = info.get('season') or ani.get('season')
        episode = info.get('episode') or ani.get('episode')

    year = ani.get('year') or info.get('year')
    release_group = ani.get('release_group')

    # If neither parser could extract a title but we have a folder name, try that
    if not title and folder_name:
        folder_info = guessit(folder_name)
        title = folder_info.get('title', folder_name)
        if not season:
            season = folder_info.get('season')

    # Determine media type
    guess_type = info.get('type', 'movie')
    if season is not None or episode is not None:
        media_type = 'episode'
    elif guess_type == 'episode':
        media_type = 'episode'
    else:
        media_type = 'movie'

    return {
        'title': title,
        'year': year,
        'season': season,
        'episode': episode,
        'media_type': media_type,
        'is_anime': is_anime,
        'release_group': release_group,
    }


def _parse_with_aniparse(filename: str) -> dict:
    """Extract metadata from a filename using aniparse.

    Returns a normalized dict with our standard keys.
    """
    try:
        result = aniparse.parse(filename)
    except Exception:
        return {}

    if not result:
        return {}

    out: dict = {}

    # Extract release group — strong anime signal
    groups = result.get('release_group', [])
    if groups:
        out['release_group'] = groups[0]

    # Extract series info
    series_list = result.get('series', [])
    if series_list:
        series = series_list[0]
        out['title'] = series.get('title')

        episodes = series.get('episode', [])
        if episodes:
            out['episode'] = episodes[0].get('number')

        seasons = series.get('season', [])
        if seasons:
            out['season'] = seasons[0].get('number')

        years = series.get('year', [])
        if years:
            out['year'] = years[0].get('number')

    # Anime detection: release_group is the strongest signal
    # (fansub groups like [HorribleSubs], [Erai-raws], [SubsPlease] are anime-specific)
    out['is_anime'] = bool(groups)

    return out
