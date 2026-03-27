from guessit import guessit


def parse_filename(
    filename: str,
    folder_name: str | None = None,
) -> dict:
    """Parse a video filename (and optional folder name) into structured metadata.

    Returns dict with keys: title, year, season, episode, media_type.
    Missing fields are None.
    """
    info = guessit(filename)

    title = info.get('title')
    season = info.get('season')
    episode = info.get('episode')
    year = info.get('year')

    # If guessit couldn't extract a title but we have a folder name, try that
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
    }
