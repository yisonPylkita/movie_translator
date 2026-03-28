from typing import NamedTuple


class MediaIdentity(NamedTuple):
    title: str  # Best-guess title (container metadata preferred)
    parsed_title: str  # Title from filename parsing (cleaner, better for text search)
    year: int | None  # Release year
    season: int | None  # Season number
    episode: int | None  # Episode number
    media_type: str  # 'movie' or 'episode'
    oshash: str  # OpenSubtitles file hash (16 hex chars)
    file_size: int  # Bytes (needed for OpenSubtitles API)
    raw_filename: str  # Original filename for fallback search
    imdb_id: str | None = None  # e.g. 'tt0903747'
    tmdb_id: int | None = None  # TMDB numeric ID
    is_anime: bool = False  # True if detected as anime (via aniparse release_group, etc.)
    release_group: str | None = None  # Fansub/release group (e.g. 'HorribleSubs')
