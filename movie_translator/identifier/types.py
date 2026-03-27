from typing import NamedTuple


class MediaIdentity(NamedTuple):
    title: str  # Best-guess title
    year: int | None  # Release year
    season: int | None  # Season number
    episode: int | None  # Episode number
    media_type: str  # 'movie' or 'episode'
    oshash: str  # OpenSubtitles file hash (16 hex chars)
    file_size: int  # Bytes (needed for OpenSubtitles API)
    raw_filename: str  # Original filename for fallback search
