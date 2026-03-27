from typing import NamedTuple


class SubtitleMatch(NamedTuple):
    language: str  # ISO 639-2B code ('eng', 'pol')
    source: str  # Provider name (e.g., 'opensubtitles')
    subtitle_id: str  # Provider-specific identifier
    release_name: str  # Subtitle release name
    format: str  # File format ('srt', 'ass', 'sub')
    score: float  # Match confidence 0.0-1.0
    hash_match: bool  # True if matched by file hash
