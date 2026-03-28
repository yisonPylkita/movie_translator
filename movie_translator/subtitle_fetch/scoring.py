"""Subtitle match scoring utilities.

Provides release name token matching to supplement provider-level scoring.
"""

import re


def _tokenize(name: str) -> set[str]:
    """Split a release name into lowercase tokens."""
    if not name:
        return set()
    return set(re.split(r'[\.\-_\s\[\]()]+', name.lower())) - {''}


def compute_release_score(video_name: str, release_name: str) -> float:
    """Score how well a subtitle release name matches a video filename.

    Returns 0.0 to 1.0 based on token overlap (Jaccard similarity).
    """
    video_tokens = _tokenize(video_name)
    release_tokens = _tokenize(release_name)

    if not video_tokens or not release_tokens:
        return 0.0

    intersection = video_tokens & release_tokens
    union = video_tokens | release_tokens

    return len(intersection) / len(union)
