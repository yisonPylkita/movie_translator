"""NapiProjekt hash: MD5 of the first 10MB of a video file."""

import hashlib
from pathlib import Path

NAPIPROJEKT_READ_SIZE = 10 * 1024 * 1024  # 10MB


def compute_napiprojekt_hash(path: Path) -> str:
    """Compute the NapiProjekt hash for a video file.

    Returns the MD5 hex digest of the first 10MB of the file.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f'Cannot hash empty file: {path}')

    with open(path, 'rb') as f:
        data = f.read(NAPIPROJEKT_READ_SIZE)

    return hashlib.md5(data).hexdigest()
