import struct
from pathlib import Path

CHUNK_SIZE = 65536  # 64KB


def compute_oshash(path: Path) -> str:
    """Compute the OpenSubtitles hash for a video file.

    Algorithm: sum all 8-byte little-endian uint64 values from the first
    64KB and last 64KB of the file, add the file size. Return as 16-char
    lowercase hex. Overflow wraps at 2^64.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f'Cannot hash empty file: {path}')

    hash_val = file_size
    read_size = min(CHUNK_SIZE, file_size)

    with open(path, 'rb') as f:
        # First 64KB
        buf = f.read(read_size)
        hash_val = _sum_chunks(buf, hash_val)

        # Last 64KB
        f.seek(max(0, file_size - CHUNK_SIZE))
        buf = f.read(read_size)
        hash_val = _sum_chunks(buf, hash_val)

    return format(hash_val, '016x')


def _sum_chunks(buf: bytes, initial: int) -> int:
    """Sum all 8-byte little-endian chunks, wrapping at 2^64."""
    # Pad to multiple of 8
    remainder = len(buf) % 8
    if remainder:
        buf += b'\x00' * (8 - remainder)

    val = initial
    for (chunk,) in struct.iter_unpack('<Q', buf):
        val = (val + chunk) & 0xFFFFFFFFFFFFFFFF
    return val
