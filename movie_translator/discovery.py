# movie_translator/discovery.py
"""Recursive video file discovery and working directory creation."""

from pathlib import Path

VIDEO_EXTENSIONS = {'.mkv', '.mp4'}


def find_videos(input_path: Path) -> list[Path]:
    """Find all video files recursively from any input.

    - If input_path is a file: return [input_path] if it's a video, else []
    - If directory: recursively find all .mkv/.mp4 files, sorted
    - Skips hidden directories (starting with '.')
    - Returns [] for nonexistent paths
    """
    if not input_path.exists():
        return []

    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [input_path]
        return []

    videos: list[Path] = []
    for path in sorted(input_path.rglob('*')):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            # Skip files inside hidden directories
            if any(part.startswith('.') for part in path.relative_to(input_path).parts):
                continue
            videos.append(path)
    return videos


def create_work_dir(video_path: Path, root_input: Path) -> Path:
    """Create temp working directory preserving relative structure.

    For video at ~/Anime/Show/S1/ep01.mkv with root ~/Anime:
    returns ~/Anime/.translate_temp/Show/S1/ep01/
    """
    try:
        relative = video_path.parent.relative_to(root_input)
    except ValueError:
        relative = Path()

    temp_root = root_input / '.translate_temp'
    work_dir = temp_root / relative / video_path.stem

    (work_dir / 'candidates').mkdir(parents=True, exist_ok=True)
    (work_dir / 'reference').mkdir(parents=True, exist_ok=True)

    return work_dir
