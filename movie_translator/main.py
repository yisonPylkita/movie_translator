import argparse
import logging
import shutil
import sys
from pathlib import Path

from .logging import console, logger, set_verbose
from .pipeline import TranslationPipeline
from .progress import ProgressTracker
from .subtitles import SubtitleExtractor


def check_dependencies() -> bool:
    """Check all required dependencies. Returns True if all satisfied."""
    import importlib.util

    from .ffmpeg import get_ffmpeg_version

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print(f'[red]❌ Python 3.10+ required, found {version.major}.{version.minor}[/red]')
        return False

    try:
        get_ffmpeg_version()
    except Exception:
        console.print('[red]❌ FFmpeg not available. Run ./setup.sh first.[/red]')
        return False

    required_packages = ['pysubs2', 'torch', 'transformers']
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            console.print(f'[red]❌ Missing package: {package}. Run ./setup.sh first.[/red]')
            return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original video'
    )
    parser.add_argument('input_dir', help='Directory containing video files (MKV, MP4)')
    parser.add_argument(
        '--device',
        choices=['cpu', 'mps'],
        default='mps' if sys.platform == 'darwin' else 'cpu',
        help='Device to use for AI translation (default: mps on macOS, cpu on others)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for AI translation (default: 16)',
    )
    parser.add_argument(
        '--model',
        choices=['allegro'],
        default='allegro',
        help='Translation model to use (default: allegro)',
    )
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Disable online subtitle fetching (use only local extraction/OCR)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process files but do not replace originals (keeps output in temp directory)',
    )
    parser.add_argument(
        '--keep-artifacts',
        action='store_true',
        help='Preserve working directories after processing (default: clean up on success)',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging output',
    )
    return parser.parse_args()


def show_summary(results: list[tuple[str, str]], dry_run: bool = False) -> None:
    """Show a brief summary of processing results."""
    successful = sum(1 for _, status in results if status == 'success')
    failed = sum(1 for _, status in results if status == 'failed')
    skipped = sum(1 for _, status in results if status == 'skipped')

    parts = []
    if successful > 0:
        parts.append(f'[green]✓ {successful} translated[/green]')
    if skipped > 0:
        parts.append(f'[blue]⏭ {skipped} skipped[/blue]')
    if failed > 0:
        parts.append(f'[red]✗ {failed} failed[/red]')

    console.print(' | '.join(parts))

    if dry_run and successful > 0:
        console.print('[yellow]Dry run - originals not modified[/yellow]')


VIDEO_EXTENSIONS = ('*.mkv', '*.mp4')


def create_working_dirs(video_path: Path, input_dir: Path) -> Path:
    """Create per-anime/per-episode working directory structure.

    Structure: input_dir/.translate_temp/<anime_folder>/<video_stem>/
    With subdirs: candidates/, reference/

    Args:
        video_path: Path to the video file.
        input_dir: Root input directory.

    Returns:
        Path to the per-episode working directory.
    """
    temp_root = input_dir / '.translate_temp'
    anime_name = video_path.parent.name
    episode_name = video_path.stem

    work_dir = temp_root / anime_name / episode_name
    (work_dir / 'candidates').mkdir(parents=True, exist_ok=True)
    (work_dir / 'reference').mkdir(parents=True, exist_ok=True)

    return work_dir


def find_video_files(input_dir: Path) -> list[Path]:
    """Find all video files in input directory (flat or one level deep)."""
    video_files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(ext))
    if not video_files:
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('.'):
                for ext in VIDEO_EXTENSIONS:
                    video_files.extend(subdir.glob(ext))
    video_files.sort()
    return video_files


def find_video_files_with_temp_dirs(input_dir: Path) -> list[tuple[Path, Path]]:
    video_files_direct: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        video_files_direct.extend(input_dir.glob(ext))
    video_files_direct.sort()

    if video_files_direct:
        temp_dir = input_dir / '.translate_temp'
        temp_dir.mkdir(exist_ok=True)
        return [(f, temp_dir) for f in video_files_direct]

    results: list[tuple[Path, Path]] = []
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            video_files_in_subdir: list[Path] = []
            for ext in VIDEO_EXTENSIONS:
                video_files_in_subdir.extend(subdir.glob(ext))
            video_files_in_subdir.sort()

            if video_files_in_subdir:
                temp_dir = subdir / '.translate_temp'
                temp_dir.mkdir(exist_ok=True)
                results.extend((f, temp_dir) for f in video_files_in_subdir)

    return results


def main():
    args = parse_args()

    # Enable verbose logging if requested
    set_verbose(args.verbose)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f'[red]❌ Directory not found: {input_dir}[/red]')
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    video_files = find_video_files(input_dir)

    if not video_files:
        console.print(f'[red]❌ No video files found in {input_dir}[/red]')
        sys.exit(1)

    total_files = len(video_files)

    if args.dry_run:
        console.print('[yellow]Dry run mode - originals will not be modified[/yellow]')

    # Suppress noisy library warnings
    logging.getLogger('transformers').setLevel(logging.ERROR)

    extractor = SubtitleExtractor()

    with ProgressTracker(total_files, console=console) as tracker:
        pipeline = TranslationPipeline(
            device=args.device,
            batch_size=args.batch_size,
            model=args.model,
            enable_fetch=not args.no_fetch,
            tracker=tracker,
        )

        for video_path in video_files:
            relative_name = (
                f'{video_path.parent.name}/{video_path.name}'
                if video_path.parent != input_dir
                else video_path.name
            )

            tracker.start_file(relative_name)
            work_dir = create_working_dirs(video_path, input_dir)
            success = False

            try:
                if extractor.has_polish_subtitles(video_path):
                    tracker.complete_file('skipped')
                    success = True
                elif pipeline.process_video_file(video_path, work_dir, dry_run=args.dry_run):
                    tracker.complete_file('success')
                    success = True
                else:
                    tracker.complete_file('failed')
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                tracker.complete_file('failed')

            # Clean up working directory on success (unless --keep-artifacts)
            if success and not args.keep_artifacts and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    # Remove empty parent dirs up to .translate_temp
                    parent = work_dir.parent
                    while parent.name != '.translate_temp' and parent != input_dir:
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                    # Remove .translate_temp itself if empty
                    temp_root = input_dir / '.translate_temp'
                    if temp_root.exists() and not any(temp_root.iterdir()):
                        temp_root.rmdir()
                except OSError as e:
                    logger.debug(f'Failed to clean up {work_dir}: {e}')


if __name__ == '__main__':
    main()
