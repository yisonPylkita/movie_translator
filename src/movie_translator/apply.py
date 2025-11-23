"""Apply translated Polish subtitles to MKV files."""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from movie_translator.config import setup_logging
from movie_translator.constants import (
    EXTENSION_MKV,
    POLISH_TRACK_NAME,
)
from movie_translator.exceptions import MKVProcessingError

logger = logging.getLogger(__name__)


def merge_subtitle(
    mkv_path: Path, english_srt: Path, polish_srt: Path, output_path: Path
) -> None:
    """Merge English and Polish subtitles into MKV file, removing all other subtitle tracks.

    Args:
        mkv_path: Original MKV file
        english_srt: Extracted English subtitle file
        polish_srt: Translated Polish subtitle file
        output_path: Output MKV file path
    """
    logger.info(f"Merging subtitles into {mkv_path.name}...")
    logger.info(f"  → Keeping only English and Polish subtitle tracks")

    # Build mkvmerge command to:
    # 1. Copy video and audio from original MKV
    # 2. Remove ALL existing subtitle tracks (-S flag)
    # 3. Add English subtitle with proper metadata
    # 4. Add Polish subtitle with proper metadata
    #
    # Note: Track options must come BEFORE each input file
    cmd = [
        "mkvmerge",
        "-o",
        str(output_path),
        # Copy video and audio, but NO subtitles from original MKV
        "-S",  # Don't copy subtitle tracks
        str(mkv_path),
        # Add English subtitle track (options before the file)
        "--language", "0:eng",
        "--track-name", "0:English",
        "--default-track", "0:yes",
        str(english_srt),
        # Add Polish subtitle track (options before the file)
        "--language", "0:pol",
        "--track-name", f"0:{POLISH_TRACK_NAME}",
        "--default-track", "0:no",
        str(polish_srt),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"mkvmerge stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"mkvmerge stderr: {result.stderr}")
        logger.info(f"  → Created {output_path.name}")
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to merge subtitles into {mkv_path.name}."
        if e.stdout:
            error_msg += f"\nmkvmerge stdout: {e.stdout}"
        if e.stderr:
            error_msg += f"\nmkvmerge stderr: {e.stderr}"
        raise MKVProcessingError(error_msg) from e


def apply_subtitles_to_file(mkv_path: Path, backup: bool = False) -> None:
    """Apply English and Polish subtitles to a single MKV file.

    Args:
        mkv_path: Path to MKV file
        backup: If True, create .bak backup of original MKV file

    Raises:
        FileNotFoundError: If MKV or subtitle files not found
        MKVProcessingError: If merging fails
    """
    if not mkv_path.exists():
        raise FileNotFoundError(f"MKV file not found: {mkv_path}")

    # Find matching subtitle files: movie.mkv -> movie_en.srt, movie_pl.srt
    english_srt = mkv_path.with_suffix("").with_name(f"{mkv_path.stem}_en.srt")
    polish_srt = mkv_path.with_suffix("").with_name(f"{mkv_path.stem}_pl.srt")

    if not english_srt.exists():
        raise FileNotFoundError(
            f"English subtitle not found: {english_srt.name}"
        )

    if not polish_srt.exists():
        raise FileNotFoundError(
            f"Polish subtitle not found: {polish_srt.name}"
        )

    # Use temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / mkv_path.name

        # Merge both subtitle tracks (removes all others)
        merge_subtitle(mkv_path, english_srt, polish_srt, temp_output)

        # Create backup if requested
        if backup:
            backup_path = mkv_path.with_suffix(".mkv.bak")
            logger.info(f"Creating backup: {backup_path.name}")
            shutil.copy2(mkv_path, backup_path)

        # Replace original file
        logger.info(f"Replacing original: {mkv_path.name}")
        shutil.move(str(temp_output), str(mkv_path))
        logger.info(f"  ✓ Applied subtitles to {mkv_path.name}")
        logger.info(f"  ✓ Final tracks: English (default) + Polish")


def main() -> None:
    """Main entry point for applying subtitles."""
    parser = argparse.ArgumentParser(
        description="Apply translated Polish subtitles to a single MKV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mkv_file",
        type=Path,
        help="MKV file to apply subtitles to",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup of original MKV file before modifying",
    )

    args = parser.parse_args()
    setup_logging()

    if not args.mkv_file.exists():
        logger.error(f"MKV file not found: {args.mkv_file}")
        sys.exit(1)

    if not args.mkv_file.suffix.lower() == ".mkv":
        logger.error(f"Not an MKV file: {args.mkv_file}")
        sys.exit(1)

    try:
        logger.info(f"Applying subtitles to: {args.mkv_file.name}")
        if args.backup:
            logger.info("Backup mode: ON (will create .bak file)")
        else:
            logger.warning("Backup mode: OFF (original file will be overwritten)")

        apply_subtitles_to_file(args.mkv_file, backup=args.backup)
        logger.info(f"✓ Apply complete: {args.mkv_file.name}")
    except (FileNotFoundError, MKVProcessingError) as e:
        logger.error(f"Apply failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Apply interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
