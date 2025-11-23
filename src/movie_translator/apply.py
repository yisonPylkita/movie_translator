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


def merge_subtitle(mkv_path: Path, srt_path: Path, output_path: Path) -> None:
    """Merge Polish subtitle into MKV file."""
    logger.info(f"Merging subtitles into {mkv_path.name}...")
    cmd = [
        "mkvmerge",
        "-o",
        str(output_path),
        str(mkv_path),
        "--language",
        "0:pl",
        "--track-name",
        f"0:{POLISH_TRACK_NAME}",
        str(srt_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  → Created {output_path.name}")
    except subprocess.CalledProcessError as e:
        raise MKVProcessingError(
            f"Failed to merge subtitle into {mkv_path.name}. Is mkvmerge installed?"
        ) from e


def apply_subtitles_to_directory(directory: Path, backup: bool = False) -> None:
    """Apply Polish subtitles to all MKV files in directory.

    Args:
        directory: Directory containing MKV and *_pl.srt files
        backup: If True, create .bak backup of original MKV files
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    mkv_files = list(directory.glob(f"*{EXTENSION_MKV}"))
    if not mkv_files:
        logger.warning(f"No MKV files found in {directory}")
        return

    logger.info(f"Found {len(mkv_files)} MKV file(s)")
    applied_count = 0
    skipped_count = 0
    failed_count = 0

    for mkv_file in mkv_files:
        try:
            # Find matching Polish subtitle: movie.mkv -> movie_pl.srt
            polish_srt = mkv_file.with_suffix("").with_name(f"{mkv_file.stem}_pl.srt")

            if not polish_srt.exists():
                logger.warning(
                    f"Skipping {mkv_file.name}: No Polish subtitle found ({polish_srt.name})"
                )
                skipped_count += 1
                continue

            # Use temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = Path(temp_dir) / mkv_file.name

                # Merge subtitles
                merge_subtitle(mkv_file, polish_srt, temp_output)

                # Create backup if requested
                if backup:
                    backup_path = mkv_file.with_suffix(".mkv.bak")
                    logger.info(f"Creating backup: {backup_path.name}")
                    shutil.copy2(mkv_file, backup_path)

                # Replace original file
                logger.info(f"Replacing original: {mkv_file.name}")
                shutil.move(str(temp_output), str(mkv_file))
                logger.info(f"  ✓ Applied subtitles to {mkv_file.name}")
                applied_count += 1

        except MKVProcessingError as e:
            logger.error(f"Failed to process {mkv_file.name}: {e}")
            failed_count += 1
        except Exception as e:
            logger.error(
                f"Unexpected error processing {mkv_file.name}: {e}", exc_info=True
            )
            failed_count += 1

    logger.info(
        f"✓ Apply complete: {applied_count} applied, "
        f"{skipped_count} skipped, {failed_count} failed"
    )


def main() -> None:
    """Main entry point for applying subtitles."""
    parser = argparse.ArgumentParser(
        description="Apply translated Polish subtitles to MKV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing MKV files and *_pl.srt subtitles",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup of original MKV files before modifying",
    )

    args = parser.parse_args()
    setup_logging()

    if not args.directory.exists():
        logger.error(f"Directory not found: {args.directory}")
        sys.exit(1)

    if not args.directory.is_dir():
        logger.error(f"Not a directory: {args.directory}")
        sys.exit(1)

    try:
        logger.info(f"Applying subtitles in: {args.directory}")
        if args.backup:
            logger.info("Backup mode: ON (will create .bak files)")
        else:
            logger.warning("Backup mode: OFF (original files will be overwritten)")

        apply_subtitles_to_directory(args.directory, backup=args.backup)
    except KeyboardInterrupt:
        logger.warning("Apply interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Apply failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
