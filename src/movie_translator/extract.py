"""Extract English subtitles from MKV files."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from movie_translator.config import setup_logging
from movie_translator.constants import (
    EXTENSION_MKV,
    TRACK_TYPE_SUBTITLE,
    LANGUAGE_ENGLISH,
    LANGUAGE_ENGLISH_SHORT,
)
from movie_translator.exceptions import SubtitleNotFoundError, MKVProcessingError

logger = logging.getLogger(__name__)


def get_track_info(mkv_path: Path) -> Dict[str, Any]:
    """Get track information from MKV file."""
    cmd = ["mkvmerge", "-J", str(mkv_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise MKVProcessingError(
            f"Failed to get track info from {mkv_path.name}. Is mkvmerge installed?"
        ) from e
    except json.JSONDecodeError as e:
        raise MKVProcessingError(
            f"Failed to parse track info from {mkv_path.name}"
        ) from e


def find_english_subtitle_track(track_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find English subtitle track in MKV file."""
    tracks = track_info.get("tracks", [])
    for track in tracks:
        if track.get("type") == TRACK_TYPE_SUBTITLE:
            props = track.get("properties", {})
            lang = props.get("language", "")
            if lang in [LANGUAGE_ENGLISH, LANGUAGE_ENGLISH_SHORT]:
                return track
    return None


def extract_subtitle(mkv_path: Path, track_id: int, output_path: Path) -> None:
    """Extract subtitle track from MKV file."""
    logger.info(f"Extracting subtitle track {track_id} from {mkv_path.name}...")
    cmd = ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{output_path}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  → Saved to {output_path.name}")
    except subprocess.CalledProcessError as e:
        raise MKVProcessingError(
            f"Failed to extract subtitle track {track_id}. Is mkvextract installed?"
        ) from e


def extract_subtitles_from_directory(directory: Path) -> None:
    """Extract English subtitles from all MKV files in directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    mkv_files = list(directory.glob(f"*{EXTENSION_MKV}"))
    if not mkv_files:
        logger.warning(f"No MKV files found in {directory}")
        return

    logger.info(f"Found {len(mkv_files)} MKV file(s)")
    extracted_count = 0
    skipped_count = 0

    for mkv_file in mkv_files:
        try:
            # Generate output filename: movie.mkv -> movie_en.srt
            output_srt = mkv_file.with_suffix("").with_name(
                f"{mkv_file.stem}_en.srt"
            )

            # Skip if already extracted
            if output_srt.exists():
                logger.info(f"Skipping {mkv_file.name} (already extracted)")
                skipped_count += 1
                continue

            track_info = get_track_info(mkv_file)
            eng_sub_track = find_english_subtitle_track(track_info)

            if not eng_sub_track:
                logger.warning(
                    f"Skipping {mkv_file.name}: No English subtitle track found"
                )
                skipped_count += 1
                continue

            track_id = eng_sub_track["id"]
            extract_subtitle(mkv_file, track_id, output_srt)
            extracted_count += 1

        except (SubtitleNotFoundError, MKVProcessingError) as e:
            logger.error(f"Failed to process {mkv_file.name}: {e}")
            skipped_count += 1
        except Exception as e:
            logger.error(
                f"Unexpected error processing {mkv_file.name}: {e}", exc_info=True
            )
            skipped_count += 1

    logger.info(
        f"✓ Extraction complete: {extracted_count} extracted, {skipped_count} skipped"
    )


def main() -> None:
    """Main entry point for subtitle extraction."""
    parser = argparse.ArgumentParser(
        description="Extract English subtitles from MKV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing MKV files",
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
        logger.info(f"Extracting subtitles from: {args.directory}")
        extract_subtitles_from_directory(args.directory)
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
