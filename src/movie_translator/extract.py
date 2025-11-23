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
    """Find English subtitle track in MKV file.

    Prefers full dialogue tracks over signs-only tracks.
    For anime, this avoids extracting "Signs & Songs" when "Full Subtitles" exists.
    """
    tracks = track_info.get("tracks", [])
    english_tracks = []

    # Collect all English subtitle tracks
    for track in tracks:
        if track.get("type") == TRACK_TYPE_SUBTITLE:
            props = track.get("properties", {})
            lang = props.get("language", "")
            if lang in [LANGUAGE_ENGLISH, LANGUAGE_ENGLISH_SHORT]:
                english_tracks.append(track)

    if not english_tracks:
        return None

    # If only one track, return it
    if len(english_tracks) == 1:
        return english_tracks[0]

    # Multiple tracks: prefer full dialogue over signs-only
    # Signs-only tracks typically have "sign" or "song" in the name
    full_tracks = []
    signs_tracks = []

    for track in english_tracks:
        props = track.get("properties", {})
        track_name = props.get("track_name", "").lower()

        # Check if this is a signs/songs only track
        if "sign" in track_name or "song" in track_name:
            signs_tracks.append(track)
        else:
            full_tracks.append(track)

    # Prefer full dialogue tracks
    if full_tracks:
        logger.info(f"Found {len(english_tracks)} English tracks, selecting full dialogue track")
        return full_tracks[0]

    # If all tracks are signs-only, return the first one
    logger.warning(f"Found {len(english_tracks)} English tracks, all appear to be signs-only")
    return english_tracks[0]


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


def get_subtitle_extension(track: dict) -> str:
    """Determine the subtitle file extension based on codec.

    Args:
        track: Track information dictionary from mkvmerge

    Returns:
        File extension (.srt, .ass, or .ssa)
    """
    props = track.get("properties", {})
    codec = props.get("codec_id", "").lower()

    # Map MKV codec IDs to file extensions
    if "ass" in codec or "s_text/ass" in codec:
        return ".ass"
    elif "ssa" in codec or "s_text/ssa" in codec:
        return ".ssa"
    else:
        # Default to .srt for SubRip and unknown formats
        return ".srt"


def extract_subtitle_from_file(mkv_path: Path, force: bool = False) -> Path:
    """Extract English subtitle from a single MKV file.

    Args:
        mkv_path: Path to MKV file
        force: If True, overwrite existing subtitle file

    Returns:
        Path to extracted subtitle file (with correct extension)

    Raises:
        SubtitleNotFoundError: If no English subtitle track found
        MKVProcessingError: If extraction fails
    """
    if not mkv_path.exists():
        raise FileNotFoundError(f"MKV file not found: {mkv_path}")

    track_info = get_track_info(mkv_path)
    eng_sub_track = find_english_subtitle_track(track_info)

    if not eng_sub_track:
        raise SubtitleNotFoundError(
            f"No English subtitle track found in {mkv_path.name}"
        )

    # Determine correct file extension based on codec
    ext = get_subtitle_extension(eng_sub_track)
    output_sub = mkv_path.with_suffix("").with_name(f"{mkv_path.stem}_en{ext}")

    logger.info(f"Detected subtitle format: {ext}")

    # Skip if already extracted (unless force=True)
    if output_sub.exists() and not force:
        logger.info(f"Skipping {mkv_path.name} (already extracted)")
        return output_sub

    track_id = eng_sub_track["id"]
    extract_subtitle(mkv_path, track_id, output_sub)

    return output_sub


def main() -> None:
    """Main entry point for subtitle extraction."""
    parser = argparse.ArgumentParser(
        description="Extract English subtitles from a single MKV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mkv_file",
        type=Path,
        help="MKV file to extract subtitles from",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing _en.srt file if it exists",
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
        logger.info(f"Extracting subtitles from: {args.mkv_file.name}")
        output_srt = extract_subtitle_from_file(args.mkv_file, force=args.force)
        logger.info(f"✓ Extraction complete: {output_srt.name}")
    except (SubtitleNotFoundError, MKVProcessingError) as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
