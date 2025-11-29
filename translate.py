#!/usr/bin/env python3
"""
Movie Subtitle Extractor - Step 1
Extract English dialogue subtitles from MKV files (skipping signs/songs)
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# ANSI colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def log_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def log_progress(message: str):
    print(f"{Colors.CYAN}{message}{Colors.NC}")

def check_dependencies():
    """Check required dependencies."""
    print("\n" + "=" * 50)
    print("  Dependency Check")
    print("=" * 50 + "\n")
    
    # Check Python
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        log_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    log_info(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    # Check mkvmerge
    if not shutil.which("mkvmerge"):
        log_error("mkvmerge not found. Please install mkvtoolnix")
        sys.exit(1)
    result = subprocess.run(["mkvmerge", "--version"], capture_output=True, text=True)
    log_info(f"mkvmerge: {result.stdout.split()[0]}")
    
    # Check mkvextract
    if not shutil.which("mkvextract"):
        log_error("mkvextract not found. Please install mkvtoolnix")
        sys.exit(1)
    
    print("\n" + Colors.GREEN + "All dependencies satisfied" + Colors.NC + "\n")

def get_track_info(mkv_path: Path) -> Dict[str, Any]:
    """Get track information from MKV file."""
    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(mkv_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        log_error(f"Failed to get track info from {mkv_path.name}: {e}")
        return {}

def find_english_subtitle_track(track_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find English subtitle track, preferring dialogue over signs/songs."""
    tracks = track_info.get("tracks", [])
    english_tracks = []
    
    # Collect all English subtitle tracks
    for track in tracks:
        if track.get("type") == "subtitles":
            props = track.get("properties", {})
            lang = props.get("language", "")
            if lang in ["eng", "en"]:
                english_tracks.append(track)
    
    if not english_tracks:
        return None
    
    # If only one track, return it
    if len(english_tracks) == 1:
        return english_tracks[0]
    
    # Multiple tracks: prefer dialogue over signs/songs
    # Look for track names that suggest dialogue
    dialogue_tracks = []
    signs_tracks = []
    
    for track in english_tracks:
        props = track.get("properties", {})
        track_name = props.get("track_name", "").lower()
        
        # Check if it's a signs/songs track
        if any(keyword in track_name for keyword in ["sign", "song", "title", "op", "ed"]):
            signs_tracks.append(track)
        else:
            dialogue_tracks.append(track)
    
    # Prefer dialogue tracks
    if dialogue_tracks:
        log_info(f"Found {len(english_tracks)} English tracks, selected dialogue track")
        return dialogue_tracks[0]
    
    # If no dialogue tracks, prefer non-forced tracks
    non_forced_tracks = []
    for track in english_tracks:
        props = track.get("properties", {})
        if not props.get("forced_track", False):
            non_forced_tracks.append(track)
    
    if non_forced_tracks:
        log_info(f"Found {len(english_tracks)} English tracks, selected non-forced track")
        return non_forced_tracks[0]
    
    # Last resort: return the first English track
    log_warning(f"Found {len(english_tracks)} English tracks, all appear to be signs/songs")
    return english_tracks[0]

def get_subtitle_extension(track: Dict[str, Any]) -> str:
    """Get subtitle file extension based on codec."""
    props = track.get("properties", {})
    codec = props.get("codec_id", "").lower()
    
    if "ass" in codec or "s_text/ass" in codec:
        return ".ass"
    elif "ssa" in codec or "s_text/ssa" in codec:
        return ".ssa"
    else:
        return ".srt"

def extract_subtitle(mkv_path: Path, track_id: int, output_path: Path) -> bool:
    """Extract subtitle track from MKV file."""
    log_info(f"Extracting subtitle track {track_id}...")
    
    cmd = ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{output_path}"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_info(f"Extraction successful: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to extract subtitle track {track_id}: {e}")
        if e.stderr:
            log_error(f"stderr: {e.stderr}")
        return False

def process_mkv_file(mkv_path: Path, output_dir: Path) -> bool:
    """Process a single MKV file to extract English subtitles."""
    log_progress(f"\n{'='*50}")
    log_progress(f"{Colors.BOLD}Processing: {mkv_path.name}{Colors.NC}")
    log_progress(f"{'='*50}")
    
    # Get track information
    track_info = get_track_info(mkv_path)
    if not track_info:
        log_error("Could not read track information")
        return False
    
    # Find English subtitle track
    eng_track = find_english_subtitle_track(track_info)
    if not eng_track:
        log_warning("No English subtitle track found")
        return False
    
    track_id = eng_track["id"]
    track_name = eng_track.get("properties", {}).get("track_name", "Unknown")
    log_info(f"Found English track: ID {track_id}, Name: '{track_name}'")
    
    # Get file extension
    subtitle_ext = get_subtitle_extension(eng_track)
    
    # Extract subtitle
    output_path = output_dir / f"{mkv_path.stem}_english{subtitle_ext}"
    
    if extract_subtitle(mkv_path, track_id, output_path):
        # Check if file was created and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            file_size = output_path.stat().st_size
            log_success(f"✅ Extracted English subtitles: {output_path.name} ({file_size:,} bytes)")
            return True
        else:
            log_error("Extracted file is empty")
            return False
    else:
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract English dialogue subtitles from MKV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Extract subtitles from all MKV files in directory
  %(prog)s /path/to/anime

  # Extract to specific output directory
  %(prog)s /path/to/anime --output /path/to/subtitles

  # Process only specific file
  %(prog)s /path/to/movie.mkv
        """
    )
    
    parser.add_argument(
        "input",
        help="MKV file or directory containing MKV files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for extracted subtitles (default: subtitles/ next to input)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        log_error(f"Input not found: {input_path}")
        sys.exit(1)
    
    input_path = input_path.resolve()
    
    # Determine output directory
    if args.output:
        output_dir = args.output.resolve()
    else:
        if input_path.is_file():
            output_dir = input_path.parent / "subtitles"
        else:
            output_dir = input_path / "subtitles"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check dependencies
    check_dependencies()
    
    # Print configuration
    print("\n" + "=" * 50)
    print("  English Subtitle Extractor")
    print("=" * 50 + "\n")
    log_info(f"Input: {input_path}")
    log_info(f"Output: {output_dir}")
    print()
    
    # Find MKV files
    if input_path.is_file():
        mkv_files = [input_path]
    else:
        mkv_files = sorted(input_path.glob("*.mkv"))
    
    if not mkv_files:
        log_warning("No MKV files found")
        sys.exit(0)
    
    log_info(f"Found {len(mkv_files)} MKV file(s)")
    
    # Process files
    successful = 0
    failed = 0
    
    for mkv_file in mkv_files:
        if process_mkv_file(mkv_file, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("  Extraction Complete")
    print("=" * 50)
    print(f"\nSummary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Total: {len(mkv_files)}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}All files processed successfully!{Colors.NC}")
        print(f"Subtitles extracted to: {output_dir}")
    else:
        print(f"\n{Colors.YELLOW}Some files failed to process.{Colors.NC}")

if __name__ == "__main__":
    main()