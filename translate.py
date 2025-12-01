#!/usr/bin/env python3
"""
Movie Translator - Final Complete Pipeline
Extract English dialogue → AI translate to Polish → Create clean MKV
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ai_translator import SubtitleTranslator as AITranslator

console = Console()

def get_memory_info() -> str:
    """Get current memory usage information."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"Process: {memory_mb:.1f}MB"
    except Exception:
        return "Memory info unavailable"

def clear_memory():
    """Clear memory caches and force garbage collection."""
    gc.collect()

def replace_polish_chars(text: str) -> str:
    """Replace Polish characters with English equivalents."""
    polish_to_english = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }

    for polish, english in polish_to_english.items():
        text = text.replace(polish, english)

    return text

def log_info(message: str):
    console.print(f"[blue][INFO][/blue] {message}")

def log_success(message: str):
    console.print(f"[green][SUCCESS][/green] {message}")

def log_warning(message: str):
    console.print(f"[yellow][WARNING][/yellow] {message}")

def log_error(message: str):
    console.print(f"[red][ERROR][/red] {message}")

def check_dependencies():
    console.print(Panel.fit("[bold blue]Dependency Check[/bold blue]", border_style="blue"))

    table = Table(show_header=False, box=None)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        log_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    log_info(f"Python: {version.major}.{version.minor}.{version.micro}")

    if not shutil.which("mkvmerge"):
        log_error("mkvmerge not found. Please install mkvtoolnix")
        sys.exit(1)
    result = subprocess.run(["mkvmerge", "--version"], capture_output=True, text=True)
    log_info(f"mkvmerge: {result.stdout.split()[0]}")

    if not shutil.which("mkvextract"):
        log_error("mkvextract not found. Please install mkvtoolnix")
        sys.exit(1)

    try:
        import importlib.util

        required_packages = ["pysubs2", "torch", "transformers"]
        missing_packages = []

        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)

        if missing_packages:
            log_error(f"Missing Python packages: {', '.join(missing_packages)}")
            log_info("Install with: uv add pysubs2 torch transformers")
            sys.exit(1)

        log_info("Python packages: pysubs2, torch, transformers")
    except ImportError as e:
        log_error(f"Missing Python package: {e}")
        log_info("Install with: uv add pysubs2 torch transformers")
        sys.exit(1)

    console.print(Panel("[bold green]✅ All dependencies satisfied[/bold green]", border_style="green"))

def get_track_info(mkv_path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(mkv_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        log_error(f"Failed to get track info from {mkv_path.name}: {e}")
        return {}

def find_english_subtitle_track(track_info: dict[str, Any]) -> dict[str, Any] | None:
    tracks = track_info.get("tracks", [])
    english_tracks = []

    for track in tracks:
        if track.get("type") == "subtitles":
            props = track.get("properties", {})
            lang = props.get("language", "")
            if lang in ["eng", "en"]:
                english_tracks.append(track)

    if not english_tracks:
        return None

    if len(english_tracks) == 1:
        return english_tracks[0]

    dialogue_tracks = []
    signs_tracks = []

    for track in english_tracks:
        props = track.get("properties", {})
        track_name = props.get("track_name", "").lower()

        if any(keyword in track_name for keyword in ["sign", "song", "title", "op", "ed"]):
            signs_tracks.append(track)
        else:
            dialogue_tracks.append(track)

    if dialogue_tracks:
        log_info(f"Found {len(english_tracks)} English tracks, selected dialogue track")
        return dialogue_tracks[0]

    non_forced_tracks = []
    for track in english_tracks:
        props = track.get("properties", {})
        if not props.get("forced_track", False):
            non_forced_tracks.append(track)

    if non_forced_tracks:
        log_info(f"Found {len(english_tracks)} English tracks, selected non-forced track")
        return non_forced_tracks[0]

    log_warning(f"Found {len(english_tracks)} English tracks, all appear to be signs/songs")
    return english_tracks[0]

def get_subtitle_extension(track: dict[str, Any]) -> str:
    props = track.get("properties", {})
    codec = props.get("codec_id", "").lower()

    if "ass" in codec or "s_text/ass" in codec:
        return ".ass"
    elif "ssa" in codec or "s_text/ssa" in codec:
        return ".ssa"
    else:
        return ".srt"

def extract_subtitle(mkv_path: Path, track_id: int, output_path: Path) -> bool:
    log_info(f"Extracting subtitle track {track_id}...")

    cmd = ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{output_path}"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success(f"Extraction successful: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to extract subtitle track {track_id}: {e}")
        if e.stderr:
            log_error(f"stderr: {e.stderr}")
        return False

def extract_dialogue_lines(ass_file: Path) -> list[tuple[int, int, str]]:
    log_info(f"📖 Reading {ass_file.name}...")

    try:
        import pysubs2

        subs = pysubs2.load(str(ass_file))
        log_info(f"   - Loaded {len(subs)} total events")
    except Exception as e:
        log_error(f"Failed to load: {e}")
        return []

    dialogue_lines = []
    skipped_count = 0
    
    original_count = len(subs)
    seen = {}
    unique_subs = []
    
    for event in subs:
        clean_text = event.plaintext.strip()
        key = (event.start, event.end, clean_text)
        
        if key not in seen and clean_text and len(clean_text) >= 2:
            seen[key] = True
            unique_subs.append(event)

    deduped_count = len(unique_subs)
    if deduped_count < original_count:
        log_info(f"   - Deduplicated: {original_count} → {deduped_count} entries (removed {original_count - deduped_count} duplicate effect layers)")

    for event in unique_subs:
        if not event.text or event.text.strip() == "":
            skipped_count += 1
            continue

        style = getattr(event, "style", "Default")
        style_lower = style.lower()

        if any(keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]):
            skipped_count += 1
            continue

        clean_text = event.plaintext.strip()

        if not clean_text:
            skipped_count += 1
            continue

        if len(clean_text) < 2:
            skipped_count += 1
            continue

        dialogue_lines.append((event.start, event.end, clean_text))

    log_info(f"   - Extracted {len(dialogue_lines)} dialogue lines")
    log_info(f"   - Skipped {skipped_count} non-dialogue events")

    del subs
    clear_memory()
    log_info(f"   - After cleanup")

    return dialogue_lines

def create_clean_english_ass(
    original_ass: Path,
    dialogue_lines: list[tuple[int, int, str]],
    output_english_ass: Path,
):
    log_info(f"🔨 Creating clean English ASS: {output_english_ass.name}")

    try:
        import pysubs2

        original_subs = pysubs2.load(str(original_ass))
        log_info(f"   - Loaded original with {len(original_subs)} events")

        clean_english_subs = pysubs2.SSAFile()
        clean_english_subs.info = original_subs.info.copy()
        clean_english_subs.styles = original_subs.styles.copy()

        dialogue_index = 0

        for event in original_subs:
            style = getattr(event, "style", "Default")
            style_lower = style.lower()

            if any(keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]):
                continue

            if event.text and event.text.strip():
                clean_text = event.plaintext.strip()

                if clean_text and len(clean_text) >= 2:
                    if dialogue_index < len(dialogue_lines):
                        start, end, original_text = dialogue_lines[dialogue_index]

                        new_event = pysubs2.SSAEvent(
                            start=event.start,
                            end=event.end,
                            style=event.style,
                            text=original_text,
                        )
                        clean_english_subs.append(new_event)
                        dialogue_index += 1

        clean_english_subs.save(str(output_english_ass))
        log_success(f"   - Saved {len(clean_english_subs)} dialogue events")
        log_info("   - Removed all non-dialogue events")

    except Exception as e:
        log_error(f"Failed to create clean English ASS: {e}")


def create_polish_ass(
    original_ass: Path,
    translated_dialogue: list[tuple[int, int, str]],
    output_polish_ass: Path,
    font_mode: str = "replace",
):
    """Create Polish ASS file with translated dialogue."""
    log_info(f"🔤 Creating Polish subtitles")

    try:
        import pysubs2

        original_subs = pysubs2.load(str(original_ass))
        log_info(f"   - Loaded {len(original_subs)} original events")

        polish_subs = pysubs2.SSAFile()
        polish_subs.info = original_subs.info.copy()
        polish_subs.styles = original_subs.styles.copy()
        dialogue_index = 0

        for event in original_subs:
            style_lower = getattr(event, "style", "Default").lower()
            if any(keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]):
                continue

            if event.text and event.text.strip():
                clean_text = event.plaintext.strip()

                if clean_text and len(clean_text) >= 2:
                    if dialogue_index < len(translated_dialogue):
                        _, _, translated_text = translated_dialogue[dialogue_index]

                        if font_mode == "replace":
                            translated_text = replace_polish_chars(translated_text)

                        new_event = pysubs2.SSAEvent(
                            start=event.start,
                            end=event.end,
                            style=event.style,
                            text=translated_text,
                        )
                        polish_subs.append(new_event)
                        dialogue_index += 1

        polish_subs.save(str(output_polish_ass))
        log_success(f"   - Saved {len(polish_subs)} translated events")

        del original_subs
        del polish_subs
        clear_memory()
        log_info(f"   - After cleanup")

    except Exception as e:
        log_error(f"Failed to create Polish ASS: {e}")


def create_clean_mkv(
    original_mkv: Path, english_ass: Path, polish_ass: Path, output_mkv: Path
):
    """Create clean MKV with only video/audio + English dialogue + Polish dialogue."""
    log_info(f"🎬 Creating clean MKV: {output_mkv.name}")
    log_info("   - Adding: English dialogue + Polish dialogue only")

    cmd = [
        "mkvmerge",
        "-o",
        str(output_mkv),
        "--no-subtitles",
        str(original_mkv),
        "--language",
        "0:eng",
        "--track-name",
        "0:English Dialogue",
        str(english_ass),
        "--language",
        "0:pol",
        "--track-name",
        "0:Polish (AI)",
        str(polish_ass),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_success("   - Clean MKV merge successful")

        if output_mkv.stat().st_size == 0:
            size_mb = output_mkv.stat().st_size / 1024 / 1024
            log_info(f"   - Output size: {size_mb:.1f} MB")

        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to merge: {e}")
        if e.stderr:
            log_error(f"   stderr: {e.stderr}")
        return False


def verify_result(output_mkv: Path):
    """Verify the clean MKV has only the desired tracks."""
    log_info(f"🔍 Verifying result: {output_mkv.name}")

    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(output_mkv)],
            capture_output=True,
            text=True,
            check=True,
        )

        track_info = json.loads(result.stdout)
        tracks = track_info.get("tracks", [])

        subtitle_tracks = []
        for track in tracks:
            if track.get("type") == "subtitles":
                props = track.get("properties", {})
                subtitle_tracks.append(
                    {
                        "id": track.get("id"),
                        "language": props.get("language", "unknown"),
                        "name": props.get("track_name", "unnamed"),
                    }
                )

        log_info(f"   - Found {len(subtitle_tracks)} subtitle tracks:")
        for track in subtitle_tracks:
            log_info(
                f"     * Track {track['id']}: {track['name']} ({track['language']})"
            )

        if len(subtitle_tracks) == 2:
            english_found = any(t["language"] == "eng" for t in subtitle_tracks)
            polish_found = any(t["language"] == "pol" for t in subtitle_tracks)

            if english_found and polish_found:
                log_success(
                    "   ✅ Perfect! Found English dialogue + Polish translation only"
                )
                return True
            else:
                log_error("   ❌ Missing required languages")
                return False
        else:
            log_error(f"   ❌ Expected 2 subtitle tracks, found {len(subtitle_tracks)}")
            return False

    except Exception as e:
        log_error(f"Failed to verify: {e}")
        return False


def translate_dialogue_lines(
    dialogue_lines: list[tuple[int, int, str]], device: str, batch_size: int
) -> list[tuple[int, int, str]]:
    """Translate dialogue lines using AI with Rich progress bar."""
    log_info(f"🤖 Step 3: AI translating to Polish...")

    translator = AITranslator(device=device, batch_size=batch_size)
    log_info(f"   - AI Translator initialized")

    if not translator.load_model():
        log_error("❌ Failed to load translation model")
        return []

    log_info(f"   - Model loaded")

    texts = [text for _, _, text in dialogue_lines]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[rate]}"),
        console=console,
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]Translating {len(texts)} texts...[/cyan]",
            total=total_batches,
            rate="0.0 lines/s"
        )
        
        def progress_callback(batch_num, total_batches, rate, error=None):
            if error:
                progress.update(task, advance=1, rate=f"❌ {error}")
            else:
                progress.update(task, advance=1, rate=f"{rate:.1f} lines/s")
        
        # Translate with progress callback
        translated_texts = translator.translate_texts(texts, progress_callback=progress_callback)
    
    log_info(f"   - Translation complete")

    # Cleanup
    translator.cleanup()
    log_info(f"   - Translator cleaned up")

    # Force garbage collection
    clear_memory()
    log_info(f"   - Final cleanup")

    # Reconstruct with timing
    translated_lines = []
    for (start, end, _), translated_text in zip(dialogue_lines, translated_texts):
        translated_lines.append((start, end, translated_text))

    return translated_lines


def process_mkv_file(
    mkv_path: Path, output_dir: Path, device: str, batch_size: int
) -> bool:
    """Process a single MKV file through the complete pipeline."""
    console.print(Panel(f"[bold blue]Processing: {mkv_path.name}[/bold blue]", border_style="blue"))

    # Create temp directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Step 1: Extract English subtitles
    log_info("📖 Step 1: Extracting English subtitles...")
    track_info = get_track_info(mkv_path)
    if not track_info:
        log_error("Could not read track information")
        return False

    eng_track = find_english_subtitle_track(track_info)
    if not eng_track:
        log_warning("No English subtitle track found")
        return False

    track_id = eng_track["id"]
    track_name = eng_track.get("properties", {}).get("track_name", "Unknown")
    log_info(f"Found English track: ID {track_id}, Name: '{track_name}'")

    subtitle_ext = get_subtitle_extension(eng_track)
    extracted_ass = temp_dir / f"{mkv_path.stem}_extracted{subtitle_ext}"

    if not extract_subtitle(mkv_path, track_id, extracted_ass):
        return False

    # Step 2: Extract dialogue lines
    log_info("🔍 Step 2: Extracting dialogue lines...")
    dialogue_lines = extract_dialogue_lines(extracted_ass)
    if not dialogue_lines:
        log_error("No dialogue lines found")
        return False

    # Step 3: AI translate
    log_info("🤖 Step 3: AI translating to Polish...")
    try:
        translated_dialogue = translate_dialogue_lines(dialogue_lines, device, batch_size)
        if not translated_dialogue:
            log_error("AI translation failed")
            return False
        log_success("   ✅ AI translation complete!")
    except Exception as e:
        log_error(f"AI translation failed: {e}")
        return False

    # Step 4: Create clean ASS files
    log_info("🔨 Step 4: Creating clean subtitle files...")
    clean_english_ass = temp_dir / f"{mkv_path.stem}_english_clean.ass"
    polish_ass = temp_dir / f"{mkv_path.stem}_polish.ass"

    create_clean_english_ass(extracted_ass, dialogue_lines, clean_english_ass)
    create_polish_ass(extracted_ass, translated_dialogue, polish_ass)

    # Step 5: Create clean MKV
    log_info("🎬 Step 5: Creating final clean MKV...")
    output_mkv = output_dir / f"{mkv_path.stem}_clean.mkv"

    if not create_clean_mkv(mkv_path, clean_english_ass, polish_ass, output_mkv):
        return False

    # Step 6: Verify result
    if not verify_result(output_mkv):
        return False

    # Clean up temp files
    shutil.rmtree(temp_dir)
    log_info("🧹 Cleaned up temporary files")

    log_success(f"🎉 SUCCESS! Clean MKV created: {output_mkv.name}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Movie Translator - Extract English dialogue and translate to Polish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process all MKV files in directory
  %(prog)s /path/to/anime

  # Process with custom settings
  %(prog)s /path/to/movies --device mps --batch-size 32

  # Process single file
  %(prog)s /path/to/movie.mkv --output /path/to/output
        """,
    )

    parser.add_argument("input", help="MKV file or directory containing MKV files")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (default: translated/ next to input)",
    )

    parser.add_argument(
        "--device",
        choices=["mps", "cpu"],
        default="mps",  # MacBook optimized
        help="Translation device (default: mps)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Translation batch size (default: 16)",
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
            output_dir = input_path.parent / "translated"
        else:
            output_dir = input_path / "translated"

    output_dir.mkdir(exist_ok=True)

    check_dependencies()

    config_table = Table(title="[bold blue]Movie Translator - Final Pipeline[/bold blue]", show_header=False, box=None)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Input", str(input_path))
    config_table.add_row("Output", str(output_dir))
    config_table.add_row("Device", args.device)
    config_table.add_row("Batch Size", str(args.batch_size))
    console.print(Panel(config_table, border_style="blue"))
    console.print()

    # Find MKV files
    if input_path.is_file():
        mkv_files = [input_path]
    else:
        mkv_files = sorted(input_path.glob("*.mkv"))

    if not mkv_files:
        log_warning("No MKV files found")
        sys.exit(0)

    log_info(f"Found {len(mkv_files)} MKV file(s)")

    # Process files with fancy progress
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:

        task = progress.add_task(
            f"[cyan]Processing {len(mkv_files)} MKV files...[/cyan]",
            total=len(mkv_files)
        )

        for mkv_file in mkv_files:
            progress.update(task, description=f"[cyan]Processing {mkv_file.name}...[/cyan]")

            if process_mkv_file(mkv_file, output_dir, args.device, args.batch_size):
                successful += 1
                progress.update(task, advance=1)
            else:
                failed += 1
                progress.update(task, advance=1)

    # Final summary with fancy panel
    summary_table = Table(title="[bold green]Translation Complete[/bold green]", show_header=False, box=None)
    summary_table.add_column("Status", style="white")
    summary_table.add_column("Count", style="white")
    summary_table.add_row("✅ Successful", str(successful), style="green")
    summary_table.add_row("❌ Failed", str(failed), style="red")
    summary_table.add_row("📁 Total", str(len(mkv_files)), style="cyan")

    console.print(Panel(summary_table, border_style="green"))

    if failed == 0:
        console.print(Panel(
            "[bold green]🎉 All files processed successfully![/bold green]\n"
            "🎬 Clean MKVs with English dialogue + Polish translation created\n"
            f"📁 Output directory: {output_dir}",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]⚠️  Some files failed to process.[/bold yellow]",
            border_style="yellow"
        ))


if __name__ == "__main__":
    main()
