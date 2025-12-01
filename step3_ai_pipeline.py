#!/usr/bin/env python3
"""
Step 3: Complete Pipeline with Real AI Translation
Extract → Clean → AI Translate → Rebuild ASS → Create MKV
"""

import subprocess
import sys
from pathlib import Path

try:
    import pysubs2
except ImportError:
    print("❌ pysubs2 not found. Install with: uv add pysubs2")
    sys.exit(1)

# Import our AI translator
from ai_translator import SubtitleTranslator


def extract_dialogue_lines(ass_file: Path) -> list[tuple[int, int, str]]:
    """Extract clean dialogue lines from ASS file."""
    print(f"📖 Reading {ass_file.name}...")

    try:
        subs = pysubs2.load(str(ass_file))
        print(f"   - Loaded {len(subs)} total events")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return []

    dialogue_lines = []
    skipped_count = 0

    for event in subs:
        # Skip empty events
        if not event.text or event.text.strip() == "":
            skipped_count += 1
            continue

        # Skip signs/songs based on style name
        style = getattr(event, "style", "Default")
        style_lower = style.lower()

        if any(
            keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]
        ):
            skipped_count += 1
            continue

        # Get clean text (remove ASS formatting)
        clean_text = event.plaintext.strip()

        # Skip empty text after cleaning
        if not clean_text:
            skipped_count += 1
            continue

        # Skip very short text (likely sound effects)
        if len(clean_text) < 2:
            skipped_count += 1
            continue

        # Add to dialogue list
        dialogue_lines.append((event.start, event.end, clean_text))

    print(f"   - Extracted {len(dialogue_lines)} dialogue lines")
    print(f"   - Skipped {skipped_count} non-dialogue events")

    return dialogue_lines


def ai_translate_dialogue(
    dialogue_lines: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Translate dialogue lines using AI."""
    print(f"🤖 AI Translating {len(dialogue_lines)} dialogue lines...")

    # Initialize AI translator
    translator = SubtitleTranslator(
        model_name="allegro/BiDi-eng-pol",
        device="auto",  # Will use MPS for Apple Silicon
        batch_size=16,  # Optimal batch size for MacBook
    )

    # Load the model
    if not translator.load_model():
        raise RuntimeError("Failed to load AI translation model")

    # Translate dialogue lines
    translated_lines = translator.translate_dialogue_lines(dialogue_lines)

    print("   ✅ AI translation complete!")
    return translated_lines


def rebuild_ass_file(
    original_ass: Path,
    translated_dialogue: list[tuple[int, int, str]],
    output_ass: Path,
):
    """Rebuild ASS file with translated dialogue."""
    print(f"🔨 Rebuilding ASS file: {output_ass.name}")

    # Load original ASS file to preserve styles and non-dialogue events
    original_subs = pysubs2.load(str(original_ass))
    print(f"   - Loaded original with {len(original_subs)} events")

    # Create new subtitle file
    new_subs = pysubs2.SSAFile()
    new_subs.info = original_subs.info.copy()
    new_subs.styles = original_subs.styles.copy()

    # Add all non-dialogue events unchanged
    dialogue_index = 0
    preserved_count = 0
    translated_count = 0

    for event in original_subs:
        # Check if this is a dialogue event
        style = getattr(event, "style", "Default")
        style_lower = style.lower()

        # Skip signs/songs - keep them as-is
        if any(
            keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]
        ):
            new_subs.append(event)
            preserved_count += 1
            continue

        # Check if this is dialogue with text
        if event.text and event.text.strip():
            clean_text = event.plaintext.strip()
            if clean_text and len(clean_text) >= 2:
                # This is dialogue - replace with translated version
                if dialogue_index < len(translated_dialogue):
                    start, end, translated_text = translated_dialogue[dialogue_index]

                    # Create new event with translated text
                    new_event = pysubs2.SSAEvent(
                        start=event.start,
                        end=event.end,
                        style=event.style,
                        text=translated_text,
                    )
                    new_subs.append(new_event)
                    dialogue_index += 1
                    translated_count += 1
                else:
                    # No translation available, keep original
                    new_subs.append(event)
                    preserved_count += 1
            else:
                # Empty or too short, keep as-is
                new_subs.append(event)
                preserved_count += 1
        else:
            # Empty event, keep as-is
            new_subs.append(event)
            preserved_count += 1

    # Save the rebuilt ASS file
    new_subs.save(str(output_ass))
    print(f"   - Saved {len(new_subs)} total events")
    print(f"   - Translated: {translated_count} dialogue events")
    print(f"   - Preserved: {preserved_count} non-dialogue events")


def merge_subtitle_to_mkv(mkv_path: Path, ass_path: Path, output_mkv: Path):
    """Merge translated ASS subtitle into MKV file."""
    print(f"🎬 Merging subtitle into MKV: {output_mkv.name}")

    cmd = [
        "mkvmerge",
        "-o",
        str(output_mkv),
        str(mkv_path),
        "--language",
        "0:pol",
        "--track-name",
        "0:Polish (AI)",
        str(ass_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   - MKV merge successful")

        # Check file size
        if output_mkv.exists():
            size_mb = output_mkv.stat().st_size / 1024 / 1024
            print(f"   - Output size: {size_mb:.1f} MB")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to merge: {e}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def verify_result(output_mkv: Path):
    """Verify the final MKV has Polish subtitles."""
    print(f"🔍 Verifying result: {output_mkv.name}")

    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(output_mkv)],
            capture_output=True,
            text=True,
            check=True,
        )

        import json

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

        print(f"   - Found {len(subtitle_tracks)} subtitle tracks:")
        for track in subtitle_tracks:
            print(f"     * Track {track['id']}: {track['name']} ({track['language']})")

        # Check for Polish track
        polish_tracks = [t for t in subtitle_tracks if t["language"] == "pol"]
        if polish_tracks:
            print("   ✅ Found Polish subtitle track!")
            return True
        else:
            print("   ❌ No Polish subtitle track found")
            return False

    except Exception as e:
        print(f"❌ Failed to verify: {e}")
        return False


def main():
    """Complete pipeline with real AI translation."""
    # File paths
    original_mkv = Path("/Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01.mkv")
    extracted_ass = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_english.ass"
    )
    translated_ass = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_ai_translated.ass"
    )
    output_mkv = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_with_ai_polish.mkv"
    )

    if not original_mkv.exists():
        print(f"❌ Original MKV not found: {original_mkv}")
        return

    if not extracted_ass.exists():
        print(f"❌ Extracted ASS not found: {extracted_ass}")
        print("Please run Step 1 first to extract subtitles")
        return

    print("🎬 Step 3: Complete Pipeline with Real AI Translation")
    print("=" * 70)

    # Step 1: Extract dialogue
    dialogue_lines = extract_dialogue_lines(extracted_ass)
    if not dialogue_lines:
        print("❌ No dialogue lines found")
        return

    # Step 2: AI translate
    try:
        translated_dialogue = ai_translate_dialogue(dialogue_lines)
    except Exception as e:
        print(f"❌ AI translation failed: {e}")
        return

    # Step 3: Rebuild ASS file
    rebuild_ass_file(extracted_ass, translated_dialogue, translated_ass)

    # Step 4: Merge into MKV
    if merge_subtitle_to_mkv(original_mkv, translated_ass, output_mkv):
        # Step 5: Verify result
        if verify_result(output_mkv):
            print("\n🎉 COMPLETE AI PIPELINE SUCCESS!")
            print(f"📁 Output MKV: {output_mkv}")
            print("✅ Polish subtitles added with AI translation")
            print("🤖 Model: allegro/BiDi-eng-pol")
            print("⚡ Device: MPS (Apple Silicon)")
        else:
            print("\n❌ Pipeline failed at verification step")
    else:
        print("\n❌ Pipeline failed at MKV merge step")


if __name__ == "__main__":
    main()
