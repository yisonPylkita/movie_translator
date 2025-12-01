#!/usr/bin/env python3
"""
Clean Pipeline: MKV with only English Dialogue + Polish Translation
Extract → Clean → AI Translate → Create Clean MKV (no signs/songs)
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


def create_clean_english_ass(
    original_ass: Path,
    dialogue_lines: list[tuple[int, int, str]],
    output_english_ass: Path,
):
    """Create clean English ASS with only dialogue (no signs/songs)."""
    print(f"🔨 Creating clean English ASS: {output_english_ass.name}")

    # Load original ASS file to get styles
    original_subs = pysubs2.load(str(original_ass))
    print(f"   - Loaded original with {len(original_subs)} events")

    # Create new subtitle file with only dialogue
    clean_english_subs = pysubs2.SSAFile()
    clean_english_subs.info = original_subs.info.copy()
    clean_english_subs.styles = original_subs.styles.copy()

    # Add only dialogue events
    dialogue_index = 0
    preserved_count = 0

    for event in original_subs:
        # Check if this is a dialogue event
        style = getattr(event, "style", "Default")
        style_lower = style.lower()

        # Skip signs/songs
        if any(
            keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]
        ):
            continue

        # Check if this is dialogue with text
        if event.text and event.text.strip():
            clean_text = event.plaintext.strip()
            if clean_text and len(clean_text) >= 2:
                # This is dialogue - use original text
                if dialogue_index < len(dialogue_lines):
                    start, end, original_text = dialogue_lines[dialogue_index]

                    # Create new event with original text
                    new_event = pysubs2.SSAEvent(
                        start=event.start,
                        end=event.end,
                        style=event.style,
                        text=original_text,
                    )
                    clean_english_subs.append(new_event)
                    dialogue_index += 1
                    preserved_count += 1
        # Skip everything else (signs, songs, empty events)

    # Save the clean English ASS file
    clean_english_subs.save(str(output_english_ass))
    print(f"   - Saved {len(clean_english_subs)} dialogue events")
    print("   - Removed all non-dialogue events")


def create_polish_ass(
    original_ass: Path,
    translated_dialogue: list[tuple[int, int, str]],
    output_polish_ass: Path,
):
    """Create Polish ASS with translated dialogue."""
    print(f"🔨 Creating Polish ASS: {output_polish_ass.name}")

    # Load original ASS file to get styles
    original_subs = pysubs2.load(str(original_ass))
    print(f"   - Loaded original with {len(original_subs)} events")

    # Create new subtitle file with only translated dialogue
    polish_subs = pysubs2.SSAFile()
    polish_subs.info = original_subs.info.copy()
    polish_subs.styles = original_subs.styles.copy()

    # Add only translated dialogue events
    dialogue_index = 0
    translated_count = 0

    for event in original_subs:
        # Check if this is a dialogue event
        style = getattr(event, "style", "Default")
        style_lower = style.lower()

        # Skip signs/songs
        if any(
            keyword in style_lower for keyword in ["sign", "song", "title", "op", "ed"]
        ):
            continue

        # Check if this is dialogue with text
        if event.text and event.text.strip():
            clean_text = event.plaintext.strip()
            if clean_text and len(clean_text) >= 2:
                # This is dialogue - use translated text
                if dialogue_index < len(translated_dialogue):
                    start, end, translated_text = translated_dialogue[dialogue_index]

                    # Create new event with translated text
                    new_event = pysubs2.SSAEvent(
                        start=event.start,
                        end=event.end,
                        style=event.style,
                        text=translated_text,
                    )
                    polish_subs.append(new_event)
                    dialogue_index += 1
                    translated_count += 1
        # Skip everything else (signs, songs, empty events)

    # Save the Polish ASS file
    polish_subs.save(str(output_polish_ass))
    print(f"   - Saved {len(polish_subs)} translated dialogue events")


def create_clean_mkv(
    original_mkv: Path, english_ass: Path, polish_ass: Path, output_mkv: Path
):
    """Create clean MKV with only video/audio + English dialogue + Polish dialogue."""
    print(f"🎬 Creating clean MKV: {output_mkv.name}")
    print("   - Adding: English dialogue + Polish dialogue only")

    # Build mkvmerge command
    cmd = [
        "mkvmerge",
        "-o",
        str(output_mkv),
        # Add video and audio tracks from original (skip all subtitles)
        "--no-subtitles",
        str(original_mkv),
        # Add clean English dialogue
        "--language",
        "0:eng",
        "--track-name",
        "0:English Dialogue",
        str(english_ass),
        # Add Polish translation
        "--language",
        "0:pol",
        "--track-name",
        "0:Polish (AI)",
        str(polish_ass),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   - Clean MKV merge successful")

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


def verify_clean_result(output_mkv: Path):
    """Verify the clean MKV has only the desired tracks."""
    print(f"🔍 Verifying clean result: {output_mkv.name}")

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

        # Check for exactly 2 subtitle tracks: English and Polish
        if len(subtitle_tracks) == 2:
            english_found = any(t["language"] == "eng" for t in subtitle_tracks)
            polish_found = any(t["language"] == "pol" for t in subtitle_tracks)

            if english_found and polish_found:
                print("   ✅ Perfect! Found English dialogue + Polish translation only")
                return True
            else:
                print("   ❌ Missing required languages")
                return False
        else:
            print(f"   ❌ Expected 2 subtitle tracks, found {len(subtitle_tracks)}")
            return False

    except Exception as e:
        print(f"❌ Failed to verify: {e}")
        return False


def main():
    """Create clean MKV with only English dialogue + Polish translation."""
    # File paths
    original_mkv = Path("/Users/arlen/Downloads/test_movies/SPY x FAMILY - S01E01.mkv")
    extracted_ass = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_english.ass"
    )
    clean_english_ass = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_clean_english.ass"
    )
    polish_ass = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_polish.ass"
    )
    clean_mkv = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_clean.mkv"
    )

    if not original_mkv.exists():
        print(f"❌ Original MKV not found: {original_mkv}")
        return

    if not extracted_ass.exists():
        print(f"❌ Extracted ASS not found: {extracted_ass}")
        print("Please run Step 1 first to extract subtitles")
        return

    print("🎬 Clean Pipeline: English Dialogue + Polish Translation Only")
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

    # Step 3: Create clean English ASS (dialogue only)
    create_clean_english_ass(extracted_ass, dialogue_lines, clean_english_ass)

    # Step 4: Create Polish ASS (translated dialogue only)
    create_polish_ass(extracted_ass, translated_dialogue, polish_ass)

    # Step 5: Create clean MKV
    if create_clean_mkv(original_mkv, clean_english_ass, polish_ass, clean_mkv):
        # Step 6: Verify result
        if verify_clean_result(clean_mkv):
            print("\n🎉 CLEAN PIPELINE SUCCESS!")
            print(f"📁 Clean MKV: {clean_mkv}")
            print("✅ Contains only: English dialogue + Polish AI translation")
            print("🚫 Removed: All signs/songs tracks")
            print("🤖 Model: allegro/BiDi-eng-pol")
            print("⚡ Device: MPS (Apple Silicon)")
        else:
            print("\n❌ Pipeline failed at verification step")
    else:
        print("\n❌ Pipeline failed at MKV merge step")


if __name__ == "__main__":
    main()
