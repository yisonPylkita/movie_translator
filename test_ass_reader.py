#!/usr/bin/env python3
"""
ASS File Reader and Manipulator
Demonstrates reading ASS files and extracting dialogue lines
"""

import sys
from pathlib import Path

try:
    import pysubs2

    print("✅ pysubs2 imported successfully")
except ImportError:
    print("❌ pysubs2 not found. Install with: pip install pysubs2")
    sys.exit(1)


def read_ass_file(file_path: Path) -> pysubs2.SSAFile | None:
    """Read ASS file and return subtitle object."""
    try:
        subs = pysubs2.load(str(file_path))
        print(f"✅ Loaded {file_path.name}")
        print(f"   - Total events: {len(subs)}")
        return subs
    except Exception as e:
        print(f"❌ Failed to load {file_path.name}: {e}")
        return None


def print_dialogue_lines(subs: pysubs2.SSAFile, max_lines: int = 20):
    """Print dialogue lines in chronological order."""
    print(f"\n📝 First {max_lines} dialogue lines:")
    print("=" * 60)

    count = 0
    for i, event in enumerate(subs):
        if count >= max_lines:
            break

        # Skip non-dialogue events (signs, songs, etc.)
        if not event.text or event.text.strip() == "":
            continue

        # Skip events that are likely signs/songs based on style name
        if hasattr(event, "style") and event.style:
            style_lower = event.style.lower()
            if any(
                keyword in style_lower
                for keyword in ["sign", "song", "title", "op", "ed"]
            ):
                continue

        # Clean up the text (remove ASS formatting)
        clean_text = event.plaintext

        if clean_text.strip():  # Only show non-empty text
            start_time = event.start / 1000  # Convert from ms to seconds
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)

            print(f"[{minutes:02d}:{seconds:02d}] {clean_text}")
            count += 1


def show_ass_info(subs: pysubs2.SSAFile):
    """Show information about the ASS file."""
    print("\n📊 ASS File Information:")
    print("=" * 60)
    print(f"Script Type: {subs.info.get('ScriptType', 'Unknown')}")
    print(f"Total Events: {len(subs)}")
    print(f"Styles: {len(subs.styles)}")

    # Show available styles
    print("\n🎨 Available Styles:")
    for style_name in subs.styles.keys():
        print(f"   - {style_name}")

    # Count events by style
    style_counts = {}
    for event in subs:
        style = getattr(event, "style", "Default")
        style_counts[style] = style_counts.get(style, 0) + 1

    print("\n📈 Events by Style:")
    for style, count in sorted(style_counts.items()):
        print(f"   - {style}: {count} events")


def modify_ass_example(subs: pysubs2.SSAFile):
    """Example of modifying ASS file - add a test subtitle."""
    print("\n✏️  Modification Example:")
    print("=" * 60)

    # Create a new subtitle event
    new_event = pysubs2.SSAEvent(
        start=1000,  # 1 second
        end=4000,  # 4 seconds
        style="Default",
        text="This is a test subtitle added by Python!",
    )

    # Add to the file
    subs.append(new_event)
    print("✅ Added test subtitle at 00:01-00:04")

    # Modify an existing subtitle
    if len(subs) > 0:
        original_text = subs[0].text
        subs[0].text = f"[MODIFIED] {original_text}"
        print("✅ Modified first subtitle")

    return subs


def main():
    """Main function to test ASS file operations."""
    # Use the extracted subtitle file
    ass_file = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_english.ass"
    )

    if not ass_file.exists():
        print(f"❌ File not found: {ass_file}")
        print("Please make sure you've extracted subtitles first")
        return

    print("🎬 ASS File Reader and Manipulator")
    print("=" * 60)

    # Read the ASS file
    subs = read_ass_file(ass_file)
    if not subs:
        return

    # Show file information
    show_ass_info(subs)

    # Print dialogue lines
    print_dialogue_lines(subs, max_lines=15)

    # Example modification (commented out to avoid changing the file)
    # modified_subs = modify_ass_example(subs)
    # pysubs2.save(modified_subs, str(ass_file.with_suffix('.modified.ass')))
    # print(f"✅ Saved modified version")


if __name__ == "__main__":
    main()
