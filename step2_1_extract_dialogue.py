#!/usr/bin/env python3
"""
Step 2.1: Extract Clean Dialogue Text from ASS Files
"""

import sys
from pathlib import Path

try:
    import pysubs2
except ImportError:
    print("❌ pysubs2 not found. Install with: uv add pysubs2")
    sys.exit(1)


def extract_dialogue_lines(ass_file: Path) -> list[tuple[int, int, str]]:
    """
    Extract clean dialogue lines from ASS file.

    Returns:
        List of tuples: (start_time_ms, end_time_ms, text)
    """
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


def format_time(ms: int) -> str:
    """Format milliseconds to readable time."""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def print_dialogue_sample(
    dialogue_lines: list[tuple[int, int, str]], max_lines: int = 10
):
    """Print a sample of dialogue lines."""
    print(f"\n📝 First {max_lines} dialogue lines:")
    print("=" * 60)

    for i, (start, end, text) in enumerate(dialogue_lines[:max_lines]):
        start_time = format_time(start)
        print(f"[{start_time}] {text}")


def save_dialogue_text(dialogue_lines: list[tuple[int, int, str]], output_file: Path):
    """Save dialogue lines to a plain text file."""
    print(f"\n💾 Saving dialogue to {output_file.name}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(dialogue_lines, 1):
            start_time = format_time(start)
            f.write(f"{i:3d}. [{start_time}] {text}\n")

    print(f"   - Saved {len(dialogue_lines)} lines")
    print(f"   - File size: {output_file.stat().st_size:,} bytes")


def main():
    """Test dialogue extraction."""
    # Use the extracted subtitle file
    ass_file = Path(
        "/Users/arlen/Downloads/test_movies/subtitles/SPY x FAMILY - S01E01_english.ass"
    )

    if not ass_file.exists():
        print(f"❌ File not found: {ass_file}")
        return

    print("🎬 Step 2.1: Extract Clean Dialogue Text")
    print("=" * 60)

    # Extract dialogue lines
    dialogue_lines = extract_dialogue_lines(ass_file)

    if not dialogue_lines:
        print("❌ No dialogue lines found")
        return

    # Show sample
    print_dialogue_sample(dialogue_lines, max_lines=15)

    # Save to text file
    output_file = ass_file.parent / "dialogue_only.txt"
    save_dialogue_text(dialogue_lines, output_file)

    print("\n✅ Step 2.1 complete!")
    print(f"📁 Dialogue text saved to: {output_file}")


if __name__ == "__main__":
    main()
