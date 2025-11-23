#!/usr/bin/env python3
"""Debug script: Extract ASS subtitles and convert to SRT (no translation)."""

import sys
import re
from pathlib import Path
import pysubs2


def strip_html_tags(text: str) -> str:
    """Remove HTML and ASS tags from subtitle text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)
    return text.strip()


def convert_ass_to_srt(input_path: Path, output_path: Path) -> None:
    """Convert ASS to SRT with deduplication and debugging."""

    print(f"\n{'='*60}")
    print(f"ASS → SRT Conversion Debug")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load ASS file
    print("[1/4] Loading ASS file...")
    subs = pysubs2.load(str(input_path))
    print(f"  ✓ Loaded {len(subs)} subtitle events")

    # Show sample before deduplication
    print(f"\n[2/4] Sample BEFORE deduplication (first 5):")
    for i, event in enumerate(subs[:5]):
        clean_text = strip_html_tags(event.text)
        print(f"  {i+1}. [{event.start//1000}s→{event.end//1000}s] {clean_text[:60]}")

    # Deduplicate
    print(f"\n[3/4] Deduplicating...")
    original_count = len(subs)
    seen = {}
    unique_subs = pysubs2.SSAFile()
    unique_subs.styles = subs.styles.copy()

    for event in subs:
        # Create key from timestamp and stripped text
        clean_text = strip_html_tags(event.text)
        key = (event.start, event.end, clean_text)

        # Only keep first occurrence
        if key not in seen:
            seen[key] = True
            unique_subs.append(event)

    deduped_count = len(unique_subs)
    removed_count = original_count - deduped_count
    print(f"  ✓ Original: {original_count} entries")
    print(f"  ✓ Unique:   {deduped_count} entries")
    print(f"  ✓ Removed:  {removed_count} duplicates ({removed_count*100//original_count}%)")

    # Show sample after deduplication
    print(f"\n[4/4] Sample AFTER deduplication (first 10):")
    for i, event in enumerate(unique_subs[:10]):
        clean_text = strip_html_tags(event.text)
        print(f"  {i+1}. [{event.start//1000}s→{event.end//1000}s] {clean_text[:60]}")

    # Sort by timestamp
    unique_subs.sort()

    # Save as SRT
    print(f"\n[SAVE] Writing to {output_path.name}...")
    unique_subs.save(str(output_path))

    # Verify output
    output_lines = output_path.read_text().strip().split('\n')
    print(f"  ✓ Saved {len(output_lines)} lines to SRT file")

    # Count subtitle entries in SRT (entries start with a number)
    entry_count = sum(1 for line in output_lines if line.strip().isdigit())
    print(f"  ✓ SRT contains {entry_count} subtitle entries")

    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Check the output: {output_path}")
    print(f"  2. Open in a subtitle editor or text editor")
    print(f"  3. Verify the content looks correct")
    print(f"  4. Test in a video player")
    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_ass_to_srt.py input.ass output.srt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() in ['.ass', '.ssa']:
        print(f"Error: Input must be ASS/SSA file, got: {input_path.suffix}")
        sys.exit(1)

    try:
        convert_ass_to_srt(input_path, output_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
