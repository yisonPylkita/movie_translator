#!/usr/bin/env python3
"""
Subtitle validation functions for the Movie Translator pipeline.
Ensures subtitle timing and content integrity.
"""

from pathlib import Path

from .utils import log_error, log_info, log_success, log_warning


def validate_cleaned_subtitles(
    original_ass: Path,
    cleaned_ass: Path,
) -> bool:
    """Validate that cleaned subtitles are a proper subset of original with matching timestamps."""
    log_info('🔍 Validating cleaned subtitles...')

    try:
        import pysubs2
    except ImportError:
        log_error('pysubs2 package not found. Install with: uv add pysubs2')
        return False

    try:
        # Load both files
        original_subs = pysubs2.load(str(original_ass))
        cleaned_subs = pysubs2.load(str(cleaned_ass))

        # Create dictionary of cleaned subs by text for easy lookup
        cleaned_dict = {}
        for event in cleaned_subs:
            clean_text = event.text.replace('\\N', ' ').strip()
            if clean_text:
                if clean_text not in cleaned_dict:
                    cleaned_dict[clean_text] = []
                cleaned_dict[clean_text].append((event.start, event.end))

        # Check each original dialogue line
        mismatches = 0
        matches = 0
        original_dialogue_count = 0
        found_in_cleaned = 0

        for event in original_subs:
            style = getattr(event, 'style', 'Default')
            style_lower = style.lower()

            # Skip non-dialogue styles
            if any(keyword in style_lower for keyword in ['sign', 'song', 'title', 'op', 'ed']):
                continue

            clean_text = event.plaintext.strip()
            if not clean_text or len(clean_text) < 2:
                continue

            original_dialogue_count += 1

            # Check if this dialogue line exists in cleaned version
            if clean_text in cleaned_dict:
                found_in_cleaned += 1

                # Check if this original event is covered by any cleaned event with the same text
                is_covered = False
                for clean_start, clean_end in cleaned_dict[clean_text]:
                    if clean_start <= event.start and clean_end >= event.end:
                        matches += 1
                        is_covered = True
                        break

                if not is_covered:
                    mismatches += 1
                    log_warning(f'   - Timing gap for "{clean_text[:30]}..."')
                    log_warning(f'     Original: {event.start} → {event.end}')
                    for clean_start, clean_end in cleaned_dict[clean_text]:
                        log_warning(f'     Cleaned:  {clean_start} → {clean_end}')

        log_info(f'   - Original dialogue lines: {original_dialogue_count}')
        log_info(f'   - Cleaned dialogue lines:  {len(cleaned_subs)}')
        log_info(f'   - Lines covered:          {matches}')
        log_info(f'   - Timing gaps:            {mismatches}')
        log_info(f'   - Lines correctly removed: {original_dialogue_count - found_in_cleaned}')

        if mismatches == 0:
            log_success('   ✅ All original events are properly covered!')
            return True
        else:
            log_error(f'   ❌ Found {mismatches} timing gaps!')
            return False

    except Exception as e:
        log_error(f'Failed to validate cleaned subtitles: {e}')
        return False
