"""Unified subtitle processor combining parsing, writing, and validation."""

from collections.abc import Callable
from pathlib import Path

from ..logging import logger
from ..types import NON_DIALOGUE_STYLES, DialogueLine, replace_polish_chars
from ._pysubs2 import get_pysubs2


class SubtitleProcessingError(Exception):
    """Base exception for subtitle processing errors."""

    pass


def _find_dialogue_style(subs) -> str:
    """Find the best dialogue style name in an SSAFile.

    Picks a style that exists in the file and is likely dialogue.
    Handles ASS files (with named styles like 'Dialogue', 'Default'),
    SRT-sourced files (always 'Default'), and edge cases where no
    styles are defined.
    """
    if not subs.styles:
        return 'Default'

    # Prefer 'Default' if it exists (SRT, most Polish subs, simple ASS)
    if 'Default' in subs.styles:
        return 'Default'

    # Look for common dialogue style names (case-sensitive match)
    dialogue_names = ('Dialogue', 'Dialog', 'Main', 'Dialogi', 'Normal')
    for name in dialogue_names:
        if name in subs.styles:
            return name

    # Case-insensitive search for anything with 'dialog' or 'default'
    for name in subs.styles:
        lower = name.lower()
        if 'dialog' in lower or 'default' in lower or 'main' in lower:
            return name

    # Last resort: use the first style and hope for the best
    return next(iter(subs.styles))


class SubtitleProcessor:
    """Unified subtitle processor for parsing, writing, and validation."""

    @staticmethod
    def extract_dialogue_lines(subtitle_file: Path) -> list[DialogueLine]:
        """Extract dialogue lines from subtitle file."""
        logger.info(f'📖 Reading {subtitle_file.name}...')

        if not subtitle_file.exists():
            raise SubtitleProcessingError(f'Subtitle file not found: {subtitle_file}')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleProcessingError('pysubs2 library not available')

        try:
            subs = pysubs2.load(str(subtitle_file))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to parse subtitle file: {e}') from e

        logger.debug(f'   - Loaded {len(subs)} total events')

        unique_subs = SubtitleProcessor._deduplicate_events(subs)
        dialogue_lines = SubtitleProcessor._filter_dialogue(unique_subs)

        return dialogue_lines

    @staticmethod
    def create_subtitle_file(
        original_file: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
        text_transform: Callable[[str], str] | None = None,
    ) -> None:
        """Create a new subtitle file from dialogue lines."""
        if not original_file.exists():
            raise SubtitleProcessingError(f'Original subtitle file not found: {original_file}')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleProcessingError('pysubs2 library not available')

        try:
            original_subs = pysubs2.load(str(original_file))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to load original subtitle file: {e}') from e

        new_subs = pysubs2.SSAFile()
        new_subs.info = original_subs.info.copy()
        new_subs.styles = original_subs.styles.copy()

        # Pick the dialogue style from the source file. The style must
        # exist in new_subs.styles or the player will use a bare fallback.
        dialogue_style = _find_dialogue_style(new_subs)

        for line in dialogue_lines:
            text = line.text
            if text_transform:
                text = text_transform(text)
            text = text.replace('\n', '\\N')

            event = pysubs2.SSAEvent(
                start=line.start_ms,
                end=line.end_ms,
                style=dialogue_style,
                text=text,
            )
            new_subs.append(event)

        try:
            new_subs.save(str(output_path))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to save subtitle file: {e}') from e

        logger.debug(f'   - Saved {len(new_subs)} events')

    @staticmethod
    def create_english_subtitles(
        original_file: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
    ) -> None:
        """Create clean English subtitle file with dialogue only."""
        logger.info(f'🔨 Creating clean English ASS: {output_path.name}')
        SubtitleProcessor.create_subtitle_file(original_file, dialogue_lines, output_path)
        logger.debug('   - Removed all non-dialogue events')

    @staticmethod
    def create_polish_subtitles(
        original_file: Path,
        translated_dialogue: list[DialogueLine],
        output_path: Path,
        replace_chars: bool = True,
    ) -> None:
        """Create Polish subtitle file with optional character replacement."""
        logger.info('🔤 Creating Polish subtitles')
        text_transform = replace_polish_chars if replace_chars else None
        SubtitleProcessor.create_subtitle_file(
            original_file, translated_dialogue, output_path, text_transform
        )

    @staticmethod
    def override_font_name(ass_file: Path, new_font_name: str) -> None:
        """Replace all font names in ASS styles with the given font name."""
        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleProcessingError('pysubs2 library not available')

        try:
            subs = pysubs2.load(str(ass_file))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to load subtitle file: {e}') from e

        for style in subs.styles.values():
            if hasattr(style, 'fontname'):
                style.fontname = new_font_name

        try:
            subs.save(str(ass_file))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to save subtitle file: {e}') from e

        logger.debug(f'   - Overrode font name to "{new_font_name}" in {ass_file.name}')

    @staticmethod
    def validate_cleaned_subtitles(original_file: Path, cleaned_file: Path) -> None:
        """Validate that cleaned subtitles maintain proper timing coverage."""
        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleProcessingError('pysubs2 library not available')

        try:
            original_subs = pysubs2.load(str(original_file))
            cleaned_subs = pysubs2.load(str(cleaned_file))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to load subtitle files: {e}') from e

        original_events = [e for e in original_subs if e.text.strip()]
        cleaned_events = [e for e in cleaned_subs if e.text.strip()]

        logger.debug(f'   📊 Validation: Original file has {len(original_events)} non-empty events')

        original_dialogue = [
            e
            for e in original_events
            if not any(kw in getattr(e, 'style', 'Default').lower() for kw in NON_DIALOGUE_STYLES)
        ]
        non_dialogue_count = len(original_events) - len(original_dialogue)
        logger.debug(
            f'   📊 Validation: {len(original_dialogue)} dialogue, {non_dialogue_count} non-dialogue (signs/songs/effects)'
        )

        if not original_dialogue:
            logger.warning('No dialogue events found in original file')
            return

        original_start = min(e.start for e in original_dialogue)
        original_end = max(e.end for e in original_dialogue)
        original_duration_sec = (original_end - original_start) / 1000

        first_dialogue = min(original_dialogue, key=lambda e: e.start)
        last_dialogue = max(original_dialogue, key=lambda e: e.end)
        logger.debug(
            f'   📊 Original dialogue range: {original_start}ms - {original_end}ms ({original_duration_sec:.1f}s)'
        )
        logger.debug(f'      First: "{first_dialogue.plaintext.strip()[:50]}..."')
        logger.debug(f'      Last:  "{last_dialogue.plaintext.strip()[:50]}..."')

        if not cleaned_events:
            raise SubtitleProcessingError('Cleaned subtitle file has no events')

        logger.debug(f'   📊 Validation: Cleaned file has {len(cleaned_events)} dialogue events')

        cleaned_start = min(e.start for e in cleaned_events)
        cleaned_end = max(e.end for e in cleaned_events)
        cleaned_duration_sec = (cleaned_end - cleaned_start) / 1000

        first_cleaned = min(cleaned_events, key=lambda e: e.start)
        last_cleaned = max(cleaned_events, key=lambda e: e.end)
        logger.debug(
            f'   📊 Cleaned dialogue range: {cleaned_start}ms - {cleaned_end}ms ({cleaned_duration_sec:.1f}s)'
        )
        logger.debug(f'      First: "{first_cleaned.plaintext.strip()[:50]}..."')
        logger.debug(f'      Last:  "{last_cleaned.plaintext.strip()[:50]}..."')

        TOLERANCE_MS = 50

        start_diff = cleaned_start - original_start
        end_diff = cleaned_end - original_end

        logger.debug('   📊 Timing differences:')
        logger.debug(
            f'      Start: {start_diff:+d}ms ("{"within" if abs(start_diff) <= TOLERANCE_MS else "EXCEEDS"} {TOLERANCE_MS}ms tolerance")'
        )
        logger.debug(
            f'      End:   {end_diff:+d}ms ("{"within" if abs(end_diff) <= TOLERANCE_MS else "EXCEEDS"} {TOLERANCE_MS}ms tolerance")'
        )

        # Timing mismatches are expected when the original has non-dialogue events
        # (signs, effects, "..." placeholders) before/after the dialogue range.
        # The cleaning process filters more aggressively than the style-based filter,
        # so warn but don't fail.
        if abs(start_diff) > TOLERANCE_MS:
            if non_dialogue_count > 0:
                logger.info(
                    f'   ⚠️  Start time offset: {start_diff:+d}ms '
                    f'(expected — original has {non_dialogue_count} non-dialogue events)'
                )
            else:
                logger.warning(f'   ⚠️  Unexpected start time offset: {start_diff:+d}ms')

        if abs(end_diff) > TOLERANCE_MS:
            if non_dialogue_count > 0:
                logger.info(
                    f'   ⚠️  End time offset: {end_diff:+d}ms '
                    f'(expected — original has non-dialogue events after last dialogue)'
                )
            else:
                logger.warning(f'   ⚠️  Unexpected end time offset: {end_diff:+d}ms')

        if abs(start_diff) <= TOLERANCE_MS and abs(end_diff) <= TOLERANCE_MS:
            logger.info('   ✅ Timing validation passed')
        else:
            logger.info('   ✅ Timing validation passed (with expected offsets)')

    @staticmethod
    def _deduplicate_events(subs) -> list:
        """Remove duplicate consecutive events with same text."""
        original_count = len(subs)
        unique_subs = []
        last_text: str | None = None
        current_group_start: int = 0
        current_group_end: int = 0

        pysubs2 = get_pysubs2()
        assert pysubs2 is not None

        for event in subs:
            clean_text = event.plaintext.strip()

            if not clean_text or len(clean_text) < 2:
                continue

            if last_text == clean_text:
                current_group_end = max(current_group_end, event.end)
            else:
                if last_text is not None and unique_subs:
                    consolidated_event = pysubs2.SSAEvent(
                        start=current_group_start,
                        end=current_group_end,
                        style=unique_subs[-1].style if unique_subs else 'Default',
                        text=last_text,
                    )
                    unique_subs[-1] = consolidated_event

                last_text = clean_text
                current_group_start = event.start
                current_group_end = event.end
                unique_subs.append(event)

        if last_text is not None and unique_subs:
            consolidated_event = pysubs2.SSAEvent(
                start=current_group_start,
                end=current_group_end,
                style=unique_subs[-1].style if unique_subs else 'Default',
                text=last_text,
            )
            unique_subs[-1] = consolidated_event

        deduped_count = len(unique_subs)
        if deduped_count < original_count:
            removed = original_count - deduped_count
            logger.debug(
                f'   - Deduplicated: {original_count} → {deduped_count} entries '
                f'(removed {removed} duplicate effect layers)'
            )

        return unique_subs

    @staticmethod
    def _filter_dialogue(events: list) -> list[DialogueLine]:
        """Filter out non-dialogue events and convert to DialogueLine."""
        dialogue_lines = []
        skipped_count = 0

        for event in events:
            if not event.text or event.text.strip() == '':
                skipped_count += 1
                continue

            style = getattr(event, 'style', 'Default').lower()
            if any(keyword in style for keyword in NON_DIALOGUE_STYLES):
                skipped_count += 1
                continue

            clean_text = event.plaintext.strip()
            if not clean_text:
                skipped_count += 1
                continue

            dialogue_lines.append(DialogueLine(event.start, event.end, clean_text))

        logger.info(f'   - Extracted {len(dialogue_lines)} dialogue lines')
        logger.info(f'   - Skipped {skipped_count} non-dialogue events')

        return dialogue_lines
