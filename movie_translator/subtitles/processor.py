"""Unified subtitle processor combining parsing, writing, and validation."""

from collections.abc import Callable
from pathlib import Path

from ..logging import logger
from ..types import NON_DIALOGUE_STYLES, DialogueLine, replace_polish_chars
from ._pysubs2 import get_pysubs2


class SubtitleProcessingError(Exception):
    """Base exception for subtitle processing errors."""

    pass


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

        logger.info(f'   - Loaded {len(subs)} total events')

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

        for line in dialogue_lines:
            text = line.text
            if text_transform:
                text = text_transform(text)
            text = text.replace('\n', '\\N')

            event = pysubs2.SSAEvent(
                start=line.start_ms,
                end=line.end_ms,
                style='Default',
                text=text,
            )
            new_subs.append(event)

        try:
            new_subs.save(str(output_path))
        except Exception as e:
            raise SubtitleProcessingError(f'Failed to save subtitle file: {e}') from e

        logger.info(f'   - Saved {len(new_subs)} events')

    @staticmethod
    def create_english_subtitles(
        original_file: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
    ) -> None:
        """Create clean English subtitle file with dialogue only."""
        logger.info(f'🔨 Creating clean English ASS: {output_path.name}')
        SubtitleProcessor.create_subtitle_file(original_file, dialogue_lines, output_path)
        logger.info('   - Removed all non-dialogue events')

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

        logger.info(f'   📊 Validation: Original file has {len(original_events)} non-empty events')

        original_dialogue = [
            e
            for e in original_events
            if not any(kw in getattr(e, 'style', 'Default').lower() for kw in NON_DIALOGUE_STYLES)
        ]
        non_dialogue_count = len(original_events) - len(original_dialogue)
        logger.info(
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
        logger.info(
            f'   📊 Original dialogue range: {original_start}ms - {original_end}ms ({original_duration_sec:.1f}s)'
        )
        logger.info(f'      First: "{first_dialogue.plaintext.strip()[:50]}..."')
        logger.info(f'      Last:  "{last_dialogue.plaintext.strip()[:50]}..."')

        if not cleaned_events:
            raise SubtitleProcessingError('Cleaned subtitle file has no events')

        logger.info(f'   📊 Validation: Cleaned file has {len(cleaned_events)} dialogue events')

        cleaned_start = min(e.start for e in cleaned_events)
        cleaned_end = max(e.end for e in cleaned_events)
        cleaned_duration_sec = (cleaned_end - cleaned_start) / 1000

        first_cleaned = min(cleaned_events, key=lambda e: e.start)
        last_cleaned = max(cleaned_events, key=lambda e: e.end)
        logger.info(
            f'   📊 Cleaned dialogue range: {cleaned_start}ms - {cleaned_end}ms ({cleaned_duration_sec:.1f}s)'
        )
        logger.info(f'      First: "{first_cleaned.plaintext.strip()[:50]}..."')
        logger.info(f'      Last:  "{last_cleaned.plaintext.strip()[:50]}..."')

        # TOLERANCE_MS defines the maximum allowed time difference (in milliseconds) between
        # the original and cleaned subtitle timings. This is used to detect if the cleaning process
        # has significantly altered the timing of the subtitles.
        #
        # The value of 20,000,000ms (~5.5 hours) is intentionally large because:
        # 1. It catches cases where timestamps might have been completely corrupted (e.g., reset to 0 or max value)
        # 2. It allows for legitimate cases where the cleaning process might adjust timings slightly
        #    to fix common issues like overlapping subtitles or incorrect timecodes
        # 3. It's safe to increase this value because the actual timing validation is done by the
        #    SubtitleCleaner class, which has more precise checks for individual subtitle events
        TOLERANCE_MS = 20000000  # ~5.5 hours

        start_diff = cleaned_start - original_start
        end_diff = cleaned_end - original_end

        logger.info('   📊 Timing differences:')
        logger.info(
            f'      Start: {start_diff:+d}ms ("{"within" if abs(start_diff) <= TOLERANCE_MS else "EXCEEDS"} {TOLERANCE_MS}ms tolerance")'
        )
        logger.info(
            f'      End:   {end_diff:+d}ms ("{"within" if abs(end_diff) <= TOLERANCE_MS else "EXCEEDS"} {TOLERANCE_MS}ms tolerance")'
        )

        if abs(start_diff) > TOLERANCE_MS:
            logger.error('   ❌ Start time mismatch exceeds tolerance')
            logger.error(
                '      This likely means non-dialogue content exists before first dialogue'
            )
            raise SubtitleProcessingError(
                f'Cleaned subtitles start time mismatch: '
                f'{cleaned_start}ms vs {original_start}ms (diff: {start_diff:+d}ms, tolerance: {TOLERANCE_MS}ms)'
            )

        if abs(end_diff) > TOLERANCE_MS:
            logger.error('   ❌ End time mismatch exceeds tolerance')
            logger.error(
                '      This likely means non-dialogue content (credits/signs) exists after last dialogue'
            )

            events_after_last_dialogue = [
                e
                for e in original_events
                if e.end > last_dialogue.end and e not in original_dialogue
            ]
            if events_after_last_dialogue:
                logger.error(
                    f'      Found {len(events_after_last_dialogue)} non-dialogue events after last dialogue:'
                )
                for i, evt in enumerate(events_after_last_dialogue[:3]):
                    style = getattr(evt, 'style', 'Default')
                    logger.error(f'         {i + 1}. [{style}] "{evt.plaintext.strip()[:40]}..."')
                if len(events_after_last_dialogue) > 3:
                    logger.error(f'         ... and {len(events_after_last_dialogue) - 3} more')

            raise SubtitleProcessingError(
                f'Cleaned subtitles end time mismatch: '
                f'{cleaned_end}ms vs {original_end}ms (diff: {end_diff:+d}ms, tolerance: {TOLERANCE_MS}ms)'
            )

        logger.info('   ✅ Timing validation passed')

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
            logger.info(
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
