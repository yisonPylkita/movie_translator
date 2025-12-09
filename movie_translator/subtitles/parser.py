"""Subtitle parsing and dialogue extraction."""

from pathlib import Path

from ..utils import log_error, log_info


class SubtitleParser:
    """Parses subtitle files and extracts dialogue lines."""

    # Keywords indicating non-dialogue styles
    NON_DIALOGUE_STYLES = ('sign', 'song', 'title', 'op', 'ed')

    def extract_dialogue_lines(self, subtitle_file: Path) -> list[tuple[int, int, str]]:
        """Extract dialogue lines from subtitle file with duplicate handling."""
        log_info(f'📖 Reading {subtitle_file.name}...')

        subs = self._load_subtitle_file(subtitle_file)
        if subs is None:
            return []

        log_info(f'   - Loaded {len(subs)} total events')

        unique_subs = self._deduplicate_events(subs)
        dialogue_lines = self._filter_dialogue(unique_subs)

        return dialogue_lines

    def _load_subtitle_file(self, subtitle_file: Path):
        """Load subtitle file using pysubs2."""
        try:
            import pysubs2
        except ImportError:
            log_error('pysubs2 package not found. Install with: uv add pysubs2')
            return None

        try:
            return pysubs2.load(str(subtitle_file))
        except Exception as e:
            log_error(f'Failed to load: {e}')
            return None

    def _deduplicate_events(self, subs) -> list:
        """Group consecutive events with the same text (animated subtitles)."""

        original_count = len(subs)
        unique_subs = []
        last_text = None
        current_group_start = None
        current_group_end = None

        for event in subs:
            clean_text = event.plaintext.strip()

            if not clean_text or len(clean_text) < 2:
                continue

            if last_text == clean_text:
                # Extend the time span for consecutive duplicates
                current_group_end = max(current_group_end, event.end)
            else:
                # Save previous group
                if last_text is not None and unique_subs:
                    self._consolidate_last_event(
                        unique_subs, current_group_start, current_group_end, last_text
                    )

                # Start new group
                last_text = clean_text
                current_group_start = event.start
                current_group_end = event.end
                unique_subs.append(event)

        # Handle the last group
        if last_text is not None and unique_subs:
            self._consolidate_last_event(
                unique_subs, current_group_start, current_group_end, last_text
            )

        deduped_count = len(unique_subs)
        if deduped_count < original_count:
            removed = original_count - deduped_count
            log_info(
                f'   - Deduplicated: {original_count} → {deduped_count} entries (removed {removed} duplicate effect layers)'
            )

        return unique_subs

    def _consolidate_last_event(self, unique_subs: list, start: int, end: int, text: str):
        """Consolidate the last event with updated timing."""
        import pysubs2

        consolidated_event = pysubs2.SSAEvent(
            start=start,
            end=end,
            style=unique_subs[-1].style if unique_subs else 'Default',
            text=text,
        )
        unique_subs[-1] = consolidated_event

    def _filter_dialogue(self, events: list) -> list[tuple[int, int, str]]:
        """Filter events to only include dialogue lines."""
        dialogue_lines = []
        skipped_count = 0

        for event in events:
            if not event.text or event.text.strip() == '':
                skipped_count += 1
                continue

            style = getattr(event, 'style', 'Default').lower()
            if any(keyword in style for keyword in self.NON_DIALOGUE_STYLES):
                skipped_count += 1
                continue

            clean_text = event.plaintext.strip()
            if not clean_text:
                skipped_count += 1
                continue

            dialogue_lines.append((event.start, event.end, clean_text))

        log_info(f'   - Extracted {len(dialogue_lines)} dialogue lines')
        log_info(f'   - Skipped {skipped_count} non-dialogue events')

        return dialogue_lines
