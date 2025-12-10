from pathlib import Path

from ..logging import logger
from ..types import NON_DIALOGUE_STYLES
from ._pysubs2 import get_pysubs2

TIMING_TOLERANCE_MS = 50


class SubtitleValidationError(Exception):
    pass


class SubtitleValidator:
    def validate_cleaned_subtitles(self, original_ass: Path, cleaned_ass: Path) -> None:
        logger.info('🔍 Validating cleaned subtitles...')

        if not original_ass.exists():
            raise SubtitleValidationError(f'Original subtitle file not found: {original_ass}')
        if not cleaned_ass.exists():
            raise SubtitleValidationError(f'Cleaned subtitle file not found: {cleaned_ass}')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleValidationError('pysubs2 library not available')

        try:
            original_subs = pysubs2.load(str(original_ass))
            cleaned_subs = pysubs2.load(str(cleaned_ass))
        except Exception as e:
            raise SubtitleValidationError(f'Failed to load subtitle files: {e}') from e

        cleaned_dict = self._build_cleaned_dict(cleaned_subs)
        stats = self._validate_coverage(original_subs, cleaned_dict)

        self._log_validation_stats(stats, len(cleaned_subs))

        if stats['mismatches'] > 0:
            raise SubtitleValidationError(f'Found {stats["mismatches"]} timing gaps')

        logger.info('   ✅ All original events are properly covered!')

    def _build_cleaned_dict(self, cleaned_subs) -> dict[str, list[tuple[int, int]]]:
        cleaned_dict = {}
        for event in cleaned_subs:
            clean_text = event.text.replace('\\N', ' ').strip()
            if clean_text:
                if clean_text not in cleaned_dict:
                    cleaned_dict[clean_text] = []
                cleaned_dict[clean_text].append((event.start, event.end))
        return cleaned_dict

    def _validate_coverage(self, original_subs, cleaned_dict: dict) -> dict:
        stats = {
            'mismatches': 0,
            'matches': 0,
            'original_dialogue_count': 0,
            'found_in_cleaned': 0,
        }

        for event in original_subs:
            style = getattr(event, 'style', 'Default').lower()
            if any(keyword in style for keyword in NON_DIALOGUE_STYLES):
                continue

            clean_text = event.plaintext.strip()
            if not clean_text or len(clean_text) < 2:
                continue

            stats['original_dialogue_count'] += 1

            if clean_text in cleaned_dict:
                stats['found_in_cleaned'] += 1
                is_covered = self._check_timing_coverage(event, cleaned_dict[clean_text])

                if is_covered:
                    stats['matches'] += 1
                else:
                    stats['mismatches'] += 1
                    self._log_timing_gap(clean_text, event, cleaned_dict[clean_text])

        return stats

    def _check_timing_coverage(self, event, cleaned_timings: list[tuple[int, int]]) -> bool:
        for clean_start, clean_end in cleaned_timings:
            start_ok = clean_start <= event.start + TIMING_TOLERANCE_MS
            end_ok = clean_end >= event.end - TIMING_TOLERANCE_MS
            if start_ok and end_ok:
                return True
        return False

    def _log_timing_gap(self, text: str, event, cleaned_timings: list[tuple[int, int]]):
        logger.warning(f'   - Timing gap for "{text[:30]}..."')
        logger.warning(f'     Original: {event.start} → {event.end}')
        for clean_start, clean_end in cleaned_timings:
            logger.warning(f'     Cleaned:  {clean_start} → {clean_end}')

    def _log_validation_stats(self, stats: dict, cleaned_count: int):
        logger.info(f'   - Original dialogue lines: {stats["original_dialogue_count"]}')
        logger.info(f'   - Cleaned dialogue lines:  {cleaned_count}')
        logger.info(f'   - Lines covered:          {stats["matches"]}')
        logger.info(f'   - Timing gaps:            {stats["mismatches"]}')
        removed = stats['original_dialogue_count'] - stats['found_in_cleaned']
        logger.info(f'   - Lines correctly removed: {removed}')
