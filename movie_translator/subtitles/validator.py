from pathlib import Path

from ..utils import log_error, log_info, log_success, log_warning


class SubtitleValidator:
    def validate_cleaned_subtitles(self, original_ass: Path, cleaned_ass: Path) -> bool:
        log_info('🔍 Validating cleaned subtitles...')

        try:
            import pysubs2
        except ImportError:
            log_error('pysubs2 package not found. Install with: uv add pysubs2')
            return False

        try:
            original_subs = pysubs2.load(str(original_ass))
            cleaned_subs = pysubs2.load(str(cleaned_ass))

            cleaned_dict = self._build_cleaned_dict(cleaned_subs)
            stats = self._validate_coverage(original_subs, cleaned_dict)

            self._log_validation_stats(stats, len(cleaned_subs))

            if stats['mismatches'] == 0:
                log_success('   ✅ All original events are properly covered!')
                return True
            else:
                log_error(f'   ❌ Found {stats["mismatches"]} timing gaps!')
                return False

        except Exception as e:
            log_error(f'Failed to validate cleaned subtitles: {e}')
            return False

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

        non_dialogue_styles = ('sign', 'song', 'title', 'op', 'ed')

        for event in original_subs:
            style = getattr(event, 'style', 'Default').lower()
            if any(keyword in style for keyword in non_dialogue_styles):
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

    def _check_timing_coverage(self, event, cleaned_timings: list[tuple[int, int]], tolerance_ms: int = 50) -> bool:
        for clean_start, clean_end in cleaned_timings:
            start_ok = clean_start <= event.start + tolerance_ms
            end_ok = clean_end >= event.end - tolerance_ms
            if start_ok and end_ok:
                return True
        return False

    def _log_timing_gap(self, text: str, event, cleaned_timings: list[tuple[int, int]]):
        log_warning(f'   - Timing gap for "{text[:30]}..."')
        log_warning(f'     Original: {event.start} → {event.end}')
        for clean_start, clean_end in cleaned_timings:
            log_warning(f'     Cleaned:  {clean_start} → {clean_end}')

    def _log_validation_stats(self, stats: dict, cleaned_count: int):
        log_info(f'   - Original dialogue lines: {stats["original_dialogue_count"]}')
        log_info(f'   - Cleaned dialogue lines:  {cleaned_count}')
        log_info(f'   - Lines covered:          {stats["matches"]}')
        log_info(f'   - Timing gaps:            {stats["mismatches"]}')
        removed = stats['original_dialogue_count'] - stats['found_in_cleaned']
        log_info(f'   - Lines correctly removed: {removed}')
