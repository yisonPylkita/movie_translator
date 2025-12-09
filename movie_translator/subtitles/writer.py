from pathlib import Path

from ..utils import clear_memory, log_error, log_info, log_success, replace_polish_chars


class SubtitleWriter:
    def create_english_ass(
        self,
        original_ass: Path,
        dialogue_lines: list[tuple[int, int, str]],
        output_path: Path,
    ):
        log_info(f'🔨 Creating clean English ASS: {output_path.name}')

        try:
            import pysubs2
        except ImportError:
            log_error('pysubs2 package not found. Install with: uv add pysubs2')
            return

        try:
            original_subs = pysubs2.load(str(original_ass))
            log_info(f'   - Loaded original with {len(original_subs)} events')

            clean_subs = self._create_subtitle_file(original_subs)
            self._add_dialogue_events(clean_subs, dialogue_lines)

            clean_subs.save(str(output_path))
            log_success(f'   - Saved {len(clean_subs)} dialogue events')
            log_info('   - Removed all non-dialogue events')

        except Exception as e:
            log_error(f'Failed to create clean English ASS: {e}')

    def create_polish_ass(
        self,
        original_ass: Path,
        translated_dialogue: list[tuple[int, int, str]],
        output_path: Path,
        replace_chars: bool = True,
    ):
        log_info('🔤 Creating Polish subtitles')

        try:
            import pysubs2
        except ImportError:
            log_error('pysubs2 package not found. Install with: uv add pysubs2')
            return

        try:
            original_subs = pysubs2.load(str(original_ass))
            log_info(f'   - Loaded {len(original_subs)} original events')

            polish_subs = self._create_subtitle_file(original_subs)
            self._add_translated_events(polish_subs, translated_dialogue, replace_chars)

            polish_subs.save(str(output_path))
            log_success(f'   - Saved {len(polish_subs)} translated events')

            del original_subs
            del polish_subs
            clear_memory()

        except Exception as e:
            log_error(f'Failed to create Polish ASS: {e}')

    def _create_subtitle_file(self, original_subs):
        import pysubs2

        new_subs = pysubs2.SSAFile()
        new_subs.info = original_subs.info.copy()
        new_subs.styles = original_subs.styles.copy()
        return new_subs

    def _add_dialogue_events(self, subs, dialogue_lines: list[tuple[int, int, str]]):
        import pysubs2

        for start, end, text in dialogue_lines:
            clean_text = text.replace('\n', '\\N')
            event = pysubs2.SSAEvent(
                start=start,
                end=end,
                style='Default',
                text=clean_text,
            )
            subs.append(event)

    def _add_translated_events(
        self,
        subs,
        translated_dialogue: list[tuple[int, int, str]],
        replace_chars: bool,
    ):
        import pysubs2

        for start, end, translated_text in translated_dialogue:
            if replace_chars:
                translated_text = replace_polish_chars(translated_text)

            clean_text = translated_text.replace('\n', '\\N')
            event = pysubs2.SSAEvent(
                start=start,
                end=end,
                style='Default',
                text=clean_text,
            )
            subs.append(event)
