import gc
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine
from ._pysubs2 import get_pysubs2

POLISH_CHAR_MAP = str.maketrans(
    'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ',
    'acelnoszzACELNOSZZ',
)


def _replace_polish_chars(text: str) -> str:
    return text.translate(POLISH_CHAR_MAP)


class SubtitleWriter:
    def create_english_ass(
        self,
        original_ass: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
    ):
        logger.info(f'🔨 Creating clean English ASS: {output_path.name}')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            return

        try:
            original_subs = pysubs2.load(str(original_ass))
            logger.info(f'   - Loaded original with {len(original_subs)} events')

            clean_subs = self._create_subtitle_file(pysubs2, original_subs)
            self._add_dialogue_events(pysubs2, clean_subs, dialogue_lines)

            clean_subs.save(str(output_path))
            logger.info(f'   - Saved {len(clean_subs)} dialogue events')
            logger.info('   - Removed all non-dialogue events')

        except Exception as e:
            logger.error(f'Failed to create clean English ASS: {e}')

    def create_polish_ass(
        self,
        original_ass: Path,
        translated_dialogue: list[DialogueLine],
        output_path: Path,
        replace_chars: bool = True,
    ):
        logger.info('🔤 Creating Polish subtitles')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            return

        try:
            original_subs = pysubs2.load(str(original_ass))
            logger.info(f'   - Loaded {len(original_subs)} original events')

            polish_subs = self._create_subtitle_file(pysubs2, original_subs)
            self._add_translated_events(pysubs2, polish_subs, translated_dialogue, replace_chars)

            polish_subs.save(str(output_path))
            logger.info(f'   - Saved {len(polish_subs)} translated events')

            del original_subs
            del polish_subs
            gc.collect()

        except Exception as e:
            logger.error(f'Failed to create Polish ASS: {e}')

    def _create_subtitle_file(self, pysubs2, original_subs):
        new_subs = pysubs2.SSAFile()
        new_subs.info = original_subs.info.copy()
        new_subs.styles = original_subs.styles.copy()
        return new_subs

    def _add_dialogue_events(self, pysubs2, subs, dialogue_lines: list[DialogueLine]):
        for line in dialogue_lines:
            clean_text = line.text.replace('\n', '\\N')
            event = pysubs2.SSAEvent(
                start=line.start_ms,
                end=line.end_ms,
                style='Default',
                text=clean_text,
            )
            subs.append(event)

    def _add_translated_events(
        self,
        pysubs2,
        subs,
        translated_dialogue: list[DialogueLine],
        replace_chars: bool,
    ):
        for line in translated_dialogue:
            text = line.text
            if replace_chars:
                text = _replace_polish_chars(text)

            clean_text = text.replace('\n', '\\N')
            event = pysubs2.SSAEvent(
                start=line.start_ms,
                end=line.end_ms,
                style='Default',
                text=clean_text,
            )
            subs.append(event)
