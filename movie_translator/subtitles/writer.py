from collections.abc import Callable
from pathlib import Path

from ..logging import logger
from ..types import DialogueLine, replace_polish_chars
from ._pysubs2 import get_pysubs2


class SubtitleWriteError(Exception):
    pass


class SubtitleWriter:
    def create_english_ass(
        self,
        original_ass: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
    ) -> None:
        logger.info(f'🔨 Creating clean English ASS: {output_path.name}')
        self._create_ass_file(original_ass, dialogue_lines, output_path)
        logger.info('   - Removed all non-dialogue events')

    def create_polish_ass(
        self,
        original_ass: Path,
        translated_dialogue: list[DialogueLine],
        output_path: Path,
        replace_chars: bool = True,
    ) -> None:
        logger.info('🔤 Creating Polish subtitles')
        text_transform = replace_polish_chars if replace_chars else None
        self._create_ass_file(original_ass, translated_dialogue, output_path, text_transform)

    def _create_ass_file(
        self,
        original_ass: Path,
        dialogue_lines: list[DialogueLine],
        output_path: Path,
        text_transform: Callable[[str], str] | None = None,
    ) -> None:
        if not original_ass.exists():
            raise SubtitleWriteError(f'Original subtitle file not found: {original_ass}')

        pysubs2 = get_pysubs2()
        if pysubs2 is None:
            raise SubtitleWriteError('pysubs2 library not available')

        try:
            original_subs = pysubs2.load(str(original_ass))
        except Exception as e:
            raise SubtitleWriteError(f'Failed to load original subtitle file: {e}') from e

        logger.info(f'   - Loaded {len(original_subs)} original events')

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
            raise SubtitleWriteError(f'Failed to save subtitle file: {e}') from e

        logger.info(f'   - Saved {len(new_subs)} events')
