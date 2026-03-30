"""AI translation and font checking stage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..context import FontInfo, PipelineContext
from ..fonts import (
    check_embedded_fonts_support_polish,
    find_system_font_for_polish,
    get_ass_font_names,
)
from ..logging import logger
from ..translation import translate_dialogue_lines

if TYPE_CHECKING:
    from ..progress import ProgressTracker


class TranslateStage:
    name = 'translate'

    def __init__(self):
        self._tracker: ProgressTracker | None = None

    def set_tracker(self, tracker: ProgressTracker):
        self._tracker = tracker

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.dialogue_lines is not None
        assert ctx.english_source is not None

        total = len(ctx.dialogue_lines)
        logger.info(f'Translating {total} lines...')

        english_source = ctx.english_source
        tracker = self._tracker

        def _check_fonts():
            supports = check_embedded_fonts_support_polish(ctx.video_path, english_source)
            if supports:
                return FontInfo(supports_polish=True)
            is_mkv = ctx.video_path.suffix.lower() == '.mkv'
            if is_mkv:
                names = get_ass_font_names(english_source)
                result = find_system_font_for_polish(names)
                if result:
                    fp, fam = result
                    fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                    return FontInfo(
                        supports_polish=False,
                        font_attachments=[fp],
                        fallback_font_family=fallback,
                    )
            return FontInfo(supports_polish=False)

        def _on_progress(lines_done: int, total_lines: int, rate: float) -> None:
            if tracker:
                tracker.set_stage_progress(lines_done, total_lines, rate)

        with ThreadPoolExecutor(max_workers=2) as pool:
            font_future = pool.submit(_check_fonts)
            translate_future = pool.submit(
                translate_dialogue_lines,
                ctx.dialogue_lines,
                ctx.config.device,
                ctx.config.batch_size,
                ctx.config.model,
                progress_callback=_on_progress,
            )

            ctx.font_info = font_future.result()
            translated = translate_future.result()

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
