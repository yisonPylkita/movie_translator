"""AI translation and font checking stage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING

from ..context import FontInfo, PipelineContext
from ..fonts import (
    check_embedded_fonts_support_polish,
    find_system_font_for_polish,
    get_ass_font_names,
)
from ..logging import logger
from ..translation import translate_dialogue_lines
from ..translation.translator import _get_translator

if TYPE_CHECKING:
    from ..progress import ProgressTracker


class TranslateStage:
    name = 'translate'

    def __init__(self):
        self._tracker: ProgressTracker | None = None

    def set_tracker(self, tracker: ProgressTracker):
        self._tracker = tracker

    def check_fonts(self, ctx: PipelineContext) -> FontInfo:
        """Check font support for Polish characters. IO-bound, safe to run in worker."""
        assert ctx.english_source is not None
        metrics = ctx.metrics
        english_source = ctx.english_source
        with metrics.span('check_fonts') as s:
            supports = check_embedded_fonts_support_polish(ctx.video_path, english_source)
            if supports:
                s.detail('supports_polish', True)
                return FontInfo(supports_polish=True)
            is_mkv = ctx.video_path.suffix.lower() == '.mkv'
            if is_mkv:
                names = get_ass_font_names(english_source)
                result = find_system_font_for_polish(names)
                if result:
                    fp, fam = result
                    fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                    s.detail('supports_polish', False)
                    s.detail('fallback_font', fam)
                    return FontInfo(
                        supports_polish=False,
                        font_attachments=[fp],
                        fallback_font_family=fallback,
                    )
            s.detail('supports_polish', False)
            return FontInfo(supports_polish=False)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.dialogue_lines is not None
        assert ctx.english_source is not None

        total = len(ctx.dialogue_lines)
        logger.info(f'Translating {total} lines...')

        dialogue_lines = ctx.dialogue_lines
        tracker = self._tracker
        metrics = ctx.metrics

        # Ensure model/backend is loaded before parallelizing (only loads once, cached after)
        if ctx.config.model == 'apple':
            from ..translation.apple_backend import _get_apple_backend

            _backend = _get_apple_backend(ctx.config.batch_size)
            if _backend is None:
                raise RuntimeError('Failed to load translation model')
        else:
            with metrics.span('load_model') as s:
                _translator, cached = _get_translator(
                    ctx.config.device, ctx.config.batch_size, ctx.config.model
                )
                s.detail('cached', cached)
            if _translator is None:
                raise RuntimeError('Failed to load translation model')

        def _translate():
            with metrics.span('batch') as s:
                input_texts = [line.text for line in dialogue_lines]
                s.detail('input_lines', len(input_texts))
                s.detail('input_chars', sum(len(t) for t in input_texts))
                s.detail('batch_size', ctx.config.batch_size)
                translated = translate_dialogue_lines(
                    dialogue_lines,
                    ctx.config.device,
                    ctx.config.batch_size,
                    ctx.config.model,
                    progress_callback=_on_progress,
                )
                if translated:
                    s.detail('output_lines', len(translated))
                    s.detail('output_chars', sum(len(line.text) for line in translated))
                    s.detail(
                        'batches',
                        (len(input_texts) + ctx.config.batch_size - 1) // ctx.config.batch_size,
                    )
                return translated

        def _on_progress(lines_done: int, total_lines: int, rate: float) -> None:
            if tracker:
                tracker.set_stage_progress(lines_done, total_lines, rate)

        with ThreadPoolExecutor(max_workers=2) as pool:
            ctx_fonts = copy_context()
            ctx_translate = copy_context()
            font_future = pool.submit(ctx_fonts.run, self.check_fonts, ctx)
            translate_future = pool.submit(ctx_translate.run, _translate)

            font_info: FontInfo = font_future.result()  # ty: ignore[invalid-assignment]
            ctx.font_info = font_info
            translated: list = translate_future.result()  # ty: ignore[invalid-assignment]

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
