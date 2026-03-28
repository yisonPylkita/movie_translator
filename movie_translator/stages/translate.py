"""AI translation and font checking stage."""

from concurrent.futures import ThreadPoolExecutor

from ..context import FontInfo, PipelineContext
from ..fonts import (
    check_embedded_fonts_support_polish,
    find_system_font_for_polish,
    get_ass_font_names,
)
from ..logging import logger
from ..translation import translate_dialogue_lines


class TranslateStage:
    name = 'translate'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info(f'Translating {len(ctx.dialogue_lines)} lines...')

        def _check_fonts():
            supports = check_embedded_fonts_support_polish(ctx.video_path, ctx.english_source)
            if supports:
                return FontInfo(supports_polish=True)
            is_mkv = ctx.video_path.suffix.lower() == '.mkv'
            if is_mkv:
                names = get_ass_font_names(ctx.english_source)
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

        with ThreadPoolExecutor(max_workers=2) as pool:
            font_future = pool.submit(_check_fonts)
            translate_future = pool.submit(
                translate_dialogue_lines,
                ctx.dialogue_lines,
                ctx.config.device,
                ctx.config.batch_size,
                ctx.config.model,
            )

            ctx.font_info = font_future.result()
            translated = translate_future.result()

        if not translated:
            raise RuntimeError('Translation failed — empty result')

        ctx.translated_lines = translated
        return ctx
