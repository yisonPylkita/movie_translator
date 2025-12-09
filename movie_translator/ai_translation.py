#!/usr/bin/env python3
"""
AI translation interface for the Movie Translator pipeline.
Handles AI model loading, translation, and cleanup.
"""

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ai_translator import SubtitleTranslator as AITranslator

from .utils import clear_memory, log_error, log_info


def translate_dialogue_lines(
    dialogue_lines: list[tuple[int, int, str]], device: str, batch_size: int, model: str = 'allegro'
) -> list[tuple[int, int, str]]:
    """Translate dialogue lines using AI with Rich progress bar."""
    from rich.console import Console

    console = Console()
    log_info('🤖 Step 3: AI translating to Polish...')

    translator = AITranslator(device=device, batch_size=batch_size, model_name=model)
    log_info('   - AI Translator initialized')

    if not translator.load_model():
        log_error('❌ Failed to load translation model')
        return []

    log_info('   - Model loaded')

    texts = [text for _, _, text in dialogue_lines]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TextColumn('{task.fields[rate]}'),
        console=console,
    ) as progress:
        task = progress.add_task(
            f'[cyan]Translating {len(texts)} texts...[/cyan]',
            total=total_batches,
            rate='0.0 lines/s',
        )

        def progress_callback(batch_num, total_batches, rate, error=None):
            if error:
                progress.update(task, advance=1, rate=f'❌ {error}')
            else:
                progress.update(task, advance=1, rate=f'{rate:.1f} lines/s')

        # Translate with progress callback
        translated_texts = translator.translate_texts(texts, progress_callback=progress_callback)

    log_info('   - Translation complete')

    # Cleanup
    translator.cleanup()
    log_info('   - Translator cleaned up')

    # Force garbage collection
    clear_memory()
    log_info('   - Final cleanup')

    # Reconstruct with timing
    translated_lines = []
    for (start, end, _), translated_text in zip(dialogue_lines, translated_texts, strict=True):
        translated_lines.append((start, end, translated_text))

    return translated_lines
