#!/usr/bin/env python3
"""
Main orchestration module for the Movie Translator pipeline.
Handles the complete workflow from input to output.
"""

import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ai_translation import translate_dialogue_lines
from .mkv_operations import create_clean_mkv, verify_result
from .subtitle_processor import (
    check_dependencies,
    create_clean_english_ass,
    create_polish_ass,
    extract_dialogue_lines,
    extract_subtitle,
    find_english_subtitle_track,
    get_subtitle_extension,
    get_track_info,
    process_image_based_subtitles,
)
from .subtitle_validator import validate_cleaned_subtitles
from .utils import log_error, log_info, log_success, log_warning

console = Console()


def process_mkv_file(
    mkv_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    model: str = 'allegro',
    enable_ocr: bool = False,
    ocr_gpu: bool = False,
) -> bool:
    """Process a single MKV file and replace it with clean version."""
    console.print(Panel(f'[bold blue]Processing: {mkv_path.name}[/bold blue]', border_style='blue'))

    # Create temp directory
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # Step 1: Extract English subtitles
    log_info('📖 Step 1: Extracting English subtitles...')
    track_info = get_track_info(mkv_path)
    if not track_info:
        log_error('Could not read track information')
        return False

    eng_track = find_english_subtitle_track(track_info, enable_ocr)
    if not eng_track:
        log_warning(
            'No suitable English subtitle track found (only image-based tracks or no tracks available)'
        )
        return False

    track_id = eng_track['id']
    track_name = eng_track.get('properties', {}).get('track_name', 'Unknown')
    log_info(f"Found English track: ID {track_id}, Name: '{track_name}'")

    # Check if this track requires OCR processing
    requires_ocr = eng_track.get('requires_ocr', False)

    if requires_ocr:
        log_info('🤖 Processing image-based subtitles with OCR...')

        # Process image-based subtitles
        extracted_srt = process_image_based_subtitles(mkv_path, track_id, output_dir, ocr_gpu)
        if not extracted_srt:
            log_error('OCR processing failed')
            return False

        # For OCR processing, we'll create a simple text-based subtitle file
        # In a full implementation, this would have proper timing from OCR
        extracted_ass = extracted_srt  # Use the OCR-generated SRT
    else:
        # Normal text-based subtitle extraction
        subtitle_ext = get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{mkv_path.stem}_extracted{subtitle_ext}'

        if not extract_subtitle(mkv_path, track_id, extracted_ass):
            return False

    # Step 2: Extract dialogue lines
    log_info('🔍 Step 2: Extracting dialogue lines...')
    dialogue_lines = extract_dialogue_lines(extracted_ass)
    if not dialogue_lines:
        log_error('No dialogue lines found')
        return False

    # Step 3: AI translate
    log_info('🤖 Step 3: AI translating to Polish...')
    try:
        translated_dialogue = translate_dialogue_lines(dialogue_lines, device, batch_size, model)
        if not translated_dialogue:
            log_error('AI translation failed')
            return False
        log_success('   ✅ AI translation complete!')
    except Exception as e:
        log_error(f'AI translation failed: {e}')
        return False

    # Step 4: Create clean ASS files
    log_info('🔨 Step 4: Creating clean subtitle files...')
    # Keep subtitle files in main output directory (not temp) so they won't be deleted
    clean_english_ass = output_dir / f'{mkv_path.stem}_english_clean.ass'
    polish_ass = output_dir / f'{mkv_path.stem}_polish.ass'

    create_clean_english_ass(extracted_ass, dialogue_lines, clean_english_ass)

    # Validate that cleaned subtitles have correct timestamps
    extracted_ass_path = output_dir / f'{mkv_path.stem}_extracted.ass'
    if not validate_cleaned_subtitles(extracted_ass_path, clean_english_ass):
        log_error('❌ Validation failed! Cleaned subtitles have timestamp mismatches.')
        return False

    create_polish_ass(extracted_ass, translated_dialogue, polish_ass)

    # Step 5: Create clean MKV (replace original)
    log_info('🎬 Step 5: Creating clean MKV to replace original...')

    # Create temporary file first
    temp_mkv = temp_dir / f'{mkv_path.stem}_temp_clean.mkv'

    if not create_clean_mkv(mkv_path, clean_english_ass, polish_ass, temp_mkv):
        return False

    # Step 6: Verify temporary file
    if not verify_result(temp_mkv):
        return False

    # Step 7: Replace original file
    log_info('🔄 Step 7: Replacing original MKV...')
    try:
        # Create backup of original (just in case)
        backup_path = mkv_path.with_suffix('.mkv.backup')
        shutil.copy2(mkv_path, backup_path)
        log_info(f'   - Created backup: {backup_path.name}')

        # Replace original with clean version
        shutil.move(str(temp_mkv), str(mkv_path))
        log_success(f'   - Replaced original: {mkv_path.name}')

        # Verify the replacement worked
        if not verify_result(mkv_path):
            log_error('   - Verification failed after replacement, restoring backup')
            shutil.move(str(backup_path), str(mkv_path))
            return False

        # Remove backup if everything is good
        backup_path.unlink()
        log_info('   - Removed backup (verification successful)')

    except Exception as e:
        log_error(f'   - Failed to replace original: {e}')
        return False

    # Clean up temp files
    shutil.rmtree(temp_dir)
    log_info('🧹 Cleaned up temporary files')

    log_success(f'🎉 SUCCESS! Original MKV replaced with clean version: {mkv_path.name}')
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original MKV'
    )
    parser.add_argument('input_dir', help='Directory containing MKV files')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        default='mps' if sys.platform == 'darwin' else 'cuda',
        help='Device to use for AI translation (default: mps on macOS, cuda on others)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=16, help='Batch size for AI translation (default: 16)'
    )
    parser.add_argument(
        '--model',
        choices=['allegro', 'flan-t5', 'mbart', 'nllb'],
        default='allegro',
        help='Translation model to use (default: allegro)',
    )
    parser.add_argument(
        '--enable-ocr', action='store_true', help='Enable OCR for image-based subtitles'
    )
    parser.add_argument('--ocr-gpu', action='store_true', help='Use GPU for OCR processing')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log_error(f'Input directory does not exist: {input_dir}')
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Find MKV files
    mkv_files = list(input_dir.glob('*.mkv'))
    if not mkv_files:
        log_error(f'No MKV files found in {input_dir}')
        sys.exit(1)

    # Create output directory
    output_dir = input_dir / 'translated'
    output_dir.mkdir(exist_ok=True)

    # Show configuration
    console.print(
        Panel.fit(
            f'[bold blue]Movie Translator - Final Pipeline[/bold blue]\n'
            f'  Input        {input_dir}\n'
            f'  Output       {output_dir}\n'
            f'  Device       {args.device}\n'
            f'  Batch Size   {args.batch_size}\n'
            f'  Model        {args.model}\n'
            f'  OCR Enabled  {args.enable_ocr}',
            border_style='blue',
        )
    )

    log_info(f'Found {len(mkv_files)} MKV file(s)')

    # Process each file
    successful = 0
    failed = 0

    for mkv_path in mkv_files:
        if process_mkv_file(
            mkv_path,
            output_dir,
            args.device,
            args.batch_size,
            args.model,
            args.enable_ocr,
            args.ocr_gpu,
        ):
            successful += 1
        else:
            failed += 1

    # Show results
    table = Table(title='Translation Results')
    table.add_column('Status', style='green')
    table.add_column('Count', justify='right')

    table.add_row('✅ Successful', str(successful))
    if failed > 0:
        table.add_row('❌ Failed', str(failed))
    table.add_row('📁 Total', str(len(mkv_files)))

    console.print(table)

    if failed == 0:
        console.print(
            Panel(
                '[bold green]🎉 All files processed successfully![/bold green]\n'
                '🎬 Clean MKVs with English dialogue + Polish translation created',
                border_style='green',
            )
        )
        console.print(f'📁 Output directory: {output_dir}')
    else:
        console.print(
            Panel(
                '[bold yellow]⚠️  Some files failed to process.[/bold yellow]', border_style='yellow'
            )
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
