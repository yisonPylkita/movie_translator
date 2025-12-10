import argparse
import sys
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .ffmpeg import get_ffmpeg_version
from .logging import console, logger
from .pipeline import TranslationPipeline
from .subtitles import SubtitleExtractor


def check_dependencies():
    console.print(Panel.fit('[bold blue]Dependency Check[/bold blue]', border_style='blue'))

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f'Python 3.10+ required, found {version.major}.{version.minor}')
        sys.exit(1)
    logger.info(f'Python: {version.major}.{version.minor}.{version.micro}')

    try:
        ffmpeg_version = get_ffmpeg_version()
        logger.info(f'FFmpeg: {ffmpeg_version}')
    except Exception as e:
        logger.error(f'FFmpeg not available: {e}')
        logger.info('FFmpeg should be installed automatically via static-ffmpeg package')
        sys.exit(1)

    _check_python_packages()

    console.print(
        Panel('[bold green]✅ All dependencies satisfied[/bold green]', border_style='green')
    )


def _check_python_packages():
    import importlib.util

    required_packages = ['pysubs2', 'torch', 'transformers']
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f'Missing Python packages: {", ".join(missing_packages)}')
        logger.info('Install with: uv add pysubs2 torch transformers')
        sys.exit(1)

    logger.info('Python packages: pysubs2, torch, transformers')
    logger.info('OCR support: Will be checked when --enable-ocr is used')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original video'
    )
    parser.add_argument('input_dir', help='Directory containing MKV files')
    parser.add_argument(
        '--device',
        choices=['cpu', 'mps'],
        default='mps' if sys.platform == 'darwin' else 'cpu',
        help='Device to use for AI translation (default: mps on macOS, cpu on others)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for AI translation (default: 16)',
    )
    parser.add_argument(
        '--model',
        choices=['allegro'],
        default='allegro',
        help='Translation model to use (default: allegro)',
    )
    parser.add_argument(
        '--enable-ocr',
        action='store_true',
        help='Enable OCR for image-based subtitles',
    )
    parser.add_argument(
        '--ocr-gpu',
        action='store_true',
        help='Use GPU for OCR processing',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process files but do not replace originals (keeps output in temp directory)',
    )
    return parser.parse_args()


def show_config(args: argparse.Namespace, input_dir: Path):
    dry_run_status = (
        '[bold yellow]YES (originals will NOT be modified)[/bold yellow]' if args.dry_run else 'No'
    )
    console.print(
        Panel.fit(
            f'[bold blue]Movie Translator[/bold blue]\n'
            f'  Input        {input_dir}\n'
            f'  Device       {args.device}\n'
            f'  Batch Size   {args.batch_size}\n'
            f'  Model        {args.model}\n'
            f'  OCR Enabled  {args.enable_ocr}\n'
            f'  Dry Run      {dry_run_status}',
            border_style='blue',
        )
    )


def show_results(
    results: list[tuple[str, str]],
    dry_run: bool = False,
):
    successful = sum(1 for _, status in results if status == 'success')
    failed = sum(1 for _, status in results if status == 'failed')
    skipped = sum(1 for _, status in results if status == 'skipped')
    total = len(results)

    summary_table = Table(title='Summary')
    summary_table.add_column('Status', style='green')
    summary_table.add_column('Count', justify='right')

    summary_table.add_row('✅ Successful', str(successful))
    if skipped > 0:
        summary_table.add_row('⏭️  Skipped', str(skipped))
    if failed > 0:
        summary_table.add_row('❌ Failed', str(failed))
    summary_table.add_row('📁 Total', str(total))

    console.print(summary_table)

    if len(results) > 1:
        details_table = Table(title='File Details')
        details_table.add_column('File', style='cyan')
        details_table.add_column('Status', justify='right')

        for filename, status in results:
            if status == 'success':
                status_str = '[green]✅ Success[/green]'
            elif status == 'skipped':
                status_str = '[blue]⏭️  Skipped[/blue]'
            else:
                status_str = '[red]❌ Failed[/red]'
            details_table.add_row(filename, status_str)

        console.print(details_table)

    if failed == 0 and successful > 0:
        if dry_run:
            console.print(
                Panel(
                    '[bold green]🏁 DRY RUN complete![/bold green]\n'
                    '🎬 Output files are in the temp directory. Originals were NOT modified.',
                    border_style='green',
                )
            )
        else:
            console.print(
                Panel(
                    '[bold green]🎉 All files processed successfully![/bold green]\n'
                    '🎬 Clean videos with English dialogue + Polish translation created',
                    border_style='green',
                )
            )
    elif failed == 0 and successful == 0 and skipped > 0:
        console.print(
            Panel(
                '[bold blue]ℹ️  All files already have Polish subtitles.[/bold blue]',
                border_style='blue',
            )
        )
    elif failed > 0:
        console.print(
            Panel(
                '[bold yellow]⚠️  Some files failed to process.[/bold yellow]',
                border_style='yellow',
            )
        )


def find_mkv_files_with_temp_dirs(input_dir: Path) -> list[tuple[Path, Path]]:
    mkv_files_direct = sorted(input_dir.glob('*.mkv'))
    if mkv_files_direct:
        temp_dir = input_dir / '.translate_temp'
        temp_dir.mkdir(exist_ok=True)
        return [(mkv, temp_dir) for mkv in mkv_files_direct]

    results: list[tuple[Path, Path]] = []
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            mkv_files_in_subdir = sorted(subdir.glob('*.mkv'))
            if mkv_files_in_subdir:
                temp_dir = subdir / '.translate_temp'
                temp_dir.mkdir(exist_ok=True)
                results.extend((mkv, temp_dir) for mkv in mkv_files_in_subdir)

    return results


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f'Input directory does not exist: {input_dir}')
        sys.exit(1)

    check_dependencies()

    mkv_files_with_temps = find_mkv_files_with_temp_dirs(input_dir)

    if not mkv_files_with_temps:
        logger.error(f'No MKV files found in {input_dir} or its subdirectories')
        sys.exit(1)

    show_config(args, input_dir)
    logger.info(f'Found {len(mkv_files_with_temps)} MKV file(s)')

    if any(mkv.parent != input_dir for mkv, _ in mkv_files_with_temps):
        subdirs = sorted({mkv.parent.name for mkv, _ in mkv_files_with_temps})
        logger.info(f'Scanning subdirectories: {", ".join(subdirs)}')

    pipeline = TranslationPipeline(
        device=args.device,
        batch_size=args.batch_size,
        model=args.model,
        enable_ocr=args.enable_ocr,
        ocr_gpu=args.ocr_gpu,
    )

    extractor = SubtitleExtractor()

    results: list[tuple[str, str]] = []
    total_files = len(mkv_files_with_temps)

    for idx, (mkv_path, temp_dir) in enumerate(mkv_files_with_temps, start=1):
        relative_name = (
            f'{mkv_path.parent.name}/{mkv_path.name}'
            if mkv_path.parent != input_dir
            else mkv_path.name
        )

        console.print(
            f'\n[bold cyan]📊 Progress: {idx}/{total_files}[/bold cyan]',
            highlight=False,
        )

        if extractor.has_polish_subtitles(mkv_path):
            logger.info(f'⏭️  Skipping {relative_name} - already has Polish subtitles')
            results.append((relative_name, 'skipped'))
            continue

        if pipeline.process_video_file(mkv_path, temp_dir, dry_run=args.dry_run):
            results.append((relative_name, 'success'))
        else:
            results.append((relative_name, 'failed'))

    show_results(results, dry_run=args.dry_run)

    failed = sum(1 for _, status in results if status == 'failed')
    if failed == 0:
        temp_dirs = sorted({temp_dir for _, temp_dir in mkv_files_with_temps})
        for temp_dir in temp_dirs:
            console.print(f'📁 Temp files kept in: {temp_dir}')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
