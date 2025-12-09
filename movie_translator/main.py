"""Main entry point for the Movie Translator CLI."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ffmpeg import SUPPORTED_VIDEO_EXTENSIONS, get_ffmpeg_version
from .pipeline import TranslationPipeline
from .utils import log_error, log_info

console = Console()


def check_dependencies():
    """Check if all required dependencies are available."""
    console.print(Panel.fit('[bold blue]Dependency Check[/bold blue]', border_style='blue'))

    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        log_error(f'Python 3.10+ required, found {version.major}.{version.minor}')
        sys.exit(1)
    log_info(f'Python: {version.major}.{version.minor}.{version.micro}')

    # Check ffmpeg (via static-ffmpeg)
    try:
        ffmpeg_version = get_ffmpeg_version()
        log_info(f'FFmpeg: {ffmpeg_version}')
    except Exception as e:
        log_error(f'FFmpeg not available: {e}')
        log_info('FFmpeg should be installed automatically via static-ffmpeg package')
        sys.exit(1)

    # Check Python packages
    _check_python_packages()

    console.print(
        Panel('[bold green]✅ All dependencies satisfied[/bold green]', border_style='green')
    )


def _check_python_packages():
    """Check required Python packages are installed."""
    import importlib.util

    required_packages = ['pysubs2', 'torch', 'transformers']
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        log_error(f'Missing Python packages: {", ".join(missing_packages)}')
        log_info('Install with: uv add pysubs2 torch transformers')
        sys.exit(1)

    log_info('Python packages: pysubs2, torch, transformers')
    log_info('OCR support: Will be checked when --enable-ocr is used')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Movie Translator - Extract English dialogue → AI translate to Polish → Replace original video'
    )
    parser.add_argument('input_dir', help='Directory containing video files (MKV, MP4, etc.)')
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
        choices=['allegro', 'mbart'],
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
    return parser.parse_args()


def show_config(args: argparse.Namespace, input_dir: Path, output_dir: Path):
    """Display configuration panel."""
    console.print(
        Panel.fit(
            f'[bold blue]Movie Translator[/bold blue]\n'
            f'  Input        {input_dir}\n'
            f'  Output       {output_dir}\n'
            f'  Device       {args.device}\n'
            f'  Batch Size   {args.batch_size}\n'
            f'  Model        {args.model}\n'
            f'  OCR Enabled  {args.enable_ocr}',
            border_style='blue',
        )
    )


def show_results(successful: int, failed: int, total: int):
    """Display results table."""
    table = Table(title='Translation Results')
    table.add_column('Status', style='green')
    table.add_column('Count', justify='right')

    table.add_row('✅ Successful', str(successful))
    if failed > 0:
        table.add_row('❌ Failed', str(failed))
    table.add_row('📁 Total', str(total))

    console.print(table)

    if failed == 0:
        console.print(
            Panel(
                '[bold green]🎉 All files processed successfully![/bold green]\n'
                '🎬 Clean MKVs with English dialogue + Polish translation created',
                border_style='green',
            )
        )
    else:
        console.print(
            Panel(
                '[bold yellow]⚠️  Some files failed to process.[/bold yellow]', border_style='yellow'
            )
        )


def main():
    """Main entry point."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log_error(f'Input directory does not exist: {input_dir}')
        sys.exit(1)

    check_dependencies()

    # Find video files (all supported formats)
    video_files = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(f'*{ext}'))

    if not video_files:
        supported = ', '.join(SUPPORTED_VIDEO_EXTENSIONS)
        log_error(f'No video files found in {input_dir}')
        log_info(f'Supported formats: {supported}')
        sys.exit(1)

    # Create output directory
    output_dir = input_dir / 'translated'
    output_dir.mkdir(exist_ok=True)

    show_config(args, input_dir, output_dir)
    log_info(f'Found {len(video_files)} video file(s)')

    # Create pipeline
    pipeline = TranslationPipeline(
        device=args.device,
        batch_size=args.batch_size,
        model=args.model,
        enable_ocr=args.enable_ocr,
        ocr_gpu=args.ocr_gpu,
    )

    # Process files
    successful = 0
    failed = 0

    for video_path in video_files:
        if pipeline.process_video_file(video_path, output_dir):
            successful += 1
        else:
            failed += 1

    show_results(successful, failed, len(video_files))

    if failed == 0:
        console.print(f'📁 Output directory: {output_dir}')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
