"""CLI handler for the 'extract' subcommand."""

import sys
from pathlib import Path

from ..logging import console, set_verbose


def parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(
        prog='movie-translator extract',
        description='Extract subtitles from video files (text tracks + OCR for burned-in)',
    )
    parser.add_argument('input', help='Video file or directory containing video files')
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for SRTs and manifest (default: <input_dir>/extracted_subs/)',
    )
    parser.add_argument(
        '--ocr-language',
        default='pl',
        help='Language hint for burned-in subtitle OCR (default: pl)',
    )
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    """Entry point for the extract subcommand."""
    args = parse_args(argv)
    set_verbose(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f'[red]❌ Not found: {input_path}[/red]')
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        root = input_path if input_path.is_dir() else input_path.parent
        output_dir = root / 'extracted_subs'

    from ..extract import run_extract

    run_extract(input_path, output_dir, ocr_language=args.ocr_language)
