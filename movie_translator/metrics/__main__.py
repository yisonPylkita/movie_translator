"""CLI entry point: python -m movie_translator.metrics compare <before> <after>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare_reports, format_comparison
from .report import load_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m movie_translator.metrics',
        description='Metrics comparison tool for movie-translator',
    )
    subparsers = parser.add_subparsers(dest='command')

    compare_parser = subparsers.add_parser('compare', help='Compare two metric reports')
    compare_parser.add_argument('before', type=Path, help='Path to the before report JSON')
    compare_parser.add_argument('after', type=Path, help='Path to the after report JSON')

    args = parser.parse_args()

    if args.command == 'compare':
        before = load_report(args.before)
        after = load_report(args.after)
        result = compare_reports(before, after)
        print(format_comparison(before, after, result))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
