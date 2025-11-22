"""Simple SRT subtitle translator - English to Polish.

Translates SRT subtitle files using local LLM models optimized for M1 MacBook Air.
Focus: Fast, local translation without external dependencies.
"""
import argparse
import logging
import sys
from pathlib import Path

from movie_translator.local_llm_provider import translate_file_local
from movie_translator.config import setup_logging
from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
)
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for subtitle translation."""
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles from English to Polish using local LLM (M1 optimized).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
  # Basic translation
  srt-translate input.srt output.srt

  # With M1 acceleration (default)
  srt-translate input.srt output.srt --device auto

  # CPU only (slower but more compatible)
  srt-translate input.srt output.srt --device cpu

  # Custom model
  srt-translate input.srt output.srt --model allegro/p5-eng2many

  # Larger batch for speed (if you have RAM)
  srt-translate input.srt output.srt --batch-size 32
""",
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input SRT file (English subtitles)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output SRT file (Polish translation)"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Translation model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=DEFAULT_DEVICE,
        help=f"Device: 'auto' detects M1/MPS, 'cpu' for compatibility (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Subtitle lines per batch - higher is faster but uses more memory (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()
    setup_logging()

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.suffix.lower() == ".srt":
        logger.error(f"Input file must be .srt format: {args.input}")
        sys.exit(1)

    # Validate output path
    if args.output.exists():
        logger.warning(f"Output file already exists and will be overwritten: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Translate
    try:
        logger.info(f"Translating: {args.input.name} → {args.output.name}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Batch size: {args.batch_size}")

        translate_file_local(
            input_path=str(args.input),
            output_path=str(args.output),
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )

        if not args.output.exists():
            logger.error("Translation failed - no output file created")
            sys.exit(1)

        logger.info(f"✓ Translation complete: {args.output}")

    except TranslationError as e:
        logger.error(f"Translation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nTranslation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
