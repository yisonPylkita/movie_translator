import argparse
import logging
import sys
from pathlib import Path

from movie_translator.local_llm_provider import translate_file
from movie_translator.config import setup_logging
from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
)
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles (English to Polish)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=Path, help="Input SRT file (English subtitles)")
    parser.add_argument(
        "output", type=Path, help="Output SRT file (Polish translation)"
    )

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

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.suffix.lower() == ".srt":
        logger.error(f"Input file must be .srt format: {args.input}")
        sys.exit(1)

    if args.output.exists():
        logger.warning(
            f"Output file already exists and will be overwritten: {args.output}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Translating: {args.input.name} → {args.output.name}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Batch size: {args.batch_size}")

        translate_file(
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
        logger.warning("Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
