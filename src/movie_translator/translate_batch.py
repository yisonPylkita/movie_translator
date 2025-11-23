"""Batch translate multiple SRT files in a directory."""

import argparse
import logging
import sys
from pathlib import Path

from movie_translator.config import setup_logging
from movie_translator.local_llm_provider import translate_file
from movie_translator.constants import (
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
)
from movie_translator.exceptions import TranslationError

logger = logging.getLogger(__name__)


def translate_directory(
    directory: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pattern: str = "*_en.srt",
) -> None:
    """Translate all English SRT files in directory to Polish.

    Args:
        directory: Directory containing SRT files
        model_name: Translation model to use
        device: Device to run on (auto/cpu/cuda/mps)
        batch_size: Number of lines per batch
        pattern: File pattern to match (default: *_en.srt)
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all English subtitle files
    srt_files = list(directory.glob(pattern))
    if not srt_files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return

    logger.info(f"Found {len(srt_files)} file(s) to translate")
    translated_count = 0
    skipped_count = 0
    failed_count = 0

    for srt_file in srt_files:
        try:
            # Generate output filename: movie_en.srt -> movie_pl.srt
            output_srt = srt_file.with_name(srt_file.name.replace("_en.srt", "_pl.srt"))

            # Skip if already translated
            if output_srt.exists():
                logger.info(f"Skipping {srt_file.name} (already translated)")
                skipped_count += 1
                continue

            logger.info(f"Translating: {srt_file.name} → {output_srt.name}")

            translate_file(
                input_path=str(srt_file),
                output_path=str(output_srt),
                model_name=model_name,
                device=device,
                batch_size=batch_size,
            )

            if output_srt.exists():
                logger.info(f"  ✓ Translation complete: {output_srt.name}")
                translated_count += 1
            else:
                logger.error(f"  ✗ Translation failed: no output file created")
                failed_count += 1

        except TranslationError as e:
            logger.error(f"Failed to translate {srt_file.name}: {e}")
            failed_count += 1
        except Exception as e:
            logger.error(
                f"Unexpected error translating {srt_file.name}: {e}", exc_info=True
            )
            failed_count += 1

    logger.info(
        f"✓ Translation complete: {translated_count} translated, "
        f"{skipped_count} skipped, {failed_count} failed"
    )


def main() -> None:
    """Main entry point for batch translation."""
    parser = argparse.ArgumentParser(
        description="Batch translate SRT subtitles (English to Polish)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing *_en.srt files to translate",
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
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_en.srt",
        help="File pattern to match (default: *_en.srt)",
    )

    args = parser.parse_args()
    setup_logging()

    if not args.directory.exists():
        logger.error(f"Directory not found: {args.directory}")
        sys.exit(1)

    if not args.directory.is_dir():
        logger.error(f"Not a directory: {args.directory}")
        sys.exit(1)

    try:
        logger.info(f"Batch translating files in: {args.directory}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Batch size: {args.batch_size}")

        translate_directory(
            directory=args.directory,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            pattern=args.pattern,
        )
    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
