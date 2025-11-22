import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from movie_translator.local_llm_provider import translate_file_local
from movie_translator.config import setup_logging, TranslationConfig
from movie_translator.constants import (
    EXTENSION_MKV,
    TRACK_TYPE_SUBTITLE,
    LANGUAGE_ENGLISH,
    LANGUAGE_ENGLISH_SHORT,
    LANGUAGE_POLISH,
    POLISH_TRACK_NAME,
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
)
from movie_translator.exceptions import (
    SubtitleNotFoundError,
    TranslationError,
    MKVProcessingError,
)

logger = logging.getLogger(__name__)


class MovieProcessor:
    def __init__(self, directory: Path, config: TranslationConfig):
        self.directory = directory
        self.config = config

    def run(self) -> None:
        if not self.directory.exists():
            error_msg = f"Directory not found: {self.directory}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        mkv_files = list(self.directory.glob(f"*{EXTENSION_MKV}"))
        if not mkv_files:
            logger.warning(f"No MKV files found in {self.directory}")
            return

        logger.info(f"Found {len(mkv_files)} MKV files.")

        for mkv_file in mkv_files:
            try:
                self.process_file(mkv_file)
            except (SubtitleNotFoundError, TranslationError, MKVProcessingError) as e:
                logger.error(f"Failed to process {mkv_file.name}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error processing {mkv_file.name}: {e}", exc_info=True
                )

    def process_file(self, mkv_path: Path) -> None:
        logger.info(f"Processing {mkv_path.name}...")
        track_info = self.get_track_info(mkv_path)
        eng_sub_track = self.find_english_subtitle_track(track_info)

        if not eng_sub_track:
            error_msg = f"No English subtitle track found in {mkv_path.name}"
            logger.warning(error_msg)
            raise SubtitleNotFoundError(error_msg)

        track_id = eng_sub_track["id"]
        logger.info(f"Found English subtitle track: ID {track_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            extracted_srt = temp_dir_path / "extracted.srt"
            translated_srt = temp_dir_path / "translated.srt"
            output_mkv = temp_dir_path / "output.mkv"

            self.extract_subtitle(mkv_path, track_id, extracted_srt)

            logger.info(f"Translating subtitles for {mkv_path.name}...")
            translate_file_local(
                input_path=str(extracted_srt),
                output_path=str(translated_srt),
                model_name=self.config.model,
                target_language=self.config.target_language,
                device=self.config.device,
                batch_size=self.config.batch_size,
            )

            if not translated_srt.exists():
                raise TranslationError("Translation did not produce output file")

            self.merge_subtitle(mkv_path, translated_srt, output_mkv)

            logger.info(f"Replacing original file {mkv_path.name}...")
            shutil.move(str(output_mkv), str(mkv_path))
            logger.info(f"Successfully processed {mkv_path.name}")

    def get_track_info(self, mkv_path: Path) -> Dict[str, Any]:
        cmd = ["mkvmerge", "-J", str(mkv_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise MKVProcessingError(
                f"Failed to get track info from {mkv_path.name}"
            ) from e
        except json.JSONDecodeError as e:
            raise MKVProcessingError(
                f"Failed to parse track info from {mkv_path.name}"
            ) from e

    def find_english_subtitle_track(
        self, track_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        tracks = track_info.get("tracks", [])
        for track in tracks:
            if track.get("type") == TRACK_TYPE_SUBTITLE:
                props = track.get("properties", {})
                lang = props.get("language", "")
                if lang in [LANGUAGE_ENGLISH, LANGUAGE_ENGLISH_SHORT]:
                    return track
        return None

    def extract_subtitle(
        self, mkv_path: Path, track_id: int, output_path: Path
    ) -> None:
        logger.info(f"Extracting track {track_id} to {output_path.name}...")
        cmd = ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{output_path}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise MKVProcessingError(
                f"Failed to extract subtitle track {track_id}"
            ) from e

    def merge_subtitle(self, mkv_path: Path, srt_path: Path, output_path: Path) -> None:
        logger.info(f"Merging translated subtitles into {output_path.name}...")
        cmd = [
            "mkvmerge",
            "-o",
            str(output_path),
            str(mkv_path),
            "--language",
            "0:pl",
            "--track-name",
            f"0:{POLISH_TRACK_NAME}",
            str(srt_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise MKVProcessingError(
                f"Failed to merge subtitle into {mkv_path.name}"
            ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate MKV subtitles using a local translation model (optimized for M1 MacBook Air).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
  # Using default model (Helsinki-NLP/opus-mt-en-pl, 78MB, fast)
  movie-translator /path/to/movies

  # Force CPU mode (slower)
  movie-translator /path/to/movies --device cpu

  # Using a different model
  movie-translator /path/to/movies --model Helsinki-NLP/opus-mt-en-de

  # Adjust batch size (higher = faster but more memory)
  movie-translator /path/to/movies --batch-size 32
""",
    )
    parser.add_argument(
        "directory", type=Path, help="Directory containing MKV files to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Local translation model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default=LANGUAGE_POLISH,
        help=f"Target language for translation (default: {LANGUAGE_POLISH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=DEFAULT_DEVICE,
        help=f"Device for local LLM - 'auto' detects M1/MPS automatically (default: {DEFAULT_DEVICE})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of subtitle lines to translate per batch (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    setup_logging()

    try:
        config = TranslationConfig(
            model=args.model,
            target_language=args.target_language,
            device=args.device,
            batch_size=args.batch_size,
        )
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    # Run processor
    try:
        processor = MovieProcessor(args.directory, config)
        processor.run()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
