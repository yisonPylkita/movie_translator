import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from movie_translator.local_llm_provider import translate_file_local
from movie_translator.constants import (
    EXTENSION_MKV,
    TRACK_TYPE_SUBTITLE,
    LANGUAGE_ENGLISH,
    LANGUAGE_ENGLISH_SHORT,
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


class MKVProcessor:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def process_file(self, mkv_path: Path) -> None:
        logger.info(f"Processing {mkv_path.name}...")

        track_info = self._get_track_info(mkv_path)
        eng_sub_track = self._find_english_subtitle_track(track_info)

        if not eng_sub_track:
            raise SubtitleNotFoundError(
                f"No English subtitle track found in {mkv_path.name}"
            )

        track_id = eng_sub_track["id"]
        logger.info(f"Found English subtitle track: ID {track_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            extracted_srt = temp_dir_path / "extracted.srt"
            translated_srt = temp_dir_path / "translated.srt"
            output_mkv = temp_dir_path / "output.mkv"

            self._extract_subtitle(mkv_path, track_id, extracted_srt)

            logger.info("Translating subtitles...")
            translate_file_local(
                input_path=str(extracted_srt),
                output_path=str(translated_srt),
                model_name=self.model,
                device=self.device,
                batch_size=self.batch_size,
            )

            if not translated_srt.exists():
                raise TranslationError("Translation did not produce output file")

            self._merge_subtitle(mkv_path, translated_srt, output_mkv)

            logger.info("Replacing original file...")
            shutil.move(str(output_mkv), str(mkv_path))
            logger.info(f"✓ Successfully processed {mkv_path.name}")

    def process_directory(self, directory: Path) -> None:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        mkv_files = list(directory.glob(f"*{EXTENSION_MKV}"))
        if not mkv_files:
            logger.warning(f"No MKV files found in {directory}")
            return

        logger.info(f"Found {len(mkv_files)} MKV files")

        for mkv_file in mkv_files:
            try:
                self.process_file(mkv_file)
            except (SubtitleNotFoundError, TranslationError, MKVProcessingError) as e:
                logger.error(f"Failed to process {mkv_file.name}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error processing {mkv_file.name}: {e}", exc_info=True
                )

    def _get_track_info(self, mkv_path: Path) -> Dict[str, Any]:
        cmd = ["mkvmerge", "-J", str(mkv_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise MKVProcessingError(
                f"Failed to get track info from {mkv_path.name}. Is mkvmerge installed?"
            ) from e
        except json.JSONDecodeError as e:
            raise MKVProcessingError(
                f"Failed to parse track info from {mkv_path.name}"
            ) from e

    def _find_english_subtitle_track(
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

    def _extract_subtitle(
        self, mkv_path: Path, track_id: int, output_path: Path
    ) -> None:
        logger.info(f"Extracting subtitle track {track_id}...")
        cmd = ["mkvextract", "tracks", str(mkv_path), f"{track_id}:{output_path}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise MKVProcessingError(
                f"Failed to extract subtitle track {track_id}. Is mkvextract installed?"
            ) from e

    def _merge_subtitle(
        self, mkv_path: Path, srt_path: Path, output_path: Path
    ) -> None:
        logger.info("Merging translated subtitles into MKV...")
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
                f"Failed to merge subtitle into {mkv_path.name}. Is mkvmerge installed?"
            ) from e
