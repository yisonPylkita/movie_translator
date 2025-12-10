import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .logging import logger
from .ocr import SubtitleOCR
from .subtitles import SubtitleExtractor, SubtitleParser, SubtitleValidator, SubtitleWriter
from .translation import translate_dialogue_lines
from .types import DialogueLine
from .video import VideoOperations


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_ocr: bool = False,
        ocr_gpu: bool = False,
        verbose: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_ocr = enable_ocr
        self.ocr_gpu = ocr_gpu
        self.verbose = verbose

        self.extractor = SubtitleExtractor(enable_ocr=enable_ocr)
        self.parser = SubtitleParser()
        self.writer = SubtitleWriter()
        self.validator = SubtitleValidator()
        self.video_ops = VideoOperations()
        self.ocr = SubtitleOCR(use_gpu=ocr_gpu) if enable_ocr else None

    def process_video_file(self, video_path: Path, temp_dir: Path, dry_run: bool = False) -> bool:
        logger.info(f'Processing: {video_path.name}')

        try:
            extracted_ass = self._extract_subtitles(video_path, temp_dir)
            if not extracted_ass:
                return False

            dialogue_lines = self._parse_dialogue(extracted_ass)
            if not dialogue_lines:
                return False

            translated_dialogue = self._translate(dialogue_lines)
            if not translated_dialogue:
                return False

            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            clean_english_ass, polish_ass = self._create_subtitle_files(
                video_path,
                temp_dir,
                extracted_ass,
                dialogue_lines,
                translated_dialogue,
                fonts_support_polish,
            )

            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            self._create_and_verify_video(video_path, clean_english_ass, polish_ass, temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            logger.info(f'Completed: {video_path.name}')
            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('Extracting subtitles...')

        track_info = self.extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = self.extractor.find_english_track(track_info)
        if not eng_track:
            logger.error('No English subtitle track found')
            return None

        track_id = eng_track['id']
        logger.info(f'Found English track: ID {track_id}')

        if eng_track.get('requires_ocr', False):
            return self._process_ocr_subtitles(video_path, track_id, output_dir)

        subtitle_ext = self.extractor.get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'

        subtitle_index = eng_track.get('subtitle_index', 0)

        self.extractor.extract_subtitle(video_path, track_id, extracted_ass, subtitle_index)

        return extracted_ass

    def _process_ocr_subtitles(
        self, video_path: Path, track_id: int, output_dir: Path
    ) -> Path | None:
        logger.info('Processing OCR subtitles...')

        if not self.ocr:
            self.ocr = SubtitleOCR(use_gpu=self.ocr_gpu)

        extracted_srt = self.ocr.process_image_based_subtitles(video_path, track_id, output_dir)
        if not extracted_srt:
            logger.error('OCR processing failed')
            return None

        return extracted_srt

    def _parse_dialogue(self, subtitle_file: Path) -> list[DialogueLine] | None:
        logger.info('Parsing dialogue...')

        dialogue_lines = self.parser.extract_dialogue_lines(subtitle_file)
        if not dialogue_lines:
            logger.error('No dialogue lines found')
            return None

        return dialogue_lines

    def _translate(self, dialogue_lines: list[DialogueLine]) -> list[DialogueLine] | None:
        logger.info(f'Translating {len(dialogue_lines)} lines...')

        try:
            translated = translate_dialogue_lines(
                dialogue_lines, self.device, self.batch_size, self.model
            )
            if not translated:
                logger.error('Translation failed')
                return None
            return translated
        except Exception as e:
            logger.error(f'Translation failed: {e}')
            return None

    def _create_subtitle_files(
        self,
        video_path: Path,
        output_dir: Path,
        extracted_ass: Path,
        dialogue_lines: list[DialogueLine],
        translated_dialogue: list[DialogueLine],
        fonts_support_polish: bool,
    ) -> tuple[Path, Path]:
        logger.info('Creating subtitle files...')

        clean_english_ass = output_dir / f'{video_path.stem}_english_clean.ass'
        polish_ass = output_dir / f'{video_path.stem}_polish.ass'

        self.writer.create_english_ass(extracted_ass, dialogue_lines, clean_english_ass)
        self.validator.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

        replace_chars = not fonts_support_polish
        self.writer.create_polish_ass(extracted_ass, translated_dialogue, polish_ass, replace_chars)

        return clean_english_ass, polish_ass

    def _create_and_verify_video(
        self,
        original_video: Path,
        english_ass: Path,
        polish_ass: Path,
        temp_video: Path,
    ) -> None:
        logger.info('Creating video...')

        self.video_ops.create_clean_video(original_video, english_ass, polish_ass, temp_video)
        self.video_ops.verify_result(temp_video)

    def _replace_original(self, video_path: Path, temp_video: Path) -> None:
        logger.info('Replacing original...')

        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)

        try:
            shutil.move(str(temp_video), str(video_path))
            self.video_ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
