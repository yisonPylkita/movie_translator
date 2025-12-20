import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .logging import logger
from .ocr import SubtitleOCR
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .video import VideoOperations


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_ocr: bool = False,
        ocr_gpu: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_ocr = enable_ocr
        self.ocr_gpu = ocr_gpu
        self._extractor = None
        self._video_ops = None
        self._ocr = None

    def process_video_file(self, video_path: Path, temp_dir: Path, dry_run: bool = False) -> bool:
        logger.info(f'Processing: {video_path.name}')

        try:
            extracted_ass = self._extract_subtitles(video_path, temp_dir)
            if not extracted_ass:
                return False

            logger.info('Parsing dialogue...')
            dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
            if not dialogue_lines:
                logger.error('No dialogue lines found')
                return False

            logger.info(f'Translating {len(dialogue_lines)} lines...')
            try:
                translated_dialogue = translate_dialogue_lines(
                    dialogue_lines, self.device, self.batch_size, self.model
                )
                if not translated_dialogue:
                    logger.error('Translation failed')
                    return False
            except Exception as e:
                logger.error(f'Translation failed: {e}')
                return False

            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            logger.info('Creating subtitle files...')
            clean_english_ass = temp_dir / f'{video_path.stem}_english_clean.ass'
            polish_ass = temp_dir / f'{video_path.stem}_polish.ass'

            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            replace_chars = not fonts_support_polish
            SubtitleProcessor.create_polish_subtitles(
                extracted_ass, translated_dialogue, polish_ass, replace_chars
            )

            logger.info('Creating video...')
            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(video_path, clean_english_ass, polish_ass, temp_video)
            video_ops.verify_result(temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            logger.info(f'Completed: {video_path.name}')
            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False

    def _get_extractor(self) -> SubtitleExtractor:
        """Lazy initialization of subtitle extractor."""
        if self._extractor is None:
            self._extractor = SubtitleExtractor(enable_ocr=self.enable_ocr)
        return self._extractor

    def _get_video_ops(self) -> VideoOperations:
        """Lazy initialization of video operations."""
        if self._video_ops is None:
            self._video_ops = VideoOperations()
        return self._video_ops

    def _get_ocr(self) -> SubtitleOCR:
        """Lazy initialization of OCR processor."""
        if self._ocr is None:
            self._ocr = SubtitleOCR(use_gpu=self.ocr_gpu)
        return self._ocr

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('Extracting subtitles...')

        extractor = self._get_extractor()

        track_info = extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = extractor.find_english_track(track_info)
        if not eng_track:
            logger.error('No English subtitle track found')
            return None

        track_id = eng_track['id']
        logger.info(f'Found English track: ID {track_id}')

        if eng_track.get('requires_ocr', False):
            return self._process_ocr_subtitles(video_path, track_id, output_dir)

        subtitle_ext = extractor.get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'
        subtitle_index = eng_track.get('subtitle_index', 0)
        extractor.extract_subtitle(video_path, track_id, extracted_ass, subtitle_index)

        return extracted_ass

    def _process_ocr_subtitles(
        self, video_path: Path, track_id: int, output_dir: Path
    ) -> Path | None:
        logger.info('Processing OCR subtitles...')

        ocr = self._get_ocr()
        extracted_srt = ocr.process_image_based_subtitles(video_path, track_id, output_dir)
        if not extracted_srt:
            logger.error('OCR processing failed')
            return None

        return extracted_srt

    def _replace_original(self, video_path: Path, temp_video: Path) -> None:
        logger.info('Replacing original...')

        backup_path = video_path.with_suffix(video_path.suffix + '.backup')
        shutil.copy2(video_path, backup_path)

        try:
            shutil.move(str(temp_video), str(video_path))
            video_ops = self._get_video_ops()
            video_ops.verify_result(video_path)
            backup_path.unlink()
        except Exception:
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
            raise
