import shutil
from pathlib import Path

from rich.panel import Panel

from .fonts import check_embedded_fonts_support_polish
from .logging import console, logger
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
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_ocr = enable_ocr
        self.ocr_gpu = ocr_gpu

        self.extractor = SubtitleExtractor(enable_ocr=enable_ocr)
        self.parser = SubtitleParser()
        self.writer = SubtitleWriter()
        self.validator = SubtitleValidator()
        self.video_ops = VideoOperations()
        self.ocr = SubtitleOCR(use_gpu=ocr_gpu) if enable_ocr else None

    def process_video_file(self, video_path: Path, temp_dir: Path) -> bool:
        console.print(
            Panel(f'[bold blue]Processing: {video_path.name}[/bold blue]', border_style='blue')
        )

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
            if not clean_english_ass or not polish_ass:
                return False

            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            if not self._create_and_verify_video(
                video_path, clean_english_ass, polish_ass, temp_video
            ):
                return False

            if not self._replace_original(video_path, temp_video):
                return False

            logger.info(
                f'🎉 SUCCESS! Original video replaced with translated version: {video_path.name}'
            )
            return True

        except Exception as e:
            logger.error(f'Pipeline failed: {e}')
            return False

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('📖 Step 1: Extracting English subtitles...')

        track_info = self.extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = self.extractor.find_english_track(track_info)
        if not eng_track:
            logger.warning('No suitable English subtitle track found')
            return None

        track_id = eng_track['id']
        track_name = eng_track.get('properties', {}).get('track_name', 'Unknown')
        logger.info(f"Found English track: ID {track_id}, Name: '{track_name}'")

        if eng_track.get('requires_ocr', False):
            return self._process_ocr_subtitles(video_path, track_id, output_dir)

        subtitle_ext = self.extractor.get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'

        subtitle_index = eng_track.get('subtitle_index', 0)

        if not self.extractor.extract_subtitle(video_path, track_id, extracted_ass, subtitle_index):
            return None

        return extracted_ass

    def _process_ocr_subtitles(
        self, video_path: Path, track_id: int, output_dir: Path
    ) -> Path | None:
        logger.info('🤖 Processing image-based subtitles with OCR...')

        if not self.ocr:
            self.ocr = SubtitleOCR(use_gpu=self.ocr_gpu)

        extracted_srt = self.ocr.process_image_based_subtitles(video_path, track_id, output_dir)
        if not extracted_srt:
            logger.error('OCR processing failed')
            return None

        return extracted_srt

    def _parse_dialogue(self, subtitle_file: Path) -> list[DialogueLine] | None:
        logger.info('🔍 Step 2: Extracting dialogue lines...')

        dialogue_lines = self.parser.extract_dialogue_lines(subtitle_file)
        if not dialogue_lines:
            logger.error('No dialogue lines found')
            return None

        return dialogue_lines

    def _translate(self, dialogue_lines: list[DialogueLine]) -> list[DialogueLine] | None:
        logger.info('🤖 Step 3: AI translating to Polish...')

        try:
            translated = translate_dialogue_lines(
                dialogue_lines, self.device, self.batch_size, self.model
            )
            if not translated:
                logger.error('AI translation failed')
                return None
            logger.info('✅ AI translation complete!')
            return translated
        except Exception as e:
            logger.error(f'AI translation failed: {e}')
            return None

    def _create_subtitle_files(
        self,
        video_path: Path,
        output_dir: Path,
        extracted_ass: Path,
        dialogue_lines: list[DialogueLine],
        translated_dialogue: list[DialogueLine],
        fonts_support_polish: bool,
    ) -> tuple[Path | None, Path | None]:
        logger.info('🔨 Step 4: Creating clean subtitle files...')

        clean_english_ass = output_dir / f'{video_path.stem}_english_clean.ass'
        polish_ass = output_dir / f'{video_path.stem}_polish.ass'

        self.writer.create_english_ass(extracted_ass, dialogue_lines, clean_english_ass)

        if not self.validator.validate_cleaned_subtitles(extracted_ass, clean_english_ass):
            logger.error('❌ Validation failed! Cleaned subtitles have timestamp mismatches.')
            return None, None

        replace_chars = not fonts_support_polish
        self.writer.create_polish_ass(extracted_ass, translated_dialogue, polish_ass, replace_chars)

        return clean_english_ass, polish_ass

    def _create_and_verify_video(
        self,
        original_video: Path,
        english_ass: Path,
        polish_ass: Path,
        temp_video: Path,
    ) -> bool:
        logger.info('🎬 Step 5: Creating clean video...')

        if not self.video_ops.create_clean_video(
            original_video, english_ass, polish_ass, temp_video
        ):
            return False

        if not self.video_ops.verify_result(temp_video):
            return False

        return True

    def _replace_original(self, video_path: Path, temp_video: Path) -> bool:
        logger.info('🔄 Step 6: Replacing original video...')

        backup_path = video_path.with_suffix(video_path.suffix + '.backup')

        try:
            shutil.copy2(video_path, backup_path)
            logger.info(f'   - Created backup: {backup_path.name}')

            shutil.move(str(temp_video), str(video_path))
            logger.info(f'   - Replaced original: {video_path.name}')

            if not self.video_ops.verify_result(video_path):
                logger.error('   - Verification failed, restoring backup')
                shutil.move(str(backup_path), str(video_path))
                return False

            backup_path.unlink()
            logger.info('   - Verified and cleaned up backup')
            return True

        except Exception as e:
            logger.error(f'   - Failed to replace original: {e}')
            if backup_path.exists() and not video_path.exists():
                shutil.move(str(backup_path), str(video_path))
                logger.info('   - Restored backup after failure')
            return False
