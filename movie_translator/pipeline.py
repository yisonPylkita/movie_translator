"""Video processing pipeline."""

import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .ocr import SubtitleOCR
from .subtitles import SubtitleExtractor, SubtitleParser, SubtitleValidator, SubtitleWriter
from .translation import translate_dialogue_lines
from .utils import log_error, log_info, log_success, log_warning
from .video import VideoOperations

console = Console()


class TranslationPipeline:
    """Orchestrates the complete video translation workflow."""

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

        # Initialize components
        self.extractor = SubtitleExtractor(enable_ocr=enable_ocr)
        self.parser = SubtitleParser()
        self.writer = SubtitleWriter()
        self.validator = SubtitleValidator()
        self.video_ops = VideoOperations()
        self.ocr = SubtitleOCR(use_gpu=ocr_gpu) if enable_ocr else None

    def process_video_file(self, video_path: Path, output_dir: Path) -> bool:
        """Process a single video file and replace it with clean version."""
        console.print(
            Panel(f'[bold blue]Processing: {video_path.name}[/bold blue]', border_style='blue')
        )

        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Extract subtitles
            extracted_ass = self._extract_subtitles(video_path, output_dir)
            if not extracted_ass:
                return False

            # Step 2: Parse dialogue
            dialogue_lines = self._parse_dialogue(extracted_ass)
            if not dialogue_lines:
                return False

            # Step 3: Translate
            translated_dialogue = self._translate(dialogue_lines)
            if not translated_dialogue:
                return False

            # Step 4: Create subtitle files
            clean_english_ass, polish_ass = self._create_subtitle_files(
                video_path, output_dir, extracted_ass, dialogue_lines, translated_dialogue
            )
            if not clean_english_ass or not polish_ass:
                return False

            # Step 5: Create and verify video
            # Preserve original extension for output
            temp_video = temp_dir / f'{video_path.stem}_temp_clean{video_path.suffix}'
            if not self._create_and_verify_video(
                video_path, clean_english_ass, polish_ass, temp_video
            ):
                return False

            # Step 6: Replace original
            if not self._replace_original(video_path, temp_video):
                return False

            # Cleanup
            shutil.rmtree(temp_dir)
            log_info('🧹 Cleaned up temporary files')

            log_success(
                f'🎉 SUCCESS! Original video replaced with clean version: {video_path.name}'
            )
            return True

        except Exception as e:
            log_error(f'Pipeline failed: {e}')
            return False

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        """Step 1: Extract English subtitles."""
        log_info('📖 Step 1: Extracting English subtitles...')

        track_info = self.extractor.get_track_info(video_path)
        if not track_info:
            log_error('Could not read track information')
            return None

        eng_track = self.extractor.find_english_track(track_info)
        if not eng_track:
            log_warning('No suitable English subtitle track found')
            return None

        track_id = eng_track['id']
        track_name = eng_track.get('properties', {}).get('track_name', 'Unknown')
        log_info(f"Found English track: ID {track_id}, Name: '{track_name}'")

        # Handle OCR if needed
        if eng_track.get('requires_ocr', False):
            return self._process_ocr_subtitles(video_path, track_id, output_dir)

        # Normal extraction
        subtitle_ext = self.extractor.get_subtitle_extension(eng_track)
        extracted_ass = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'

        # Get subtitle stream index for ffmpeg
        subtitle_index = eng_track.get('subtitle_index', 0)

        if not self.extractor.extract_subtitle(video_path, track_id, extracted_ass, subtitle_index):
            return None

        return extracted_ass

    def _process_ocr_subtitles(
        self, video_path: Path, track_id: int, output_dir: Path
    ) -> Path | None:
        """Process image-based subtitles with OCR."""
        log_info('🤖 Processing image-based subtitles with OCR...')

        if not self.ocr:
            self.ocr = SubtitleOCR(use_gpu=self.ocr_gpu)

        extracted_srt = self.ocr.process_image_based_subtitles(video_path, track_id, output_dir)
        if not extracted_srt:
            log_error('OCR processing failed')
            return None

        return extracted_srt

    def _parse_dialogue(self, subtitle_file: Path) -> list[tuple[int, int, str]] | None:
        """Step 2: Extract dialogue lines."""
        log_info('🔍 Step 2: Extracting dialogue lines...')

        dialogue_lines = self.parser.extract_dialogue_lines(subtitle_file)
        if not dialogue_lines:
            log_error('No dialogue lines found')
            return None

        return dialogue_lines

    def _translate(
        self, dialogue_lines: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]] | None:
        """Step 3: AI translate to Polish."""
        log_info('🤖 Step 3: AI translating to Polish...')

        try:
            translated = translate_dialogue_lines(
                dialogue_lines, self.device, self.batch_size, self.model
            )
            if not translated:
                log_error('AI translation failed')
                return None
            log_success('   ✅ AI translation complete!')
            return translated
        except Exception as e:
            log_error(f'AI translation failed: {e}')
            return None

    def _create_subtitle_files(
        self,
        video_path: Path,
        output_dir: Path,
        extracted_ass: Path,
        dialogue_lines: list[tuple[int, int, str]],
        translated_dialogue: list[tuple[int, int, str]],
    ) -> tuple[Path | None, Path | None]:
        """Step 4: Create clean subtitle files."""
        log_info('🔨 Step 4: Creating clean subtitle files...')

        clean_english_ass = output_dir / f'{video_path.stem}_english_clean.ass'
        polish_ass = output_dir / f'{video_path.stem}_polish.ass'

        self.writer.create_english_ass(extracted_ass, dialogue_lines, clean_english_ass)

        # Validate
        extracted_ass_path = output_dir / f'{video_path.stem}_extracted.ass'
        if not self.validator.validate_cleaned_subtitles(extracted_ass_path, clean_english_ass):
            log_error('❌ Validation failed! Cleaned subtitles have timestamp mismatches.')
            return None, None

        self.writer.create_polish_ass(extracted_ass, translated_dialogue, polish_ass)

        return clean_english_ass, polish_ass

    def _create_and_verify_video(
        self,
        original_video: Path,
        english_ass: Path,
        polish_ass: Path,
        temp_video: Path,
    ) -> bool:
        """Step 5: Create clean video and verify."""
        log_info('🎬 Step 5: Creating clean video...')

        if not self.video_ops.create_clean_video(
            original_video, english_ass, polish_ass, temp_video
        ):
            return False

        if not self.video_ops.verify_result(temp_video):
            return False

        return True

    def _replace_original(self, video_path: Path, temp_video: Path) -> bool:
        """Step 6: Replace original file with backup."""
        log_info('🔄 Step 6: Replacing original video...')

        backup_path = video_path.with_suffix(video_path.suffix + '.backup')

        try:
            # Create backup
            shutil.copy2(video_path, backup_path)
            log_info(f'   - Created backup: {backup_path.name}')

            # Replace original
            shutil.move(str(temp_video), str(video_path))
            log_success(f'   - Replaced original: {video_path.name}')

            # Verify replacement
            if not self.video_ops.verify_result(video_path):
                log_error('   - Verification failed after replacement, restoring backup')
                shutil.move(str(backup_path), str(video_path))
                return False

            # Remove backup
            backup_path.unlink()
            log_info('   - Removed backup (verification successful)')
            return True

        except Exception as e:
            log_error(f'   - Failed to replace original: {e}')
            return False
