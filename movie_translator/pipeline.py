import os
import shutil
from pathlib import Path

from .fonts import check_embedded_fonts_support_polish
from .identifier import identify_media
from .inpainting import remove_burned_in_subtitles
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available, probe_for_burned_in_subtitles
from .subtitle_fetch import SubtitleFetcher
from .subtitle_fetch.providers.animesub import AnimeSubProvider
from .subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .types import OCRResult
from .video import VideoOperations


class TranslationPipeline:
    def __init__(
        self,
        device: str = 'mps',
        batch_size: int = 16,
        model: str = 'allegro',
        enable_fetch: bool = True,
        tracker=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.enable_fetch = enable_fetch
        self.tracker = tracker
        self._extractor = None
        self._video_ops = None
        self._ocr_results: list[OCRResult] | None = None

    def _stage(self, name: str, info: str = ''):
        if self.tracker:
            self.tracker.set_stage(name, info)

    def _try_fetch_subtitles(self, video_path: Path, output_dir: Path) -> dict[str, Path]:
        if not self.enable_fetch:
            return {}

        try:
            identity = identify_media(video_path)
        except Exception as e:
            logger.warning(f'Media identification failed: {e}')
            return {}

        providers: list = [AnimeSubProvider()]

        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))

        fetcher = SubtitleFetcher(providers)

        try:
            return fetcher.fetch_subtitles(identity, ['eng', 'pol'], output_dir)
        except Exception as e:
            logger.warning(f'Subtitle fetch failed: {e}')
            return {}

    def process_video_file(self, video_path: Path, temp_dir: Path, dry_run: bool = False) -> bool:
        self._ocr_results = None

        try:
            # Step 1: Identify media
            self._stage('identify')
            logger.info(f'Identifying: {video_path.name}')

            # Step 2: Fetch subtitles
            self._stage('fetch')
            fetched = self._try_fetch_subtitles(video_path, temp_dir)
            fetched_eng = fetched.get('eng')
            fetched_pol = fetched.get('pol')

            if fetched_pol:
                self._stage('fetch', 'Polish from AnimeSub')
                logger.info('Fetched Polish subtitles')
            elif fetched_eng:
                self._stage('fetch', 'English only')
                logger.info('Fetched English subtitles')
            else:
                self._stage('fetch', 'none found')

            # Step 3: Extract embedded subtitles (if needed)
            self._stage('extract')
            if fetched_eng:
                extracted_ass = fetched_eng
            else:
                extracted_ass = self._extract_subtitles(video_path, temp_dir)
                if not extracted_ass:
                    return False

            # Step 4: Translate (if needed)
            if fetched_pol:
                logger.info('Using fetched Polish — skipping translation')
                polish_ass = fetched_pol
                dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
                if not dialogue_lines:
                    logger.error('No dialogue lines found')
                    return False
                translated_dialogue = None
                self._stage('translate', 'skipped')
            else:
                self._stage('translate')
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
                polish_ass = None

            # Step 5: Create subtitle files
            self._stage('create')
            fonts_support_polish = check_embedded_fonts_support_polish(video_path, extracted_ass)

            clean_english_ass = temp_dir / f'{video_path.stem}_english_clean.ass'
            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            if polish_ass is None:
                polish_ass = temp_dir / f'{video_path.stem}_polish.ass'
                replace_chars = not fonts_support_polish
                SubtitleProcessor.create_polish_subtitles(
                    extracted_ass, translated_dialogue, polish_ass, replace_chars
                )

            # Step 5.5: Inpaint burned-in subtitles if detected
            source_video = video_path
            if self._ocr_results:
                logger.info('Removing burned-in subtitles...')
                inpainted_video = temp_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
                remove_burned_in_subtitles(
                    video_path, inpainted_video, self._ocr_results, self.device
                )
                source_video = inpainted_video

            # Step 6: Mux final video
            self._stage('mux')
            temp_video = temp_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(source_video, clean_english_ass, polish_ass, temp_video)
            video_ops.verify_result(temp_video)

            if not dry_run:
                self._replace_original(video_path, temp_video)

            return True

        except Exception as e:
            logger.error(f'Failed: {video_path.name} - {e}')
            return False

    def _get_extractor(self) -> SubtitleExtractor:
        if self._extractor is None:
            self._extractor = SubtitleExtractor()
        return self._extractor

    def _get_video_ops(self) -> VideoOperations:
        if self._video_ops is None:
            self._video_ops = VideoOperations()
        return self._video_ops

    def _extract_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        extractor = self._get_extractor()

        track_info = extractor.get_track_info(video_path)
        if not track_info:
            logger.error('Could not read track information')
            return None

        eng_track = extractor.find_english_track(track_info)
        if not eng_track:
            if not is_vision_ocr_available():
                logger.error('No English subtitle track found (OCR not available)')
                return None
            if not probe_for_burned_in_subtitles(video_path):
                logger.info('No burned-in subtitles detected — skipping OCR')
                return None
            return self._extract_burned_in_subtitles(video_path, output_dir)

        track_id = eng_track['id']
        codec = eng_track.get('codec_name', '?')
        logger.info(f'English track: {codec} (ID {track_id})')

        subtitle_ext = extractor.get_subtitle_extension(eng_track)
        extracted_sub = output_dir / f'{video_path.stem}_extracted{subtitle_ext}'
        subtitle_index = eng_track.get('subtitle_index', 0)
        extractor.extract_subtitle(video_path, track_id, extracted_sub, subtitle_index)

        return extracted_sub

    def _extract_burned_in_subtitles(self, video_path: Path, output_dir: Path) -> Path | None:
        logger.info('Attempting burned-in subtitle OCR...')
        result = extract_burned_in_subtitles(video_path, output_dir)
        if result is None:
            return None
        self._ocr_results = result.ocr_results
        return result.srt_path

    def _replace_original(self, video_path: Path, temp_video: Path) -> None:
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
