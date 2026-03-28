import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .fonts import (
    check_embedded_fonts_support_polish,
    find_system_font_for_polish,
    get_ass_font_names,
)
from .identifier import identify_media
from .inpainting import remove_burned_in_subtitles
from .logging import logger
from .ocr import extract_burned_in_subtitles, is_vision_ocr_available, probe_for_burned_in_subtitles
from .subtitle_fetch import SubtitleFetcher, SubtitleValidator
from .subtitle_fetch.providers.animesub import AnimeSubProvider
from .subtitle_fetch.providers.napiprojekt import NapiProjektProvider
from .subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from .subtitle_fetch.providers.podnapisi import PodnapisiProvider
from .subtitle_fetch.types import SubtitleMatch
from .subtitles import SubtitleExtractor, SubtitleProcessor
from .translation import translate_dialogue_lines
from .types import OCRResult, SubtitleFile
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

    def _build_fetcher(self, video_path: Path | None = None) -> SubtitleFetcher | None:
        """Create a SubtitleFetcher with all configured providers."""
        if not self.enable_fetch:
            return None
        providers: list = [AnimeSubProvider()]

        # Podnapisi (always available, no API key needed)
        providers.append(PodnapisiProvider())

        # NapiProjekt (needs video path for hash computation)
        if video_path is not None:
            napi = NapiProjektProvider()
            napi.set_video_path(video_path)
            providers.append(napi)

        # OpenSubtitles (needs API key)
        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))

        return SubtitleFetcher(providers)

    def _extract_reference(self, video_path: Path, work_dir: Path) -> Path | None:
        """Extract a reference subtitle for validation.

        Tries embedded English track first, falls back to OCR.
        Returns the path to the reference subtitle file, or None.
        """
        extractor = self._get_extractor()
        ref_dir = work_dir / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)

        # Try embedded English track first
        track_info = extractor.get_track_info(video_path)
        if track_info:
            eng_track = extractor.find_english_track(track_info)
            if eng_track:
                subtitle_ext = extractor.get_subtitle_extension(eng_track)
                ref_path = ref_dir / f'{video_path.stem}_reference{subtitle_ext}'
                try:
                    subtitle_index = eng_track.get('subtitle_index', 0)
                    extractor.extract_subtitle(
                        video_path, eng_track['id'], ref_path, subtitle_index
                    )
                    logger.info(f'Extracted embedded reference: {ref_path.name}')
                    return ref_path
                except Exception as e:
                    logger.warning(f'Failed to extract embedded reference: {e}')

        # Fall back to OCR
        if is_vision_ocr_available() and probe_for_burned_in_subtitles(video_path):
            ocr_result = self._extract_burned_in_subtitles(video_path, ref_dir)
            if ocr_result:
                logger.info(f'Extracted OCR reference: {ocr_result.name}')
                return ocr_result

        return None

    def _search_and_validate(
        self, video_path: Path, work_dir: Path, reference_path: Path | None
    ) -> dict[str, Path]:
        """Search all providers, download all candidates, validate, select best per language.

        Returns {language_code: subtitle_file_path} for the best subtitle per language.
        """
        fetcher = self._build_fetcher(video_path=video_path)
        if fetcher is None:
            return {}

        # Identify media
        try:
            identity = identify_media(video_path)
        except Exception as e:
            logger.warning(f'Media identification failed: {e}')
            return {}

        # Search all providers
        try:
            all_matches = fetcher.search_all(identity, ['eng', 'pol'])
        except Exception as e:
            logger.warning(f'Subtitle search failed: {e}')
            return {}

        if not all_matches:
            logger.info('No subtitles found from any provider')
            return {}

        logger.info(f'Found {len(all_matches)} subtitle candidate(s)')

        # Download all candidates
        candidates_dir = work_dir / 'candidates'
        candidates_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[tuple[SubtitleMatch, Path]] = []

        def _download(i_match: tuple[int, SubtitleMatch]) -> tuple[SubtitleMatch, Path]:
            i, match = i_match
            filename = f'{match.source}_{match.language}_{i}.{match.format}'
            output_path = candidates_dir / filename
            fetcher.download_candidate(match, output_path)
            return match, output_path

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_download, (i, m)): m for i, m in enumerate(all_matches)}
            for future in as_completed(futures):
                try:
                    downloaded.append(future.result())
                except Exception as e:
                    match = futures[future]
                    logger.warning(f'Failed to download candidate {match.subtitle_id}: {e}')

        if not downloaded:
            logger.warning('All candidate downloads failed')
            return {}

        logger.info(f'Downloaded {len(downloaded)} candidate(s)')

        # Validate against reference if available
        if reference_path is not None:
            try:
                validator = SubtitleValidator(reference_path)
                validated = validator.validate_candidates(downloaded, min_threshold=0.3)
            except Exception as e:
                logger.warning(f'Validation failed, falling back to provider scoring: {e}')
                validated = None
        else:
            logger.warning('No reference subtitle available — using provider scoring only')
            validated = None

        # Select best per language — returns (path, source_name) tuples
        result: dict[str, tuple[Path, str]] = {}

        if validated is not None:
            if validated:
                logger.info(f'{len(validated)} candidate(s) passed validation')
                for match, path, score in validated:
                    if match.language not in result:
                        result[match.language] = (path, match.source)
                        logger.info(
                            f'Best {match.language}: {match.release_name} '
                            f'(validation score: {score:.3f}, source: {match.source})'
                        )
            else:
                logger.warning('No candidates passed validation threshold')
        else:
            # Fall back to provider scoring (pick best per language from downloaded)
            for match, path in downloaded:
                if match.language not in result:
                    result[match.language] = (path, match.source)
                    logger.info(
                        f'Best {match.language} (unvalidated): {match.release_name} '
                        f'(provider score: {match.score:.2f}, source: {match.source})'
                    )

        return result

    def process_video_file(self, video_path: Path, work_dir: Path, dry_run: bool = False) -> bool:
        self._ocr_results = None

        try:
            # Step 1: Identify media
            self._stage('identify')
            logger.info(f'Identifying: {video_path.name}')

            # Step 2: Extract reference subtitle for validation
            self._stage('extract', 'reference')
            reference_path = self._extract_reference(video_path, work_dir)
            if reference_path:
                logger.info(f'Reference subtitle: {reference_path.name}')
            else:
                logger.info('No reference subtitle available')

            # Step 3: Search and validate fetched subtitles
            self._stage('fetch')
            fetched = self._search_and_validate(video_path, work_dir, reference_path)
            fetched_eng_result = fetched.get('eng')
            fetched_pol_result = fetched.get('pol')
            fetched_eng = fetched_eng_result[0] if fetched_eng_result else None
            fetched_pol = fetched_pol_result[0] if fetched_pol_result else None
            fetched_pol_source = fetched_pol_result[1] if fetched_pol_result else None

            if fetched_pol:
                self._stage('fetch', 'Polish validated')
                logger.info('Fetched Polish subtitles')
            elif fetched_eng:
                self._stage('fetch', 'English validated')
                logger.info('Fetched English subtitles')
            else:
                self._stage('fetch', 'none found')

            # Step 4: Determine English source
            #   fetched_eng > reference_path > _extract_subtitles fallback
            self._stage('extract')
            if fetched_eng:
                extracted_ass = fetched_eng
            elif reference_path:
                extracted_ass = reference_path
            else:
                extracted_ass = self._extract_subtitles(video_path, work_dir)
                if not extracted_ass:
                    return False

            # Step 5: Extract dialogue, then translate + check fonts in parallel
            dialogue_lines = SubtitleProcessor.extract_dialogue_lines(extracted_ass)
            if not dialogue_lines:
                logger.error('No dialogue lines found')
                return False

            self._stage('translate')
            logger.info(f'Translating {len(dialogue_lines)} lines...')

            # Font check runs concurrently with translation (I/O vs GPU)
            def _check_fonts():
                supports = check_embedded_fonts_support_polish(video_path, extracted_ass)
                if supports:
                    return supports, [], None
                is_mkv = video_path.suffix.lower() == '.mkv'
                if is_mkv:
                    names = get_ass_font_names(extracted_ass)
                    result = find_system_font_for_polish(names)
                    if result:
                        fp, fam = result
                        fallback = None if any(fam.lower() == n.lower() for n in names) else fam
                        return False, [fp], fallback
                return False, [], None

            with ThreadPoolExecutor(max_workers=2) as pool:
                font_future = pool.submit(_check_fonts)
                translate_future = pool.submit(
                    translate_dialogue_lines,
                    dialogue_lines,
                    self.device,
                    self.batch_size,
                    self.model,
                )

                fonts_support_polish, font_attachments, fallback_font_family = font_future.result()

                try:
                    translated_dialogue = translate_future.result()
                    if not translated_dialogue:
                        logger.error('Translation failed')
                        return False
                except Exception as e:
                    logger.error(f'Translation failed: {e}')
                    return False

            # Step 6: Create subtitle files
            self._stage('create')
            is_mkv = video_path.suffix.lower() == '.mkv'
            replace_chars = False

            if not fonts_support_polish:
                if is_mkv:
                    if font_attachments:
                        logger.info(
                            f'   - Will embed system font "{font_attachments[0].name}" for Polish support'
                        )
                    else:
                        logger.warning(
                            '   - No system font with Polish support found, replacing characters'
                        )
                        replace_chars = True
                else:
                    logger.info(
                        '   - MP4 container does not support font attachments, replacing characters'
                    )
                    replace_chars = True

            # Clean English dialogue only (no signs/songs/OP/ED)
            clean_english_ass = work_dir / f'{video_path.stem}_english_clean.ass'
            SubtitleProcessor.create_english_subtitles(
                extracted_ass, dialogue_lines, clean_english_ass
            )
            SubtitleProcessor.validate_cleaned_subtitles(extracted_ass, clean_english_ass)

            # AI-translated Polish subtitles
            ai_polish_ass = work_dir / f'{video_path.stem}_polish_ai.ass'
            SubtitleProcessor.create_polish_subtitles(
                extracted_ass, translated_dialogue, ai_polish_ass, replace_chars
            )
            if fallback_font_family:
                SubtitleProcessor.override_font_name(ai_polish_ass, fallback_font_family)

            # Build subtitle track list:
            #   1. Downloaded Polish (default, if available)
            #   2. AI Polish
            #   3. Clean English
            subtitle_files: list[SubtitleFile] = []
            if fetched_pol:
                pol_title = f'Polish ({fetched_pol_source})' if fetched_pol_source else 'Polish'
                subtitle_files.append(SubtitleFile(fetched_pol, 'pol', pol_title, is_default=True))
                if fallback_font_family:
                    SubtitleProcessor.override_font_name(fetched_pol, fallback_font_family)

            subtitle_files.append(
                SubtitleFile(
                    ai_polish_ass,
                    'pol',
                    'Polish (AI)',
                    is_default=not bool(fetched_pol),
                )
            )
            subtitle_files.append(
                SubtitleFile(clean_english_ass, 'eng', 'English Dialogue', is_default=False)
            )

            # Step 6.5: Inpaint burned-in subtitles if detected
            source_video = video_path
            if self._ocr_results:
                logger.info('Removing burned-in subtitles...')
                inpainted_video = work_dir / f'{video_path.stem}_inpainted{video_path.suffix}'
                remove_burned_in_subtitles(
                    video_path, inpainted_video, self._ocr_results, self.device
                )
                source_video = inpainted_video

            # Step 7: Mux final video (all existing subs stripped, only our tracks added)
            self._stage('mux')
            temp_video = work_dir / f'{video_path.stem}_temp{video_path.suffix}'
            video_ops = self._get_video_ops()
            video_ops.create_clean_video(
                source_video,
                subtitle_files,
                temp_video,
                font_attachments=font_attachments or None,
            )
            video_ops.verify_result(temp_video, expected_tracks=subtitle_files)

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
