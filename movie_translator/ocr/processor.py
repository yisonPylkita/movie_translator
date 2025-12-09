import shutil
import subprocess
from pathlib import Path

from ..logging import logger


class SubtitleOCR:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr = None
        self.initialized = False

    def check_availability(self) -> bool:
        try:
            import importlib.util

            return (
                importlib.util.find_spec('cv2') is not None
                and importlib.util.find_spec('paddleocr') is not None
            )
        except ImportError:
            return False

    def initialize(self) -> bool:
        if self.initialized:
            return True

        if not self.check_availability():
            return False

        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=self.use_gpu,
                show_log=False,
            )
            self.initialized = True
            logger.info('   - OCR initialized successfully')
            return True
        except Exception as e:
            logger.warning(f'Failed to initialize OCR: {e}')
            return False

    def extract_text_from_image(self, image_path: Path) -> str:
        if not self.initialize():
            return ''

        try:
            assert self.ocr is not None
            result = self.ocr.ocr(str(image_path), cls=False)
            if result and result[0]:
                return ' '.join([line[1][0] for line in result[0] if line[1][0].strip()])
        except Exception as e:
            logger.warning(f'OCR failed for {image_path.name}: {e}')
        return ''

    def cleanup(self):
        self.ocr = None
        self.initialized = False

    def process_image_based_subtitles(
        self,
        mkv_path: Path,
        track_id: int,
        output_dir: Path,
    ) -> Path | None:
        if not self.check_availability():
            logger.warning('OCR not available - skipping image-based subtitles')
            return None

        logger.info('🤖 Processing image-based subtitles with OCR...')
        logger.warning('   - Full OCR implementation requires PGS extraction tools')
        logger.warning('   - Using placeholder implementation')

        output_srt = output_dir / f'{mkv_path.stem}_ocr_extracted.srt'

        try:
            placeholder_content = """1
00:00:01,000 --> 00:00:03,000
[Image-based subtitle - OCR processing needed]

2
00:00:05,000 --> 00:00:07,000
[This is a placeholder - implement PGS extraction first]
"""
            output_srt.write_text(placeholder_content)
            logger.info(f'   - Created placeholder SRT: {output_srt.name}')
            return output_srt
        except Exception as e:
            logger.error(f'Failed to create placeholder SRT: {e}')
            return None
        finally:
            self.cleanup()

    def extract_pgs_to_images(
        self,
        mkv_path: Path,
        track_id: int,
        output_dir: Path,
    ) -> list[Path]:
        logger.info('🖼️  Extracting PGS subtitles to images...')

        pgs_dir = output_dir / 'pgs_temp'
        pgs_dir.mkdir(exist_ok=True)

        pgs_file = pgs_dir / f'{mkv_path.stem}_track{track_id}.sup'
        cmd = ['mkvextract', 'tracks', str(mkv_path), f'{track_id}:{pgs_file}']

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f'   - PGS track extracted: {pgs_file.name}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to extract PGS track {track_id}: {e}')
            return []

        logger.warning('   - PGS to image conversion requires BDSup2Sub (not implemented)')
        logger.warning('   - Skipping PGS track processing')

        shutil.rmtree(pgs_dir)
        return []
