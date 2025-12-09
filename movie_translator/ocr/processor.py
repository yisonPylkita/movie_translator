import importlib.util
from pathlib import Path

from ..logging import logger


class SubtitleOCR:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr = None
        self.initialized = False

    def check_availability(self) -> bool:
        return (
            importlib.util.find_spec('cv2') is not None
            and importlib.util.find_spec('paddleocr') is not None
        )

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

    def cleanup(self):
        self.ocr = None
        self.initialized = False

    def process_image_based_subtitles(
        self,
        video_path: Path,
        track_id: int,
        output_dir: Path,
    ) -> Path | None:
        logger.error('❌ Image-based subtitle OCR is not yet implemented')
        logger.info('   - This feature requires PGS extraction tools (BDSup2Sub)')
        logger.info('   - Consider using text-based subtitles (ASS/SRT) instead')
        return None
