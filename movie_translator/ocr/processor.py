import importlib.util
from pathlib import Path

from ..logging import logger


class SubtitleOCR:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def check_availability(self) -> bool:
        return (
            importlib.util.find_spec('cv2') is not None
            and importlib.util.find_spec('paddleocr') is not None
        )

    def cleanup(self):
        pass

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
