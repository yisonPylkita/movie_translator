import sys
from pathlib import Path

from ..logging import logger


def is_available() -> bool:
    if sys.platform != 'darwin':
        return False
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401

        return True
    except ImportError:
        return False


def recognize_text(image_path: Path, language: str = 'en') -> str:
    if not image_path.exists():
        return ''

    try:
        import Quartz
        import Vision

        # Load image via CoreGraphics
        url = Quartz.CFURLCreateWithFileSystemPath(
            None, str(image_path), Quartz.kCFURLPOSIXPathStyle, False
        )
        image_source = Quartz.CGImageSourceCreateWithURL(url, None)
        if image_source is None:
            logger.debug(f'Could not create image source for {image_path}')
            return ''

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            logger.debug(f'Could not load image from {image_path}')
            return ''

        # Create and configure text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)

        # Execute request
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        success = handler.performRequests_error_([request], None)
        if not success[0]:
            logger.debug(f'Vision request failed for {image_path}')
            return ''

        # Extract recognized text
        results = request.results()
        if not results:
            return ''

        lines = []
        for observation in results:
            candidate = observation.topCandidates_(1)
            if candidate:
                lines.append(candidate[0].string())

        return '\n'.join(lines)

    except Exception as e:
        logger.debug(f'OCR error for {image_path}: {e}')
        return ''
