import sys
from pathlib import Path

from ..logging import logger
from ..types import BoundingBox


def is_available() -> bool:
    if sys.platform != 'darwin':
        return False
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401

        return True
    except ImportError:
        return False


def recognize_text_with_boxes(
    image_path: Path, language: str = 'en'
) -> list[tuple[str, BoundingBox]]:
    """Recognize text in image, returning text and bounding boxes.

    Bounding boxes are normalized (0-1) with top-left origin.
    """
    if not image_path.exists():
        return []

    try:
        import Quartz
        import Vision

        url = Quartz.CFURLCreateWithFileSystemPath(
            None, str(image_path), Quartz.kCFURLPOSIXPathStyle, False
        )
        image_source = Quartz.CGImageSourceCreateWithURL(url, None)
        if image_source is None:
            logger.debug(f'Could not create image source for {image_path}')
            return []

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            logger.debug(f'Could not load image from {image_path}')
            return []

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        success = handler.performRequests_error_([request], None)
        if not success[0]:
            logger.debug(f'Vision request failed for {image_path}')
            return []

        observations = request.results()
        if not observations:
            return []

        results: list[tuple[str, BoundingBox]] = []
        for observation in observations:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            text = candidates[0].string()
            bbox = observation.boundingBox()
            # Vision uses bottom-left origin — convert to top-left
            box = BoundingBox(
                x=bbox.origin.x,
                y=1.0 - bbox.origin.y - bbox.size.height,
                width=bbox.size.width,
                height=bbox.size.height,
            )
            results.append((text, box))

        return results

    except Exception as e:
        logger.debug(f'OCR error for {image_path}: {e}')
        return []


def recognize_text(image_path: Path, language: str = 'en') -> str:
    """Recognize text in image, returning concatenated text string."""
    results = recognize_text_with_boxes(image_path, language)
    return '\n'.join(text for text, _ in results)
