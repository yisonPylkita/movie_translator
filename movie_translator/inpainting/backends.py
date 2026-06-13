"""Inpainting backend implementations.

Multiple backends for filling video regions after subtitle removal.
All frame-level backends share the same interface: inpaint(image, mask) -> image.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class InpaintBackend(Protocol):
    """Protocol for frame-level inpainting backends."""

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint masked regions of the image.

        Args:
            image: RGB input image.
            mask: Grayscale mask. White (255) = regions to inpaint.

        Returns:
            Inpainted RGB image.
        """
        ...


class OpenCVTeleaBackend:
    """OpenCV Telea algorithm. Fast classical inpainting."""

    def __init__(self, radius: int = 5):
        self._radius = radius

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import cv2

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask_arr = np.array(mask.convert('L'))
        result = cv2.inpaint(img_bgr, mask_arr, self._radius, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


class OpenCVNSBackend:
    """OpenCV Navier-Stokes algorithm. Fast classical inpainting."""

    def __init__(self, radius: int = 5):
        self._radius = radius

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import cv2

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask_arr = np.array(mask.convert('L'))
        result = cv2.inpaint(img_bgr, mask_arr, self._radius, cv2.INPAINT_NS)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


BACKENDS = {
    'opencv-telea': OpenCVTeleaBackend,
    'opencv-ns': OpenCVNSBackend,
}


def create_backend(name: str, device: str = 'cpu') -> InpaintBackend:
    """Create an inpainting backend by name.

    Args:
        name: One of 'opencv-telea', 'opencv-ns'.
        device: Device for ML backends (unused for OpenCV backends).
    """
    if name not in BACKENDS:
        raise ValueError(f'Unknown backend: {name!r}. Choose from: {", ".join(BACKENDS)}')
    return BACKENDS[name]()
