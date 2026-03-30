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


def _load_simple_lama(device):
    """Load SimpleLama, ensuring the JIT model uses map_location=cpu.

    The big-lama.pt checkpoint was saved on CUDA, so torch.jit.load without
    map_location raises NotImplementedError on CPU-only machines. We temporarily
    patch torch.jit.load to always pass map_location=cpu, then restore the original.
    """
    import torch
    from simple_lama_inpainting import SimpleLama

    original_jit_load = torch.jit.load

    def _patched_jit_load(f, map_location=None, **kwargs):
        return original_jit_load(f, map_location=torch.device('cpu'), **kwargs)

    torch.jit.load = _patched_jit_load  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]
    try:
        instance = SimpleLama(device=torch.device('cpu'))
    finally:
        torch.jit.load = original_jit_load

    instance.model.to(device)
    instance.device = device
    return instance


class LamaBackend:
    """LaMa neural network inpainting. Highest quality, slowest."""

    def __init__(self, device: str = 'cpu'):
        import torch

        from ..logging import logger

        torch_device = torch.device(device)
        try:
            self._model = _load_simple_lama(torch_device)
        except Exception:
            logger.warning(f'LaMa failed to load on {device}, falling back to CPU')
            self._model = _load_simple_lama(torch.device('cpu'))

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        return self._model(image, mask.convert('L'))


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
    'lama': LamaBackend,
    'opencv-telea': OpenCVTeleaBackend,
    'opencv-ns': OpenCVNSBackend,
}


def create_backend(name: str, device: str = 'cpu') -> InpaintBackend:
    """Create an inpainting backend by name.

    Args:
        name: One of 'lama', 'opencv-telea', 'opencv-ns'.
        device: Device for ML backends ('cpu', 'mps', 'cuda').
    """
    if name not in BACKENDS:
        raise ValueError(f'Unknown backend: {name!r}. Choose from: {", ".join(BACKENDS)}')
    cls = BACKENDS[name]
    if name == 'lama':
        return cls(device=device)  # type: ignore[unknown-argument]  # ty:ignore[unknown-argument]
    return cls()
