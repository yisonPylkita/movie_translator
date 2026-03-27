import unittest.mock

import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from ..logging import logger


def _load_simple_lama(device: torch.device) -> SimpleLama:
    """Load SimpleLama, ensuring the JIT model uses map_location=cpu.

    The big-lama.pt checkpoint was saved on CUDA, so torch.jit.load without
    map_location raises NotImplementedError on CPU-only machines. We patch
    torch.jit.load to always pass map_location=cpu before constructing the
    SimpleLama instance, then move the model to the requested device.
    """
    original_jit_load = torch.jit.load

    def _patched_jit_load(f, map_location=None, **kwargs):
        return original_jit_load(f, map_location=torch.device('cpu'), **kwargs)

    with unittest.mock.patch('torch.jit.load', _patched_jit_load):
        instance = SimpleLama(device=torch.device('cpu'))

    # Move to the requested device now that the model is safely loaded
    instance.model.to(device)
    instance.device = device
    return instance


class Inpainter:
    """Wraps LaMa model for single-image inpainting."""

    def __init__(self, device: str = 'cpu'):
        torch_device = torch.device(device)
        try:
            self._model = _load_simple_lama(torch_device)
        except Exception:
            logger.warning(f'LaMa failed to load on {device}, falling back to CPU')
            self._model = _load_simple_lama(torch.device('cpu'))

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint masked regions of the image.

        Args:
            image: RGB input image.
            mask: Grayscale mask. White (255) = regions to inpaint.

        Returns:
            Inpainted RGB image.
        """
        return self._model(image, mask.convert('L'))
