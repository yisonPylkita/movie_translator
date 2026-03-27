import numpy as np
import pytest
from PIL import Image

from movie_translator.inpainting.inpainter import Inpainter


@pytest.mark.slow
class TestInpainter:
    def test_inpaints_masked_region(self):
        # Create a red image with a white rectangle (simulating subtitle text)
        image = Image.new('RGB', (256, 256), (180, 60, 60))
        pixels = image.load()
        for x in range(80, 176):
            for y in range(200, 240):
                pixels[x, y] = (255, 255, 255)

        # Mask covers the white rectangle
        mask = Image.new('L', (256, 256), 0)
        mask_pixels = mask.load()
        for x in range(70, 186):
            for y in range(190, 250):
                mask_pixels[x, y] = 255

        inpainter = Inpainter(device='cpu')
        result = inpainter.inpaint(image, mask)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)

        # The masked region should no longer be pure white
        result_array = np.array(result)
        masked_region = result_array[200:240, 80:176]
        mean_brightness = masked_region.mean()
        # Inpainted region should blend with red background, not stay white
        assert mean_brightness < 220

    def test_unmasked_image_unchanged(self):
        image = Image.new('RGB', (128, 128), (100, 150, 200))
        mask = Image.new('L', (128, 128), 0)  # empty mask

        inpainter = Inpainter(device='cpu')
        result = inpainter.inpaint(image, mask)

        # With empty mask, result should be very similar to input
        diff = np.abs(np.array(result).astype(float) - np.array(image).astype(float))
        assert diff.mean() < 5.0
