from movie_translator.inpainting.mask_generator import generate_mask
from movie_translator.types import BoundingBox


class TestGenerateMask:
    def test_creates_mask_with_correct_dimensions(self):
        boxes = [BoundingBox(x=0.1, y=0.8, width=0.8, height=0.1)]
        mask = generate_mask(boxes, frame_width=1920, frame_height=1080)

        assert mask.size == (1920, 1080)
        assert mask.mode == 'L'

    def test_white_pixels_in_box_region(self):
        boxes = [BoundingBox(x=0.25, y=0.75, width=0.5, height=0.2)]
        mask = generate_mask(boxes, frame_width=100, frame_height=100, dilation_px=0)

        # Center of box region should be white (255)
        assert mask.getpixel((50, 85)) == 255

    def test_black_pixels_outside_box_region(self):
        boxes = [BoundingBox(x=0.25, y=0.75, width=0.5, height=0.2)]
        mask = generate_mask(boxes, frame_width=100, frame_height=100, dilation_px=0)

        # Top-left corner should be black (0)
        assert mask.getpixel((5, 5)) == 0

    def test_dilation_expands_mask(self):
        boxes = [BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1)]
        mask_no_dilation = generate_mask(boxes, 200, 200, dilation_px=0)
        mask_with_dilation = generate_mask(boxes, 200, 200, dilation_px=20)

        # Count white pixels — dilated mask should have more
        no_dil_count = sum(1 for p in mask_no_dilation.getdata() if p == 255)
        dil_count = sum(1 for p in mask_with_dilation.getdata() if p == 255)
        assert dil_count > no_dil_count

    def test_multiple_boxes(self):
        boxes = [
            BoundingBox(x=0.1, y=0.1, width=0.2, height=0.1),
            BoundingBox(x=0.6, y=0.8, width=0.3, height=0.1),
        ]
        mask = generate_mask(boxes, 100, 100, dilation_px=0)

        # Both regions should be white
        assert mask.getpixel((20, 15)) == 255
        assert mask.getpixel((75, 85)) == 255

    def test_empty_boxes_returns_black_mask(self):
        mask = generate_mask([], frame_width=100, frame_height=100)

        assert all(p == 0 for p in mask.getdata())

    def test_dilation_clamps_to_frame_bounds(self):
        # Box near edge — dilation shouldn't exceed frame bounds
        boxes = [BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)]
        mask = generate_mask(boxes, 100, 100, dilation_px=20)

        # Should not raise, and top-left should be white
        assert mask.getpixel((0, 0)) == 255
