from PIL import Image, ImageDraw

from ..types import BoundingBox


def generate_mask(
    boxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
    dilation_px: int = 20,
) -> Image.Image:
    """Generate a binary mask from bounding boxes. White (255) = inpaint region."""
    mask = Image.new('L', (frame_width, frame_height), 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        x1 = max(0, int(box.x * frame_width) - dilation_px)
        y1 = max(0, int(box.y * frame_height) - dilation_px)
        x2 = min(frame_width, int((box.x + box.width) * frame_width) + dilation_px)
        y2 = min(frame_height, int((box.y + box.height) * frame_height) + dilation_px)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask
