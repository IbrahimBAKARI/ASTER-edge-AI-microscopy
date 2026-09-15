"""Crop -> tensor contract. MUST stay byte-identical to the deployed
``aster_pipeline/transforms.py`` of Software_Dev_Micro_Edge.

Any divergence here silently breaks the block-1 -> block-2 interface documented in
``docs/ARCHITECTURE_CONTRACT.md`` section 2.1. The parity test in
``tests/test_preprocess_parity.py`` compares this module against the deployed one
when the deployed repository is available.
"""

from __future__ import annotations

from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


class SquarePad:
    """Square padding using the median of the four RGB corner pixels."""

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((max(0, width - 1), 0)),
            rgb.getpixel((0, max(0, height - 1))),
            rgb.getpixel((max(0, width - 1), max(0, height - 1))),
        ]
        # Deployed code: np.median(corners, axis=0).astype(np.uint8).
        # Median of four values is the mean of the two middle ones; astype(uint8)
        # TRUNCATES. int() truncates too, so this reproduces it exactly.
        fill = []
        for channel in range(3):
            ordered = sorted(corner[channel] for corner in corners)
            fill.append(int((ordered[1] + ordered[2]) / 2))
        return ImageOps.expand(rgb, border=(left, top, right, bottom), fill=tuple(fill))


def square_pad(image: Image.Image) -> Image.Image:
    return SquarePad()(image)


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Torch transform identical to the deployed one. Imports torchvision lazily so
    that the Mac-side data preparation needs only Pillow."""
    from torchvision import transforms

    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
