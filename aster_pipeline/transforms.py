from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SquarePad:
    """Notebook-exact square padding using the median of the four RGB corners."""

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        corners = np.asarray(
            [
                rgb.getpixel((0, 0)),
                rgb.getpixel((max(0, width - 1), 0)),
                rgb.getpixel((0, max(0, height - 1))),
                rgb.getpixel((max(0, width - 1), max(0, height - 1))),
            ]
        )
        fill = tuple(np.median(corners, axis=0).astype(np.uint8).tolist())
        return ImageOps.expand(rgb, border=(left, top, right, bottom), fill=fill)


def build_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
