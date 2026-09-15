from __future__ import annotations

from PIL import Image


def expand_and_clip_box(
    box: tuple[float, float, float, float], width: int, height: int, padding: float = 0.10
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    px, py = box_width * padding, box_height * padding
    return (
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(width - 1, x2 + px)),
        int(min(height - 1, y2 + py)),
    )


def extract_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image | None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    # PIL's right/lower coordinates are exclusive, matching the notebook's array slice.
    crop = image.crop((x1, y1, x2, y2))
    return crop if crop.width > 0 and crop.height > 0 else None
