"""Field -> single-cell crops, using the deployed block-1 convention.

Block 2 never sees a full field. It sees a bag of single-cell crops produced by
`expand_and_clip_box` + `extract_crop`, in native pixels, variable size. Any source that
ships full fields (LeukemiaAttr's 640x640 images with COCO boxes, ALL-IDB1, the prototype
x40 fields) must pass through this module BEFORE the crop->tensor contract of
`preprocess.py` is applied.

The two functions below are byte-identical to
`Software_Dev_Micro_Edge/aster_pipeline/crop_extraction.py`, including the `int()`
truncation, the `width - 1` / `height - 1` clipping, and the rule that a degenerate box
drops the detection entirely rather than yielding a padded stub.
`tests/test_preprocess_parity.py` asserts that equality when the deployed repo is present.

Resizing a full field to 224 instead of cropping it would silently produce a "cell" that
is a whole smear. `guard_is_single_cell` exists so that mistake fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

PADDING = 0.10  # config/inference.yaml -> yolo.padding


def expand_and_clip_box(
    box: tuple[float, float, float, float], width: int, height: int, padding: float = PADDING
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
    crop = image.crop((x1, y1, x2, y2))
    return crop if crop.width > 0 and crop.height > 0 else None


def yolo_to_xyxy(box_norm: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    """YOLO `cx cy w h` normalised -> absolute xyxy."""
    cx, cy, bw, bh = (float(v) for v in box_norm)
    return (
        (cx - bw / 2) * width, (cy - bh / 2) * height,
        (cx + bw / 2) * width, (cy + bh / 2) * height,
    )


def coco_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """COCO `x y w h` -> xyxy."""
    x, y, w, h = (float(v) for v in bbox)
    return (x, y, x + w, y + h)


def crops_from_boxes(
    image_path: Path, boxes: Sequence[tuple[float, float, float, float]], padding: float = PADDING
) -> Iterator[tuple[int, Image.Image]]:
    """Yield `(box_index, crop)` for every box that survives expansion and clipping.

    A degenerate box yields nothing at all, exactly as in the deployed pipeline: the
    detection is dropped, not padded, and it does not count towards the session's N.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        for index, box in enumerate(boxes):
            crop = extract_crop(image, expand_and_clip_box(box, width, height, padding))
            if crop is not None:
                yield index, crop.copy()


# --- guard ------------------------------------------------------------------
# Empirical envelope of the single-cell sources actually in the corpus:
#   cAItomorph 144, AML-MLL 144, Taleqani 224, ALL-IDB2 257, PBC 360x363, AML-LMU 400.
# A field is 640 (LeukemiaAttr), 1712, 2592 (ALL-IDB1) or 4032 (prototype x40).
MAX_SINGLE_CELL_SIDE = 512


def guard_is_single_cell(size: tuple[int, int], source: str, path: Path) -> None:
    """Raise if an image looks like a field and reached the cell pipeline uncropped."""
    if max(size) > MAX_SINGLE_CELL_SIDE:
        raise ValueError(
            f"{source}: {path.name} is {size[0]}x{size[1]}, larger than the "
            f"{MAX_SINGLE_CELL_SIDE} px single-cell envelope. This looks like a full "
            f"field. Block 2 consumes single-cell crops only - give the adapter its "
            f"boxes so crops.crops_from_boxes() runs first, or exclude the source. "
            f"Resizing a field to 224 would produce a 'cell' that is a whole smear."
        )
