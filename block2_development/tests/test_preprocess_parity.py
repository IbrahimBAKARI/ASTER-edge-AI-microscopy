"""The block-1 -> block-2 contract must not drift.

Skipped when the deployed repository is not beside this one.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aster_block2.preprocess import SquarePad as OurSquarePad  # noqa: E402

DEPLOYED = Path(__file__).resolve().parents[2]   # repository root: the deployed aster_pipeline/


def _deployed_importable() -> bool:
    """The deployed transforms import numpy and torchvision; skip where they are absent
    (the Mac prep venv is deliberately Pillow-only). The parity test runs in Colab and in
    the CPU env, which is where it matters."""
    if not (DEPLOYED / "aster_pipeline" / "transforms.py").exists():
        return False
    sys.path.insert(0, str(DEPLOYED))
    try:
        import aster_pipeline.transforms  # noqa: F401
        return True
    except Exception:
        return False


DEPLOYED_OK = _deployed_importable()


@pytest.mark.skipif(not DEPLOYED_OK, reason="deployed repository not importable here")
def test_square_pad_is_byte_identical():
    sys.path.insert(0, str(DEPLOYED))
    from aster_pipeline.transforms import SquarePad as DeployedSquarePad

    ours, theirs = OurSquarePad(), DeployedSquarePad()
    generator = random.Random(42)
    for _ in range(200):
        width = generator.randint(9, 300)
        height = generator.randint(9, 300)
        image = Image.new("RGB", (width, height))
        image.putdata([(generator.randrange(256), generator.randrange(256),
                        generator.randrange(256)) for _ in range(width * height)])
        assert ours(image).tobytes() == theirs(image).tobytes(), (width, height)


@pytest.mark.skipif(not DEPLOYED_OK, reason="deployed repository not importable here")
def test_constants_match():
    sys.path.insert(0, str(DEPLOYED))
    from aster_pipeline import transforms as deployed
    from aster_block2 import preprocess as ours

    assert ours.IMAGENET_MEAN == deployed.IMAGENET_MEAN
    assert ours.IMAGENET_STD == deployed.IMAGENET_STD


@pytest.mark.skipif(not DEPLOYED_OK, reason="deployed repository not importable here")
def test_crop_extraction_is_identical():
    """A full field becomes crops the same way on the Mac, in Colab and on the Jetson."""
    sys.path.insert(0, str(DEPLOYED))
    from aster_pipeline.crop_extraction import expand_and_clip_box as deployed_expand
    from aster_pipeline.crop_extraction import extract_crop as deployed_extract

    from aster_block2.crops import expand_and_clip_box as ours_expand
    from aster_block2.crops import extract_crop as ours_extract

    generator = random.Random(1)
    for _ in range(20000):
        width = generator.randint(50, 4032)
        height = generator.randint(50, 3040)
        x1 = generator.uniform(0, width)
        y1 = generator.uniform(0, height)
        x2 = x1 + generator.uniform(0, width - x1)
        y2 = y1 + generator.uniform(0, height - y1)
        box = (x1, y1, x2, y2)
        assert ours_expand(box, width, height) == deployed_expand(box, width, height)

    image = Image.new("RGB", (200, 150), (10, 20, 30))
    for box in ((0, 0, 0, 0), (10, 10, 5, 50), (10, 10, 60, 60)):
        ours = ours_extract(image, box)
        theirs = deployed_extract(image, box)
        assert (ours is None) == (theirs is None)
        if ours is not None:
            assert ours.tobytes() == theirs.tobytes()
