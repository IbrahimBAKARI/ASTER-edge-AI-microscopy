"""Overlay / crop encoding: the persistence-cost fix must not lose crop pixels."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from aster_pipeline.config import OutputConfig, load_config

ROOT_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[1] / "config" / "inference.yaml"


def _sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8), "RGB")


def test_default_specs_are_jpeg_overlay_and_png_crop():
    output = OutputConfig()
    assert output.annotated_spec() == (".jpg", "JPEG", {"quality": 92})
    assert output.crop_spec() == (".png", "PNG", {"compress_level": 1})


def test_lossless_archival_forces_png_everywhere():
    output = OutputConfig(lossless_archival=True)
    assert output.annotated_spec()[:2] == (".png", "PNG")
    assert output.crop_spec()[:2] == (".png", "PNG")
    assert output.crop_spec()[2] == {"compress_level": 6}


def test_png_compress_level_does_not_change_a_single_crop_pixel():
    image = _sample_image()
    encoded = {}
    for level in (1, 6):
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=level)
        encoded[level] = np.asarray(Image.open(BytesIO(buffer.getvalue())))
    assert np.array_equal(encoded[1], encoded[6])
    assert np.array_equal(encoded[1], np.asarray(image))


def test_config_parses_output_block():
    config = load_config(ROOT_CONFIG)
    assert config.output.annotated_format == "jpeg"
    assert config.output.crop_format == "png"
    assert 0 <= config.output.crop_png_compress_level <= 9


def test_config_rejects_bad_compress_level(tmp_path):
    bad = tmp_path / "inference.yaml"
    bad.write_text(
        "workspace_root: .\n"
        "yolo: {weights: w.pt}\n"
        "mil: {checkpoints_dir: c, calibration_bundle: b.joblib}\n"
        "output: {crop_png_compress_level: 42}\n"
    )
    with pytest.raises(ValueError, match="crop_png_compress_level"):
        load_config(bad)
