from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class YoloConfig:
    weights: Path
    tensorrt_weights: Path | None = None
    confidence: float = 0.50
    iou: float = 0.50
    image_size: int = 640
    padding: float = 0.10
    class_id: int = 0


@dataclass(frozen=True)
class OutputConfig:
    """Encoding of the persisted overlay and crop images.

    The annotated overlay is a non-authoritative visualisation and defaults to
    JPEG. The crops are the retained audit evidence and default to PNG, which is
    always pixel-lossless regardless of ``crop_png_compress_level`` (that knob
    only trades file size for encode speed). ``lossless_archival`` forces both
    products to maximally-compressed lossless PNG for a reproducibility archive.

    Full-field overlays encoded as PNG zlib level 6 (Pillow's default) cost
    ~1.8 s each on the Orin Nano CPU -- this is the historical ~10.5 s
    ``output_persistence_ms`` bottleneck. JPEG q92 brings that to ~30 ms with no
    consequence for traceability (the crops stay pixel-exact PNG).
    """

    annotated_format: str = "jpeg"
    annotated_jpeg_quality: int = 92
    crop_format: str = "png"
    crop_png_compress_level: int = 1
    crop_jpeg_quality: int = 95
    lossless_archival: bool = False

    def _spec(self, fmt: str, jpeg_quality: int, png_compress_level: int) -> tuple[str, str, dict]:
        if self.lossless_archival or fmt == "png":
            return (".png", "PNG", {"compress_level": 6 if self.lossless_archival else png_compress_level})
        if fmt in ("jpg", "jpeg"):
            return (".jpg", "JPEG", {"quality": jpeg_quality})
        raise ValueError(f"Unsupported image format: {fmt!r}")

    def annotated_spec(self) -> tuple[str, str, dict]:
        return self._spec(self.annotated_format, self.annotated_jpeg_quality, 6)

    def crop_spec(self) -> tuple[str, str, dict]:
        return self._spec(self.crop_format, self.crop_jpeg_quality, self.crop_png_compress_level)


@dataclass(frozen=True)
class Block2Config:
    """Block 2: single ResNet18 encoder + cell head + gated-attention MIL + the
    frozen decision grid. ``encoder_engine`` is derived (never portable across
    machines) rather than read from YAML; it is built on-device by
    ``scripts/build_block2_trt.sh`` next to ``encoder_onnx``.
    """

    encoder_onnx: Path
    heads: Path
    grid: Path
    thresholds: Path
    ood_stats: Path
    quantifier: Path
    image_size: int = 224
    embedding_size: int = 512
    encode_chunk: int = 64
    bag_size: int = 200
    min_classified_cells_screening: int = 100
    min_classified_cells_pattern: int = 200
    min_classified_cells_reference: int = 400
    gamma: float = 0.90
    seed: int = 42

    @property
    def encoder_engine(self) -> Path:
        return self.encoder_onnx.with_name("encoder_fp16.engine")


@dataclass(frozen=True)
class PipelineConfig:
    workspace_root: Path
    yolo: YoloConfig
    block2: Block2Config | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    image_extensions: tuple[str, ...] = field(
        default=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    )
    save_crops: bool = True


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _output_config(raw: dict[str, Any]) -> OutputConfig:
    config = OutputConfig(
        annotated_format=str(raw.get("annotated_format", OutputConfig.annotated_format)).lower(),
        annotated_jpeg_quality=int(raw.get("annotated_jpeg_quality", OutputConfig.annotated_jpeg_quality)),
        crop_format=str(raw.get("crop_format", OutputConfig.crop_format)).lower(),
        crop_png_compress_level=int(raw.get("crop_png_compress_level", OutputConfig.crop_png_compress_level)),
        crop_jpeg_quality=int(raw.get("crop_jpeg_quality", OutputConfig.crop_jpeg_quality)),
        lossless_archival=bool(raw.get("lossless_archival", OutputConfig.lossless_archival)),
    )
    if config.annotated_format not in ("png", "jpg", "jpeg") or config.crop_format not in ("png", "jpg", "jpeg"):
        raise ValueError("output image formats must be one of: png, jpg, jpeg")
    if not 0 <= config.crop_png_compress_level <= 9:
        raise ValueError("crop_png_compress_level must be between 0 and 9")
    return config


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    workspace = _resolve(config_path.parent, raw.get("workspace_root", "../.."))
    yolo_raw = raw.get("yolo", {})
    block2_raw = raw.get("block2")
    block2 = (
        Block2Config(
            encoder_onnx=_resolve(workspace, block2_raw["encoder"]),
            heads=_resolve(workspace, block2_raw["heads"]),
            grid=_resolve(workspace, block2_raw["grid"]),
            thresholds=_resolve(workspace, block2_raw["thresholds"]),
            ood_stats=_resolve(workspace, block2_raw["ood_stats"]),
            quantifier=_resolve(workspace, block2_raw["quantifier"]),
            image_size=int(block2_raw.get("image_size", 224)),
            embedding_size=int(block2_raw.get("embedding_size", 512)),
            encode_chunk=int(block2_raw.get("encode_chunk", 64)),
            bag_size=int(block2_raw.get("bag_size", 200)),
            min_classified_cells_screening=int(block2_raw.get("min_classified_cells_screening", 100)),
            min_classified_cells_pattern=int(block2_raw.get("min_classified_cells_pattern", 200)),
            min_classified_cells_reference=int(block2_raw.get("min_classified_cells_reference", 400)),
            gamma=float(block2_raw.get("gamma", 0.90)),
            seed=int(block2_raw.get("seed", 42)),
        )
        if block2_raw
        else None
    )
    return PipelineConfig(
        workspace_root=workspace,
        yolo=YoloConfig(
            weights=_resolve(workspace, yolo_raw["weights"]),
            tensorrt_weights=(
                _resolve(workspace, yolo_raw["tensorrt_weights"])
                if yolo_raw.get("tensorrt_weights")
                else None
            ),
            confidence=float(yolo_raw.get("confidence", 0.50)),
            iou=float(yolo_raw.get("iou", 0.50)),
            image_size=int(yolo_raw.get("image_size", 640)),
            padding=float(yolo_raw.get("padding", 0.10)),
            class_id=int(yolo_raw.get("class_id", 0)),
        ),
        output=_output_config(raw.get("output", {}) or {}),
        block2=block2,
        image_extensions=tuple(x.lower() for x in raw.get("image_extensions", PipelineConfig.__dataclass_fields__["image_extensions"].default)),
        save_crops=bool(raw.get("save_crops", True)),
    )
