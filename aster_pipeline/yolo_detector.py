from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from io import BytesIO
from typing import Callable

from PIL import ImageDraw
from ultralytics import YOLO

from .config import OutputConfig, YoloConfig
from .crop_extraction import expand_and_clip_box, extract_crop
from .image_io import load_rgb
from .schemas import Detection
from .timing import Timings

# libpng / libjpeg release the GIL during encoding, so a small thread pool turns
# the per-field crop encodes into concurrent work. Bounded to keep the footprint
# predictable on the 8 GB board; override with ASTER_PERSIST_WORKERS.
_PERSIST_WORKERS = max(1, int(os.environ.get("ASTER_PERSIST_WORKERS", "4")))


class YoloWBCDetector:
    """Single-class WBC localizer. This class never produces cell diagnoses."""

    def __init__(self, config: YoloConfig, device: str, output: OutputConfig | None = None) -> None:
        if not config.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {config.weights}")
        self.config = config
        self.device = device
        self.output = output or OutputConfig()
        self.model = YOLO(str(config.weights), task="detect")
        # Raw TensorRT plans do not carry Ultralytics' optional class-name metadata.
        # The engine is nevertheless structurally single-class (output channels = 5)
        # and is parity-checked against this exact WBC checkpoint.
        if config.weights.suffix != ".engine":
            names = {k: str(v).lower() for k, v in self.model.names.items()}
            if names != {config.class_id: "wbc"}:
                raise ValueError(f"Expected a single WBC class, found: {self.model.names}")

    def detect(
        self,
        image_paths: list[Path],
        annotated_dir: Path,
        crops_dir: Path | None,
        write_artifact: Callable[[Path, bytes], object] | None = None,
        timings: Timings | None = None,
    ) -> list[Detection]:
        """Detect fields, optionally deferring writes and/or recording timings.

        ``write_artifact``, when supplied, receives a destination and encoded
        image bytes instead of a direct synchronous save -- the pipeline uses
        this to queue the fsync+rename on a background writer. The encoding
        itself (the costly step for full-field overlays) happens here, under
        ``output_persistence_ms``, in the container/quality given by
        ``OutputConfig`` (JPEG q92 overlays, pixel-exact PNG level 1 crops).
        ``timings``, when supplied, records per-stage benchmark buckets
        (``yolo_inference_ms``, ``wbc_crop_extraction_ms``,
        ``output_persistence_ms``). The two parameters are independent.
        """
        crop_ext, crop_format, crop_kwargs = self.output.crop_spec()
        annotated_ext, annotated_format, annotated_kwargs = self.output.annotated_spec()
        if write_artifact is None:
            annotated_dir.mkdir(parents=True, exist_ok=True)
            if crops_dir is not None:
                crops_dir.mkdir(parents=True, exist_ok=True)
        # One field at a time supports the deployed static batch-1 TensorRT engine
        # and matches the live acquisition workflow.
        results = []
        for path in image_paths:
            context = timings.measure("yolo_inference_ms", gpu=True) if timings else nullcontext()
            with context:
                results.extend(self.model.predict(
                    source=str(path),
                    imgsz=self.config.image_size,
                    conf=self.config.confidence,
                    iou=self.config.iou,
                    classes=[self.config.class_id],
                    device=self.device,
                    verbose=False,
                    stream=False,
                ))
        detections: list[Detection] = []
        for field_index, (path, result) in enumerate(zip(image_paths, results)):
            context = timings.measure("wbc_crop_extraction_ms") if timings else nullcontext()
            with context:
                image = load_rgb(path)
                draw = ImageDraw.Draw(image)
                raw_boxes = [] if result.boxes is None else list(result.boxes)
                valid_index = 0
                field_detections: list[Detection] = []
                for box in raw_boxes:
                    score = float(box.conf.detach().cpu().reshape(-1)[0])
                    class_id = int(box.cls.detach().cpu().reshape(-1)[0])
                    if score < self.config.confidence or class_id != self.config.class_id:
                        continue
                    coordinates = tuple(float(v) for v in box.xyxy.detach().cpu().reshape(-1).tolist())
                    crop_box = expand_and_clip_box(coordinates, image.width, image.height, self.config.padding)
                    crop = extract_crop(image, crop_box)
                    if crop is None:
                        continue
                    crop_id = f"field{field_index:03d}_{path.stem}_wbc_{valid_index:04d}"
                    crop_path = crops_dir / f"{crop_id}{crop_ext}" if crops_dir is not None else None
                    field_detections.append(
                        Detection(
                            crop_id=crop_id,
                            source_image=path,
                            score=score,
                            class_id=class_id,
                            box=coordinates,
                            crop_box=crop_box,
                            crop_path=crop_path,
                            image_rgb=crop.copy(),
                        )
                    )
                    x1, y1, x2, y2 = coordinates
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 80), width=3)
                    draw.text((x1, max(0, y1 - 14)), f"WBC {score:.2f}", fill=(0, 220, 80))
                    valid_index += 1
            detections.extend(field_detections)
            context = timings.measure("output_persistence_ms") if timings else nullcontext()
            with context:
                def _encode_crop(detection: Detection) -> tuple[Path, bytes]:
                    image_rgb = detection.image_rgb
                    if crop_format == "JPEG" and image_rgb.mode != "RGB":
                        image_rgb = image_rgb.convert("RGB")
                    buffer = BytesIO()
                    image_rgb.save(buffer, format=crop_format, **crop_kwargs)
                    return detection.crop_path, buffer.getvalue()

                pending = [d for d in field_detections if d.crop_path is not None]
                if len(pending) > 1 and _PERSIST_WORKERS > 1:
                    with ThreadPoolExecutor(max_workers=min(_PERSIST_WORKERS, len(pending))) as pool:
                        encoded = list(pool.map(_encode_crop, pending))
                else:
                    encoded = [_encode_crop(d) for d in pending]
                for crop_path, payload in encoded:
                    if write_artifact is None:
                        crop_path.write_bytes(payload)
                    else:
                        write_artifact(crop_path, payload)

                annotated_path = annotated_dir / f"{field_index:03d}_{path.stem}_annotated{annotated_ext}"
                annotated = image
                if annotated_format == "JPEG" and image.mode != "RGB":
                    annotated = image.convert("RGB")
                payload = BytesIO()
                annotated.save(payload, format=annotated_format, **annotated_kwargs)
                if write_artifact is None:
                    annotated_path.write_bytes(payload.getvalue())
                else:
                    write_artifact(annotated_path, payload.getvalue())
        return detections

    def detect_frame_bgr(self, frame):
        """Return live WBC boxes as ``(x1, y1, x2, y2, confidence)``."""
        results = self.model.predict(
            source=frame, imgsz=self.config.image_size, conf=self.config.confidence,
            iou=self.config.iou, classes=[self.config.class_id], device=self.device,
            verbose=False, stream=False,
        )
        boxes = []
        for result in results:
            for box in ([] if result.boxes is None else list(result.boxes)):
                score = float(box.conf.detach().cpu().reshape(-1)[0])
                class_id = int(box.cls.detach().cpu().reshape(-1)[0])
                if class_id == self.config.class_id and score >= self.config.confidence:
                    x1, y1, x2, y2 = box.xyxy.detach().cpu().reshape(-1).tolist()
                    boxes.append((float(x1), float(y1), float(x2), float(y2), score))
        return boxes
