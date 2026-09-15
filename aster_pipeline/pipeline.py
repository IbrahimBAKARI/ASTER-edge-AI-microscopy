from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import torch

from .artifact_writer import SessionArtifactWriter, atomic_write_bytes
from .block2.loader import load_block2_scorer
from .config import PipelineConfig
from .device import resolve_device
from .image_io import discover_images
from .schemas import PipelineResult
from .timing import Timings
from .transforms import build_eval_transform
from YOLO import WBCDetector


# Stage-bucket names used in ``timing_ms`` for early-exit sessions (0 or too
# few detected WBC to attempt block 2), so consumers of benchmark output
# always find every key from the completed-session shape, with a value of 0.
_SKIPPED_STAGE_KEYS = {
    False: ("crop_preparation", "mil", "calibration_and_decision"),
    True: (
        "crop_preprocessing_ms",
        "block2_encode_ms",
        "block2_heads_ms",
        "block2_grid_ms",
        "result_rendering_ms",
    ),
}


class LeukemiaPipeline:
    def __init__(
        self, config: PipelineConfig, aggregator: str = "max", device: str = "auto", mil_backend: str = "auto"
    ) -> None:
        if config.block2 is None:
            raise ValueError(
                "config.block2 is not set; add a block2: section to config/inference.yaml"
            )
        self.config = config
        self.aggregator = aggregator
        self.device = resolve_device(device)
        self.block2_backend = "pytorch"
        self.backend_warning: str | None = None
        self.artifact_writer = SessionArtifactWriter()
        self.transform = build_eval_transform(config.block2.image_size)
        self.grid_sha256 = hashlib.sha256(config.block2.grid.read_bytes()).hexdigest()

        engine_path = config.block2.encoder_engine
        wants_tensorrt = mil_backend in ("tensorrt", "auto") and self.device.type == "cuda"
        self.block2 = None
        if wants_tensorrt and engine_path.is_file():
            try:
                self.block2 = load_block2_scorer(config.block2, device="cuda", backend="tensorrt")
                self.block2_backend = "tensorrt_fp16"
            except Exception as exc:
                if mil_backend == "tensorrt":
                    raise
                self.backend_warning = f"TensorRT unavailable; PyTorch fallback was used: {exc}"
        elif mil_backend == "tensorrt":
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        if self.block2 is None:
            torch_device = "cuda" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu"
            self.device = torch.device(torch_device)
            self.block2 = load_block2_scorer(config.block2, device=torch_device, backend="torch")
            self.block2_backend = "pytorch"

        yolo_config = config.yolo
        if (
            self.block2_backend == "tensorrt_fp16"
            and config.yolo.tensorrt_weights is not None
            and config.yolo.tensorrt_weights.is_file()
        ):
            yolo_config = replace(config.yolo, weights=config.yolo.tensorrt_weights)
        try:
            self.detector = WBCDetector(yolo_config, str(self.device), config.output)
        except Exception as exc:
            if yolo_config.weights == config.yolo.weights:
                raise
            self.backend_warning = f"TensorRT YOLO unavailable; PyTorch YOLO fallback was used: {exc}"
            self.detector = WBCDetector(config.yolo, str(self.device), config.output)

    def run(
        self, input_path: str | Path, session_id: str, output_dir: str | Path, benchmark: bool = False
    ) -> PipelineResult:
        started = perf_counter_ns()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        logger = _configure_logger(output / "run.log", session_id)
        # ``benchmark=False`` (the default, used by every production caller)
        # must reproduce current behaviour exactly: no CUDA sync, no extra
        # stage buckets, single "yolo"/"mil"/... keys, one artifact submit.
        synchronize = torch.cuda.synchronize if benchmark and torch.cuda.is_available() else None
        timings = Timings(synchronize=synchronize)
        warnings_list: list[str] = []
        if self.backend_warning:
            warnings_list.append(self.backend_warning)
        with timings.measure("session_image_loading_ms" if benchmark else "image_loading"):
            image_paths = discover_images(input_path, self.config.image_extensions)
        logger.info("Loaded %d field(s)", len(image_paths))
        if benchmark:
            detections = self.detector.detect(
                image_paths,
                output / "annotated",
                output / "crops" if self.config.save_crops else None,
                write_artifact=self._queue_artifact,
                timings=timings,
            )
        else:
            with timings.measure("yolo"):
                detections = self.detector.detect(
                    image_paths,
                    output / "annotated",
                    output / "crops" if self.config.save_crops else None,
                    write_artifact=self._queue_artifact,
                )
        if benchmark:
            with timings.measure("output_persistence_ms"):
                self._queue_manifest(output / "crop_manifest.csv", detections, self.config.yolo.padding)
        else:
            self._queue_manifest(output / "crop_manifest.csv", detections, self.config.yolo.padding)

        if not detections:
            warnings_list.append("No WBC was detected; block 2 was not executed and no session-level classification is available.")
            for key in _SKIPPED_STAGE_KEYS[benchmark]:
                timings.values.setdefault(key, 0.0)
            timings.values["total_analysis_ms" if benchmark else "total"] = (perf_counter_ns() - started) / 1_000_000.0
            result = PipelineResult(
                session_id=session_id,
                status="no_decision",
                label="no_decision",
                tier="none",
                number_of_fields=len(image_paths),
                number_of_detected_wbc=0,
                device=self.device.type,
                inference_backend=self.block2_backend,
                grid_sha256=self.grid_sha256,
                warnings=warnings_list,
                timing_ms=timings.rounded(),
            )
            self._persist_result(output / "result.json", result, timings, benchmark)
            return result

        # No pre-filter on the raw detected count here: the grid's own R0 rule (on
        # N_c, the number of CLASSIFIED leukocytes, which can only be <= the raw
        # count) is what decides insufficient_evidence/out_of_domain/tier -- exactly
        # the same encode-then-decide flow as the verified kit SessionScorer, so a
        # borderline session (e.g. the ~95-crop ALL-IDB benchmark fixture) still
        # exercises the full block 2 pass and produces a meaningful latency number.
        with timings.measure("crop_preprocessing_ms" if benchmark else "crop_preparation"):
            tensors = torch.stack([self.transform(detection.image_rgb) for detection in detections])
        with timings.measure("block2_encode_ms" if benchmark else "mil", gpu=benchmark):
            session_result, _detail = self.block2.score(
                tensors, session_id=session_id, number_of_fields=len(image_paths)
            )
        # ``score()`` already timed its own encode/heads/grid split internally
        # (``session_result.timing_ms``); split the single wall-clock measurement
        # above into the three non-overlapping stages the benchmark scripts read,
        # so they still sum back to exactly what was measured.
        if benchmark:
            heads_ms = float(session_result.timing_ms.get("heads_ms", 0.0))
            grid_ms = float(session_result.timing_ms.get("grid_ms", 0.0))
            combined = timings.values.get("block2_encode_ms", 0.0)
            timings.values["block2_encode_ms"] = max(combined - heads_ms - grid_ms, 0.0)
            timings.values["block2_heads_ms"] = heads_ms
            timings.values["block2_grid_ms"] = grid_ms
        timings.values["total_analysis_ms" if benchmark else "total"] = (perf_counter_ns() - started) / 1_000_000.0
        result = PipelineResult(
            session_id=session_id,
            status=session_result.status,
            label=session_result.label,
            tier=session_result.tier,
            final_calibrated_probability=session_result.uncertainty.get("p_abn"),
            flags=session_result.flags,
            reasons=session_result.reasons,
            population_profile=session_result.population_profile,
            quantities=session_result.quantities,
            verdicts=session_result.verdicts,
            uncertainty=session_result.uncertainty,
            evidence=session_result.evidence,
            number_of_fields=len(image_paths),
            number_of_detected_wbc=len(detections),
            number_of_classified_leukocytes=session_result.number_of_classified_leukocytes,
            device=self.device.type,
            inference_backend=self.block2_backend,
            grid_sha256=self.grid_sha256,
            warnings=warnings_list,
            timing_ms=timings.rounded(),
        )
        if benchmark:
            with timings.measure("result_rendering_ms"):
                result.to_dict()
            # ``timing_ms`` was snapshotted above, before this block ran -- refresh
            # it so the persisted file carries its own result_rendering_ms figure.
            result.timing_ms = timings.rounded()
        self._persist_result(output / "result.json", result, timings, benchmark)
        logger.info("Completed with label=%s, tier=%s", session_result.label, session_result.tier)
        return result

    def _persist_result(self, path: Path, result: PipelineResult, timings: Timings, benchmark: bool) -> None:
        """Queue the single, final ``result.json`` write for this session.

        ``result.timing_ms`` was already snapshotted onto ``result`` before this
        call, so it cannot include this call's own duration -- the same
        self-referential limit that already applies to ``total``/
        ``total_analysis_ms``. This intentionally writes the file exactly
        once; do not add a second write to "fix up" a missing timing key.
        """
        if benchmark:
            with timings.measure("output_persistence_ms"):
                self._queue_result(path, result)
        else:
            self._queue_result(path, result)

    def flush_artifacts(self, timeout: float | None = None) -> float:
        """Wait for every previously scheduled artefact (use before shutdown).

        Returns the elapsed ``artifact_flush_ms``: the time until every queued
        crop, annotation, manifest and result.json for this pipeline is
        fsync'd and atomically renamed into place. This is distinct from
        ``result.json``'s own ``output_persistence_ms``, which only measures
        the time spent encoding and enqueueing artefacts onto the background
        writer, not their actual completion.
        """
        started = perf_counter_ns()
        self.artifact_writer.flush(timeout)
        return (perf_counter_ns() - started) / 1_000_000.0

    def close(self) -> None:
        self.artifact_writer.close()

    def _queue_artifact(self, path: Path, payload: bytes) -> None:
        self.artifact_writer.submit(lambda: atomic_write_bytes(path, payload))

    def _queue_result(self, path: Path, result: PipelineResult) -> None:
        self._queue_artifact(path, json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"))

    def _queue_manifest(self, path: Path, detections, padding: float) -> None:
        fields = ["crop_id", "source_image", "crop_path", "score", "class_id", "x1", "y1", "x2", "y2", "padding"]
        rows = []
        for detection in detections:
            x1, y1, x2, y2 = detection.crop_box
            rows.append({"crop_id": detection.crop_id, "source_image": detection.source_image,
                         "crop_path": detection.crop_path or "", "score": detection.score,
                         "class_id": detection.class_id, "x1": x1, "y1": y1, "x2": x2,
                         "y2": y2, "padding": padding})
        def write() -> None:
            from io import StringIO
            stream = StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            atomic_write_bytes(path, stream.getvalue().encode("utf-8"))
        self.artifact_writer.submit(write)


def _configure_logger(path: Path, session_id: str) -> logging.Logger:
    logger = logging.getLogger(f"leukemia_pipeline.{session_id}.{id(path)}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
