"""Adapter between the workspace UI payload and ASTER's production backend."""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from backend import ASTERBackend, AnalysisRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "inference.yaml"

# Block 2's label vocabulary (aster_pipeline/block2/grid.py) -> UI display text,
# risk tone, and (for the acute/chronic patterns) the quantity whose fraction
# of classified leukocytes is shown as "positive cells".
_LABEL_DISPLAY = {
    "non_leukemic": "No Leukemic Pattern",
    "suspicious_for_neoplasm": "Suspicious for Neoplasm",
    "acute_blastic__myeloid_oriented": "Acute Myeloid Pattern",
    "acute_blastic__lymphoid_oriented": "Acute Lymphoid Pattern",
    "acute_blastic__lineage_indeterminate": "Acute Pattern, Lineage Uncertain",
    "chronic_myeloid_pattern": "Chronic Myeloid Pattern",
    "chronic_lymphoid_pattern": "Chronic Lymphoid Pattern",
    "indeterminate": "Indeterminate",
}
_HIGH_RISK_LABELS = {
    "suspicious_for_neoplasm",
    "acute_blastic__myeloid_oriented",
    "acute_blastic__lymphoid_oriented",
    "acute_blastic__lineage_indeterminate",
    "chronic_myeloid_pattern",
    "chronic_lymphoid_pattern",
}
_EVIDENCE_QUANTITY_BY_LABEL = {
    "acute_blastic__myeloid_oriented": "blast_frac",
    "acute_blastic__lymphoid_oriented": "blast_frac",
    "acute_blastic__lineage_indeterminate": "blast_frac",
    "chronic_myeloid_pattern": "ig_frac",
    "chronic_lymphoid_pattern": "lymph_frac",
}


class WorkspaceLeukemiaModelRunner:
    """Run the local ASTER pipeline and translate its result for the UI."""

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self._backend: ASTERBackend | None = None

    def __call__(
        self,
        images: list[np.ndarray | str | Path],
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        return self.analyze_full_images(images, progress_callback)

    def analyze_full_images(
        self,
        images: list[np.ndarray | str | Path],
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        if not images:
            return self._empty_result()

        started = perf_counter()
        self._progress(progress_callback, 5, "Preparing microscopy fields...")
        with tempfile.TemporaryDirectory(prefix="aster_workspace_") as temporary:
            temporary_root = Path(temporary)
            input_dir = temporary_root / "inputs"
            output_dir = temporary_root / "outputs"
            input_dir.mkdir()
            source_frames = self._materialize_images(images, input_dir, progress_callback)

            self._progress(progress_callback, 18, "Running ASTER detection and session inference...")
            backend = self._get_backend()
            pipeline_result = backend.analyze(
                AnalysisRequest(input_dir, "workspace-session", output_dir)
            )
            self._progress(progress_callback, 92, "Loading annotated detection previews...")
            # This bridge removes its temporary session directory on return, so
            # wait here only for its own previews.  Persistent backend callers
            # can return immediately and flush on application shutdown.
            backend.flush_artifacts()
            display_frames = self._load_annotated_frames(output_dir / "annotated") or source_frames

        result = self._ui_result(pipeline_result.to_dict(), len(source_frames))
        result["_display_frames"] = display_frames
        result["timing_details"] = {
            "backend_seconds": round(pipeline_result.timing_ms.get("total", 0.0) / 1000, 3),
            "total_seconds": round(perf_counter() - started, 3),
        }
        self._progress(progress_callback, 99, "Finalizing result...")
        return result

    def _get_backend(self) -> ASTERBackend:
        if self._backend is None:
            self._backend = ASTERBackend(CONFIG_PATH, device=self.device)
        return self._backend

    @staticmethod
    def _preview_copy(frame: np.ndarray, max_side: int = 1200) -> np.ndarray:
        longest = max(frame.shape[0], frame.shape[1])
        if longest <= max_side:
            return frame.copy()
        scale = max_side / float(longest)
        return cv2.resize(
            frame,
            (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _materialize_images(self, images, input_dir, progress_callback):
        # Only downscaled copies are kept in memory (display fallback); the
        # analysis itself reads the full-resolution files written here.
        frames: list[np.ndarray] = []
        for index, image in enumerate(images, start=1):
            if isinstance(image, np.ndarray):
                frame = image
            else:
                frame = cv2.imread(str(Path(image).expanduser()))
            if frame is None:
                raise ValueError(f"Unable to read image {index}.")
            # JPEG q97: the Arducam sensor already delivers MJPEG, and PNG of a
            # 12 MP field is ~5x slower to encode on the Orin for no visible gain.
            target = input_dir / f"field_{index:03d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                raise RuntimeError(f"Unable to prepare image {index} for analysis.")
            frames.append(self._preview_copy(frame))
            self._progress(
                progress_callback,
                5 + int(index * 10 / len(images)),
                f"Prepared image {index}/{len(images)}...",
            )
        return frames

    @classmethod
    def _load_annotated_frames(cls, annotated_dir: Path) -> list[np.ndarray]:
        if not annotated_dir.is_dir():
            return []
        return [
            cls._preview_copy(frame)
            for path in sorted(annotated_dir.iterdir())
            if path.is_file() and (frame := cv2.imread(str(path))) is not None
        ]

    @staticmethod
    def _progress(callback, value: int, detail: str) -> None:
        if callback is not None:
            callback(max(0, min(99, value)), detail)

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "label": "No Images",
            "confidence": 0,
            "risk_level": "warning",
            "images_used": 0,
            "summary": "Capture or import at least one microscopy image before running analysis.",
            "total_detected_cells": 0,
            "positive_cells": 0,
            "max_positive_probability": None,
        }

    @staticmethod
    def _ui_result(result: dict[str, Any], images_used: int) -> dict[str, Any]:
        status = result["status"]
        label_code = result.get("label", "no_decision")
        tier = result.get("tier", "none")
        flags = result.get("flags", [])
        reasons = result.get("reasons", [])
        quantities = result.get("quantities", {})
        n_classified = int(result.get("number_of_classified_leukocytes", 0))
        probability = result.get("final_calibrated_probability")
        confidence = int(round(float(probability or 0) * 100))
        detected = int(result["number_of_detected_wbc"])
        warnings = result.get("warnings", [])
        reason_text = "; ".join(reasons)

        if status == "no_decision":
            display_label, risk, summary = (
                "No Detection",
                "warning",
                "No WBC was detected; no session-level classification is available.",
            )
        elif status == "insufficient_evidence":
            display_label, risk, summary = (
                "Insufficient Evidence",
                "indeterminate",
                reason_text or "Below the routine manual differential; acquire more fields.",
            )
        elif status == "out_of_domain":
            display_label, risk, summary = (
                "Outside Validated Domain",
                "warning",
                reason_text or "This session falls outside the validated acquisition domain.",
            )
        elif label_code in _HIGH_RISK_LABELS:
            display_label = _LABEL_DISPLAY.get(label_code, label_code)
            risk = "high"
            summary = reason_text or "Suspicious morphology detected. Expert confirmatory review is recommended."
            if tier == "screening":
                summary = f"{summary} (screening tier: MIL score only, below the full pattern differential)"
        elif label_code == "indeterminate":
            display_label, risk, summary = (
                "Indeterminate",
                "indeterminate",
                reason_text or "The grid could not reach a confident call at this session's cell count.",
            )
        else:  # non_leukemic
            display_label = _LABEL_DISPLAY.get(label_code, label_code)
            risk = "low"
            summary = reason_text or "No suspicious morphology detected. Continue routine expert validation."

        if "APL_suspicion" in flags:
            display_label = f"{display_label} + APL suspicion (urgent)"
            risk = "high"
            summary = f"Abnormal promyelocytes predominate -- urgent, confirm PML::RARA. {summary}"

        if warnings:
            summary = f"{summary} ({'; '.join(warnings)})" if summary else "; ".join(warnings)

        evidence_quantity = _EVIDENCE_QUANTITY_BY_LABEL.get(label_code)
        positive_cells = (
            round(quantities.get(evidence_quantity, 0.0) * n_classified) if evidence_quantity else 0
        )

        return {
            "label": display_label,
            "confidence": confidence,
            "risk_level": risk,
            "images_used": images_used,
            "summary": summary,
            "mode": "aster_session_pipeline",
            "total_detected_cells": detected,
            "positive_cells": positive_cells,
            "max_positive_probability": probability,
            "detection_backend": f"YOLO + ResNet/MIL ({result.get('inference_backend', 'pytorch')})",
            "aster_result": result,
        }


def build_workspace_model_runner(*_args, **_kwargs) -> WorkspaceLeukemiaModelRunner:
    return WorkspaceLeukemiaModelRunner()
