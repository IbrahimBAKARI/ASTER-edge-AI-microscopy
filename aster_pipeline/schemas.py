from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class Detection:
    crop_id: str
    source_image: Path
    score: float
    class_id: int
    box: tuple[float, float, float, float]
    crop_box: tuple[int, int, int, int]
    crop_path: Path | None = None
    image_rgb: Any = field(default=None, repr=False)


Status = Literal["completed", "insufficient_evidence", "out_of_domain", "no_decision", "failed"]

# Block 2 v2 label vocabulary that maps to the deployed, back-compat ``final_label``.
_LEGACY_AML_LABELS = {"acute_blastic__myeloid_oriented"}
_LEGACY_NEGATIVE_LABELS = {"non_leukemic"}


def legacy_final_label(label: str | None) -> Literal["AML", "non_leukemic_control", "no_decision"]:
    """Map block 2 v2's label vocabulary onto the deployed three-value ``final_label``,
    so ``Interface/model_integration.py`` keeps comparing against ``"AML"`` unchanged."""
    if label in _LEGACY_AML_LABELS:
        return "AML"
    if label in _LEGACY_NEGATIVE_LABELS:
        return "non_leukemic_control"
    return "no_decision"


@dataclass
class PipelineResult:
    session_id: str
    status: Status
    label: str = "no_decision"
    tier: str = "none"
    final_calibrated_probability: float | None = None
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    population_profile: dict[str, float] = field(default_factory=dict)
    quantities: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    number_of_fields: int = 0
    number_of_detected_wbc: int = 0
    number_of_classified_leukocytes: int = 0
    device: str = "cpu"
    inference_backend: str = "pytorch"
    grid_sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def final_label(self) -> Literal["AML", "non_leukemic_control", "no_decision"]:
        return legacy_final_label(self.label)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_label"] = self.final_label  # legacy, last so it is never dropped
        return payload
