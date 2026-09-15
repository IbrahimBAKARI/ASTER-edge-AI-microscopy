"""result.json v2. Extends the deployed schema; ``final_label`` is kept and derived,
so Interface/ and backend/service.py of the deployed pipeline keep working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["completed", "insufficient_evidence", "out_of_domain", "no_decision", "failed"]

LEGACY_AML_LABELS = {"acute_blastic__myeloid_oriented"}
LEGACY_NEGATIVE_LABELS = {"non_leukemic"}


def legacy_final_label(label: str) -> str:
    """Map the v2 vocabulary onto the deployed three-value ``final_label``."""
    if label in LEGACY_AML_LABELS:
        return "AML"
    if label in LEGACY_NEGATIVE_LABELS:
        return "non_leukemic_control"
    return "no_decision"


@dataclass
class SessionResult:
    session_id: str
    status: Status
    label: str
    tier: str
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
    timing_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def final_label(self) -> str:
        return legacy_final_label(self.label)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_label"] = self.final_label  # legacy, last so it is never dropped
        return payload
