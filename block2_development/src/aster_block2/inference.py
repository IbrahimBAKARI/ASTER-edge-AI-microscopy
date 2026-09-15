"""End-to-end block 2: a bag of crop paths (or tensors) -> the frozen grid's verdict.

This module is the reason the Colab notebook and the Jetson agree. The notebook calls
`SessionScorer.score` and so does `integration/`; there is no second implementation to
drift. The same discipline as the crop->tensor contract, one level up.

Encode once, decide many times: one encoder pass per unique crop, then every head reads
the cached [N, 512] matrix. The deployed block 2 ran the encoder 750 times per session
for ~183 unique crops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .calibration import TemperatureScaler
from .grid import GridResult, Thresholds, evaluate
from .ood import NOT_VALIDATED, UNKNOWN, DomainGate
from .quantify import Quantifier
from .preprocess import build_eval_transform
from .sampling import deterministic_bag
from .schema import SessionResult

CELL_CLASSES = [
    "myeloblast", "lymphoblast", "promyelocyte", "promyelocyte_abnormal",
    "myelocyte", "metamyelocyte", "band_neutrophil", "segmented_neutrophil",
    "basophil", "eosinophil", "monocyte", "lymphocyte", "lymphocyte_atypical",
    "smudge_cell", "erythroblast", "other_artifact",
]
BLAST_LINEAGE = ("lymphoblast", "myeloblast")


@dataclass
class SessionScorer:
    encoder: torch.nn.Module
    cell_head: torch.nn.Module
    mil_head: torch.nn.Module
    thresholds: Thresholds
    domain_gate: DomainGate | None = None
    calibrator: TemperatureScaler | None = None
    quantifier: Quantifier | None = None
    device: str = "cpu"
    bag_size: int = 200
    encode_chunk: int = 64
    image_size: int = 224
    seed: int = 42                      # config/inference.yaml -> mil.seed

    def __post_init__(self) -> None:
        self.transform = build_eval_transform(self.image_size)
        self.lymphoblast_index = CELL_CLASSES.index("lymphoblast")
        self.myeloblast_index = CELL_CLASSES.index("myeloblast")
        for module in (self.encoder, self.cell_head, self.mil_head):
            module.eval().to(self.device)

    # -- stage B ------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, crops: Sequence[Path] | torch.Tensor) -> torch.Tensor:
        if isinstance(crops, torch.Tensor):
            tensors = crops
        else:
            tensors = torch.stack([self.transform(Image.open(path).convert("RGB")) for path in crops])
        chunks = [self.encoder(tensors[i:i + self.encode_chunk].to(self.device)).cpu()
                  for i in range(0, len(tensors), self.encode_chunk)]
        return torch.cat(chunks)

    # -- stages B, C, D -----------------------------------------------------
    @torch.inference_mode()
    def score(self, crops: Sequence[Path] | torch.Tensor, session_id: str = "session",
              number_of_fields: int = 0) -> tuple[SessionResult, dict]:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        features = self.encode(crops)
        timings["encode_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        probabilities = self.cell_head(features.to(self.device))["probabilities"].cpu()
        predicted = probabilities.argmax(-1)
        counts = {name: 0 for name in CELL_CLASSES}
        for index in predicted.tolist():
            counts[CELL_CLASSES[index]] += 1

        blast_mask = (predicted == self.lymphoblast_index) | (predicted == self.myeloblast_index)
        if bool(blast_mask.any()):
            lymphoid = float(probabilities[blast_mask, self.lymphoblast_index].sum())
            myeloid = float(probabilities[blast_mask, self.myeloblast_index].sum())
            lineage_post = lymphoid / (lymphoid + myeloid + 1e-9)
        else:
            lineage_post = None

        # The bag is a SEEDED RANDOM sample, exactly as aster_pipeline/sampling.py does.
        # Taking the first `bag_size` crops would take the first fields scanned, and a
        # smear is not homogeneous (feathered edge vs body) - that is a spatial bias, not
        # a sample.
        bag_index = deterministic_bag(len(features), self.bag_size, self.seed)
        output = self.mil_head(features[bag_index].to(self.device))
        raw = float(output["probability"])
        p_abn = self.calibrator.apply(raw) if self.calibrator else raw
        attention = output["attention"].cpu()
        timings["heads_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        ood_score, domain_verdict = None, None
        if self.domain_gate is not None:
            domain_verdict, ood_score = self.domain_gate.verdict(features.numpy())
        grid_result: GridResult = evaluate(
            counts, self.thresholds, n_localized=len(features),
            p_abn=p_abn, lineage_post=lineage_post,
            ood_score=ood_score if domain_verdict in (None, UNKNOWN) else None,
            quantifier=self.quantifier,
        )
        if domain_verdict == NOT_VALIDATED:
            grid_result.label = "out_of_domain"
            grid_result.reasons.insert(0, "acquisition domain recognised but not validated "
                                          "for a decision (prototype x40)")
        timings["grid_ms"] = (time.perf_counter() - start) * 1000

        order = attention.argsort(descending=True)[:10].tolist()
        result = SessionResult(
            session_id=session_id,
            status="completed" if grid_result.label not in
                   ("insufficient_evidence", "out_of_domain") else grid_result.label,
            label=grid_result.label, tier=grid_result.tier, flags=grid_result.flags,
            reasons=grid_result.reasons, quantities=grid_result.quantities,
            verdicts=grid_result.verdicts,
            population_profile={k: v / max(grid_result.n_classified, 1) for k, v in counts.items()},
            uncertainty={"p_abn_raw": raw, "p_abn": p_abn,
                         "ood_score": ood_score if ood_score is not None else float("nan"),
                         "ood_threshold": self.domain_gate.threshold if self.domain_gate else float("nan")},
            evidence={"top_attended": order,
                      "attention_weights": [float(attention[i]) for i in order]},
            number_of_fields=number_of_fields,
            number_of_detected_wbc=len(features),
            number_of_classified_leukocytes=grid_result.n_classified,
            device=self.device, timing_ms=timings,
        )
        detail = {"counts": counts, "features": features, "attention": attention,
                  "lineage_post": lineage_post, "domain_verdict": domain_verdict}
        return result, detail
