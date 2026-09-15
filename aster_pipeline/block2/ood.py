"""Domain gate: Mahalanobis distance on encoder features.

PREREGISTRATION.md R0 and section 5.2. Fitted on in-domain development bags at a 99 %
in-domain pass rate, and stored as a mean vector plus a tied precision matrix - one
matmul at inference, which is why the gate is affordable on the Jetson.

The gate has three outcomes, not two (PREREGISTRATION.md section 1.2):
  in_domain             the decision proceeds
  domain_not_validated  the domain is recognised but no decision was validated on it
                        -- this is the prototype x40 case today
  out_of_domain         the features match no known acquisition domain
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALIDATED = "in_domain"
NOT_VALIDATED = "domain_not_validated"
UNKNOWN = "out_of_domain"


@dataclass
class DomainGate:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float
    known_domains: dict[str, dict] | None = None
    validated_domains: tuple[str, ...] = ("training",)

    def score(self, features: np.ndarray) -> float:
        """Median per-crop squared Mahalanobis distance over the bag.

        The median rather than the mean: one artefact crop must not move the gate.
        """
        delta = features - self.mean
        return float(np.median(np.einsum("ij,jk,ik->i", delta, self.precision, delta)))

    def crop_fraction_out(self, features: np.ndarray) -> float:
        delta = features - self.mean
        distances = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return float((distances > self.threshold).mean())

    def verdict(self, features: np.ndarray) -> tuple[str, float]:
        score = self.score(features)
        if score <= self.threshold:
            return VALIDATED, score
        if self.known_domains:
            nearest, best = None, np.inf
            for name, stats in self.known_domains.items():
                delta = features - np.asarray(stats["mean"])
                distance = float(np.median(
                    np.einsum("ij,jk,ik->i", delta, np.asarray(stats["precision"]), delta)))
                if distance < best:
                    nearest, best = name, distance
            if nearest is not None and best <= self.known_domains[nearest]["threshold"]:
                if nearest in self.validated_domains:
                    return VALIDATED, best
                return NOT_VALIDATED, best
        return UNKNOWN, score

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, precision=self.precision, threshold=self.threshold)

    @classmethod
    def fit(cls, features: np.ndarray, bags: list[np.ndarray], pass_rate: float = 0.99,
            shrinkage: float = 1e-3) -> "DomainGate":
        mean = features.mean(0)
        covariance = np.cov(features.T) + shrinkage * np.eye(features.shape[1])
        precision = np.linalg.inv(covariance)
        gate = cls(mean=mean, precision=precision, threshold=np.inf)
        gate.threshold = float(np.percentile([gate.score(bag) for bag in bags], 100 * pass_rate))
        return gate

    @classmethod
    def load(cls, path: Path) -> "DomainGate":
        data = np.load(path)
        return cls(mean=data["mean"], precision=data["precision"], threshold=float(data["threshold"]))
