"""Temperature scaling for the MIL head.

PREREGISTRATION.md section 5.2: fitted on the development calibration split, never on the
patients the head was trained on and never on cAItomorph.

Temperature scaling rather than the deployed per-fold Platt calibrators: one parameter,
it cannot change the ranking (so AUROC is untouched), and it is honest about what it does
- it rescales confidence, it does not improve discrimination. The deployed pipeline chose
the worse discriminator because its transferred calibration looked better; separating the
two is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _logit(probability: float, eps: float = 1e-6) -> float:
    probability = min(max(probability, eps), 1 - eps)
    return math.log(probability / (1 - probability))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def apply(self, probability: float) -> float:
        return 1.0 / (1.0 + math.exp(-_logit(probability) / self.temperature))

    def fit(self, probabilities, labels, grid=None) -> "TemperatureScaler":
        """Minimise the negative log-likelihood over a temperature grid.

        A grid rather than gradient descent: one parameter, a bounded range, and a search
        that is trivially reproducible - no optimiser state to record.
        """
        grid = grid or [0.05 * i for i in range(1, 101)]  # 0.05 .. 5.00
        best, best_nll = 1.0, math.inf
        for temperature in grid:
            nll = 0.0
            for probability, label in zip(probabilities, labels):
                scaled = 1.0 / (1.0 + math.exp(-_logit(probability) / temperature))
                scaled = min(max(scaled, 1e-9), 1 - 1e-9)
                nll -= label * math.log(scaled) + (1 - label) * math.log(1 - scaled)
            if nll < best_nll:
                best, best_nll = temperature, nll
        self.temperature = best
        return self


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    """ECE, the metric the deployed block 2 reported at 0.047 (MAX) and 0.348 (attention)."""
    total, error = len(probabilities), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        chosen = [(p, y) for p, y in zip(probabilities, labels) if (low < p <= high) or (index == 0 and p == 0)]
        if not chosen:
            continue
        confidence = sum(p for p, _ in chosen) / len(chosen)
        accuracy = sum(y for _, y in chosen) / len(chosen)
        error += len(chosen) / total * abs(confidence - accuracy)
    return error
