"""Session-level recalibration of a grid quantity against a manual differential.

Amendment 9 (PREREGISTRATION.md). Post hoc, declared before it was run.

Why the cell-level quantifier is not enough for `blast_frac`. `quantify.py` inverts TPR
and FPR measured on single validation cells from the cell-level sources. Those rates do
not transfer to whole sessions of another acquisition domain: on the AML-MLL controls,
whose manual differential counts ZERO blasts, the corrected blast fraction still sits near
12 %, and the cAItomorph stem-cell donors behave the same way. R4 (`blast_frac < 5 %`)
then cannot fire on normal blood, whatever the rest of the smear looks like.

The correction is the same model as `quantify.py`, fitted one level up. Per patient,

    observed blast fraction = a + b * manual blast fraction + patient-level scatter

with `a` the session-level false-positive rate and `b` = TPR - FPR at session level. It is
fitted on the 189 AML-MLL patients against their manual 100-cell differential. The cell
head never saw AML-MLL (no cell labels), and cAItomorph takes no part.

Uncertainty that reaches the frozen three-way test:
  - binomial counting of the session, inflated by a dispersion factor `phi` that absorbs
    the patient-to-patient scatter of the classifier's error (quasi-binomial);
  - the estimation error of (a, b), from a patient-level bootstrap.
It is turned into an effective (k, n) through the design effect, exactly as
`Quantifier.effective_counts` does, so the frozen test consumes it unchanged.

The correction itself is pure standard library (it runs on the Jetson); only `fit` needs
numpy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .quantify import MIN_SEPARATION, Quantifier


@dataclass
class SessionCalibration:
    """observed = intercept + slope * true, fitted across patients."""

    intercept: float          # a: observed fraction when the manual count is zero
    slope: float              # b: session-level TPR - FPR
    var_intercept: float
    var_slope: float
    covariance: float
    dispersion: float         # phi >= 1, quasi-binomial scatter between patients
    n_patients: int
    diagnostics: dict = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, observed_count: Sequence[int], n_classified: Sequence[int],
            manual_fraction: Sequence[float], strata: Sequence[str],
            manual_total: Sequence[float] | None = None,
            bootstrap: int = 2000, seed: int = 0) -> "SessionCalibration":
        """Ordinary least squares of observed on manual fraction, one row per patient.

        strata        e.g. "control" / "aml": phi is the LARGER of the per-stratum Pearson
                      dispersions, so a stratum that scatters more is never averaged away.
        manual_total  cells in the manual count, only to report the regression-dilution
                      ratio (the manual count is itself a 100-cell sample).
        """
        import numpy as np

        k = np.asarray(observed_count, float)
        n = np.asarray(n_classified, float)
        m = np.asarray(manual_fraction, float)
        strata = np.asarray(strata)
        q = k / n

        def ols(qq, mm):
            slope = np.cov(qq, mm, bias=True)[0, 1] / np.var(mm)
            return qq.mean() - slope * mm.mean(), slope

        a, b = ols(q, m)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            index = rng.integers(0, len(q), len(q))
            if np.var(m[index]) > 0:
                draws.append(ols(q[index], m[index]))
        draws = np.asarray(draws)
        cov = np.cov(draws.T)

        fitted = np.clip(a + b * m, 1e-4, 1 - 1e-4)
        pearson = (q - fitted) ** 2 / (fitted * (1 - fitted) / n)
        dof = len(q) / max(len(q) - 2, 1)
        per_stratum = {str(s): float(pearson[strata == s].mean() * dof) for s in np.unique(strata)}
        phi = max(1.0, *per_stratum.values())

        diagnostics = {
            "slope_ci95": [float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
            "intercept_ci95": [float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
            "dispersion_per_stratum": per_stratum,
            "observed_median_per_stratum": {str(s): float(np.median(q[strata == s]))
                                            for s in np.unique(strata)},
            "n_per_stratum": {str(s): int((strata == s).sum()) for s in np.unique(strata)},
            "bootstrap": len(draws),
        }
        if manual_total is not None:
            t = np.asarray(manual_total, float)
            diagnostics["reliability_ratio"] = float(1 - np.mean(m * (1 - m) / t) / np.var(m))
        return cls(float(a), float(b), float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1]),
                   float(phi), int(len(q)), diagnostics)

    # -- correction (stdlib only) ------------------------------------------
    @property
    def usable(self) -> bool:
        """Pre-declared: the lower 95 % bootstrap bound of the slope clears MIN_SEPARATION."""
        low = self.diagnostics.get("slope_ci95", [self.slope])[0]
        return self.slope >= MIN_SEPARATION and low >= MIN_SEPARATION

    def correct(self, observed_count: int, n: int) -> tuple[float, float, bool]:
        if n <= 0:
            return 0.0, 0.0, False
        observed = observed_count / n
        smoothed = (observed_count + 0.5) / (n + 1)
        if not self.usable:
            return observed, smoothed * (1 - smoothed) / n, False
        corrected = (observed - self.intercept) / self.slope
        # delta method: d/da = -1/b, d/db = -corrected/b ; counting scatter inflated by phi
        variance = (self.dispersion * smoothed * (1 - smoothed) / n
                    + self.var_intercept
                    + corrected ** 2 * self.var_slope
                    + 2 * corrected * self.covariance) / self.slope ** 2
        return min(max(corrected, 0.0), 1.0), max(variance, 1e-12), True

    def effective_counts(self, observed_count: int, n: int) -> tuple[int, int, bool]:
        """Same design-effect construction as Quantifier.effective_counts, capped at n."""
        proportion, variance, ok = self.correct(observed_count, n)
        if not ok:
            return observed_count, n, False
        smoothed = (observed_count + 0.5) / (n + 1)
        variance_raw = smoothed * (1 - smoothed) / n
        n_effective = int(min(n, max(1.0, round(n * variance_raw / variance))))
        return int(round(proportion * n_effective)), n_effective, True


@dataclass
class RecalibratedQuantifier:
    """The cell-level Quantifier, with some grid quantities recalibrated per session.

    Drop-in for `grid.evaluate(quantifier=...)`, which only calls `effective_counts`.
    """

    base: Quantifier
    sessions: dict[str, SessionCalibration] = field(default_factory=dict)

    def effective_counts(self, name: str, observed_count: int, n: int) -> tuple[int, int, bool]:
        if name in self.sessions:
            return self.sessions[name].effective_counts(observed_count, n)
        return self.base.effective_counts(name, observed_count, n)

    def correct(self, name: str, observed_count: int, n: int) -> tuple[float, float, bool]:
        if name in self.sessions:
            return self.sessions[name].correct(observed_count, n)
        return self.base.correct(name, observed_count, n)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"base": self.base.report(),
             "sessions": {k: asdict(v) for k, v in self.sessions.items()}}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RecalibratedQuantifier":
        from .quantify import ClassRates
        data = json.loads(Path(path).read_text())
        base = Quantifier(rates={k: ClassRates(v["tpr"], v["fpr"], v["n_positive"], v["n_negative"])
                                 for k, v in data["base"].items()})
        return cls(base, {k: SessionCalibration(**v) for k, v in data["sessions"].items()})
