"""Prevalence estimation - counting is not classifying.

The whole decision grid consumes PROPORTIONS. Estimating a proportion by counting a
classifier's argmax predictions ("Classify and Count") is the known-worst prevalence
estimator: its bias depends on the classifier's error rates and on the true prevalence,
and for rare classes it is severe.

The failure this prevents, concretely. Suppose the smudge-cell classifier has 95 %
specificity - a good figure. On a NORMAL smear with no smudge cells at all, 5 % of the
other leukocytes are counted as smudge. `smudge_frac >= 0.02` is crossed, and a
proportion test that treats those counts as clean multinomial draws asserts it with high
confidence. R3 would fire on normal blood because of classifier error, not biology.

Correction: per-class Adjusted Classify and Count. With TPR and FPR measured on a
validation split,

    observed = TPR * true + FPR * (1 - true)
    true_hat = (observed - FPR) / (TPR - FPR)

and the variance is propagated by the delta method over the three estimated quantities.
The corrected proportion and its variance are then turned into an EFFECTIVE count and
sample size, so the frozen three-way test of `proportion_test.py` can consume them
unchanged: the classifier's uncertainty widens the interval instead of being ignored.

References for the method: Forman's Adjusted Classify and Count; Saerens-Latinne-
Decaestecker for the EM variant. Only ACC is implemented here - it is the one whose
uncertainty propagates in closed form, which is what the grid needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import json
import math

MIN_SEPARATION = 0.05   # TPR - FPR below this: the class is not quantifiable


@dataclass
class ClassRates:
    """Per-class TPR/FPR measured on a validation split, with their own uncertainty."""

    tpr: float
    fpr: float
    n_positive: int
    n_negative: int

    @property
    def separation(self) -> float:
        return self.tpr - self.fpr

    @property
    def var_tpr(self) -> float:
        return self.tpr * (1 - self.tpr) / max(self.n_positive, 1)

    @property
    def var_fpr(self) -> float:
        return self.fpr * (1 - self.fpr) / max(self.n_negative, 1)


@dataclass
class Quantifier:
    """ACC correction for every class, fitted once on the validation split."""

    rates: dict[str, ClassRates] = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, true_labels: Iterable[str], predicted_labels: Iterable[str],
            classes: Iterable[str]) -> "Quantifier":
        true_labels, predicted_labels = list(true_labels), list(predicted_labels)
        rates = {}
        for name in classes:
            positives = [p for t, p in zip(true_labels, predicted_labels) if t == name]
            negatives = [p for t, p in zip(true_labels, predicted_labels) if t != name]
            tpr = sum(p == name for p in positives) / len(positives) if positives else 0.0
            fpr = sum(p == name for p in negatives) / len(negatives) if negatives else 0.0
            rates[name] = ClassRates(tpr, fpr, len(positives), len(negatives))
        return cls(rates=rates)

    @classmethod
    def fit_groups(cls, true_labels: Iterable[str], predicted_labels: Iterable[str],
                   groups: dict[str, Iterable[str]]) -> "Quantifier":
        """Fit one ACC correction per GRID QUANTITY.

        The grid's criteria are sums of classes (`blast_frac` = myeloblast + lymphoblast +
        abnormal promyelocyte), so the correction must be fitted on the same sums: the
        error rate of "is this cell a blast" is not the sum of the three per-class rates.
        """
        true_labels, predicted_labels = list(true_labels), list(predicted_labels)
        rates = {}
        for name, members in groups.items():
            members = set(members)
            positives = [p in members for t, p in zip(true_labels, predicted_labels) if t in members]
            negatives = [p in members for t, p in zip(true_labels, predicted_labels) if t not in members]
            tpr = sum(positives) / len(positives) if positives else 0.0
            fpr = sum(negatives) / len(negatives) if negatives else 0.0
            rates[name] = ClassRates(tpr, fpr, len(positives), len(negatives))
        return cls(rates=rates)

    # -- correction ---------------------------------------------------------
    def correct(self, name: str, observed_count: int, n: int) -> tuple[float, float, bool]:
        """Return (corrected proportion, its variance, quantifiable?).

        `quantifiable` is False when TPR and FPR are too close to separate: the class
        cannot support a proportion claim at all, and the caller must abstain rather than
        report a corrected number that means nothing.
        """
        if n <= 0:
            return 0.0, 0.0, False
        observed = observed_count / n
        rate = self.rates.get(name)
        if rate is None or rate.separation < MIN_SEPARATION:
            return observed, observed * (1 - observed) / n, False

        corrected = (observed - rate.fpr) / rate.separation
        # delta method over observed, tpr, fpr
        d_observed = 1.0 / rate.separation
        d_fpr = (observed - rate.tpr) / rate.separation ** 2
        d_tpr = -(observed - rate.fpr) / rate.separation ** 2
        variance = (
            d_observed ** 2 * observed * (1 - observed) / n
            + d_tpr ** 2 * rate.var_tpr
            + d_fpr ** 2 * rate.var_fpr
        )
        return min(max(corrected, 0.0), 1.0), max(variance, 1e-12), True

    def effective_counts(self, name: str, observed_count: int, n: int) -> tuple[int, int, bool]:
        """Corrected proportion expressed as (k_eff, n_eff) for the frozen proportion test.

        The effective size uses the DESIGN EFFECT, n_eff = n * Var_raw / Var_corrected,
        rather than p(1-p)/Var. Both express "how much evidence is left after correcting",
        but the design effect stays well behaved when the corrected proportion sits at 0 or
        1 - exactly the case of a normal smear, where p(1-p)/Var collapses n_eff to 1 and
        turns a legitimate REJECT into an indeterminate. n_eff is capped at n: correcting
        can only ever weaken the evidence, never strengthen it beyond the cells counted.
        """
        proportion, variance, ok = self.correct(name, observed_count, n)
        if not ok:
            return observed_count, n, False
        # Var_raw uses the JEFFREYS-SMOOTHED proportion, (k + 1/2)/(n + 1), not k/n.
        # With k = 0 the raw variance is exactly zero, the design effect collapses n_eff to
        # 1, and a decisive REJECT becomes indeterminate - which would have put every
        # blast-free session (normal, CML, CLL) into indeterminate. The smoothing matches
        # the Jeffreys posterior the frozen test already uses.
        smoothed = (observed_count + 0.5) / (n + 1)
        variance_raw = smoothed * (1 - smoothed) / n
        n_effective = int(min(n, max(1.0, round(n * variance_raw / max(variance, 1e-12)))))
        return int(round(proportion * n_effective)), n_effective, True

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        return {
            name: {"tpr": round(r.tpr, 4), "fpr": round(r.fpr, 4),
                   "separation": round(r.separation, 4),
                   "n_positive": r.n_positive, "n_negative": r.n_negative,
                   "quantifiable": r.separation >= MIN_SEPARATION}
            for name, r in sorted(self.rates.items())
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.report(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Quantifier":
        data = json.loads(Path(path).read_text())
        return cls(rates={k: ClassRates(v["tpr"], v["fpr"], v["n_positive"], v["n_negative"])
                          for k, v in data.items()})


def quantification_error(true_counts: dict[str, int], predicted_counts: dict[str, int]) -> float:
    """Mean absolute error on class proportions - what the grid actually consumes.

    Reported per epoch as a diagnostic alongside the classification metrics: a model can
    have excellent macro recall and still estimate proportions badly, and it is the
    proportions the decision rests on.
    """
    total_true = sum(true_counts.values()) or 1
    total_predicted = sum(predicted_counts.values()) or 1
    names = set(true_counts) | set(predicted_counts)
    return sum(abs(true_counts.get(n, 0) / total_true - predicted_counts.get(n, 0) / total_predicted)
               for n in names) / max(len(names), 1)
