"""Does the grid do what PREREGISTRATION.md section 4 says?

Synthetic sessions with known compositions, run through the real `evaluate()`. These are
behaviour tests, not unit tests: each case is a clinical situation and the expected verdict
is the one the pre-registration commits to. If a case fails, either the code or the
document is wrong - never adjust the expectation to match the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aster_block2.grid import Thresholds, evaluate  # noqa: E402
from aster_block2.quantify import ClassRates, Quantifier  # noqa: E402


def resolved() -> Thresholds:
    """Plausible Phase-1 values, so the grid is executable in a test."""
    t = Thresholds()
    t.ig_cml, t.baso_cml = 0.10, 0.02
    t.lymph_cll, t.smudge_cll, t.atypical_reactive = 0.50, 0.02, 0.10
    t.apl_abn_promy = 0.10
    t.lineage_lo, t.lineage_hi = 0.35, 0.65
    t.p_abn = 0.50
    t.ood_mahalanobis, t.min_classified_frac, t.mil_temperature = 100.0, 0.60, 1.0
    return t


def good_quantifier() -> Quantifier:
    """A competent classifier: 92 % TPR, 3 % FPR on every quantity."""
    from aster_block2.grid import QUANTITIES
    return Quantifier(rates={name: ClassRates(0.92, 0.03, 2000, 20000) for name in QUANTITIES})


def session(**fractions) -> dict:
    """Cell counts for a 500-cell session; the remainder is segmented neutrophils."""
    total = 500
    counts = {k: int(round(v * total)) for k, v in fractions.items()}
    counts["segmented_neutrophil"] = counts.get("segmented_neutrophil", 0) + \
        total - sum(counts.values())
    return counts


CASES = [
    # name, counts, p_abn, lineage, expected label, expected flag
    ("normal", session(lymphocyte=0.30, monocyte=0.06, eosinophil=0.03, basophil=0.005),
     0.10, None, "non_leukemic", None),
    ("AML", session(myeloblast=0.45, lymphocyte=0.15, monocyte=0.05),
     0.95, 0.05, "acute_blastic__myeloid_oriented", None),
    ("ALL", session(lymphoblast=0.50, lymphocyte=0.15),
     0.95, 0.95, "acute_blastic__lymphoid_oriented", None),
    ("blast lineage ambiguous", session(myeloblast=0.25, lymphoblast=0.25, lymphocyte=0.15),
     0.95, 0.50, "acute_blastic__lineage_indeterminate", None),
    ("APL", session(promyelocyte_abnormal=0.55, myeloblast=0.05, lymphocyte=0.10),
     0.95, 0.05, "acute_blastic__myeloid_oriented", "APL_suspicion"),
    ("CML", session(myelocyte=0.12, metamyelocyte=0.08, promyelocyte=0.04,
                    basophil=0.10, band_neutrophil=0.10, lymphocyte=0.10),
     0.40, None, "chronic_myeloid_pattern", None),
    ("CLL", session(lymphocyte=0.70, smudge_cell=0.12, monocyte=0.03),
     0.40, None, "chronic_lymphoid_pattern", None),
]


@pytest.mark.parametrize("name,counts,p_abn,lineage,expected,flag", CASES)
def test_case(name, counts, p_abn, lineage, expected, flag):
    result = evaluate(counts, resolved(), n_localized=520, p_abn=p_abn,
                      lineage_post=lineage, ood_score=10.0, quantifier=good_quantifier())
    assert result.label == expected, f"{name}: got {result.label} - {result.reasons}"
    if flag:
        assert flag in result.flags, f"{name}: missing flag {flag}"


def test_reactive_lymphocytosis_is_not_cll():
    """42 reactive patients are the hard negatives of the primary test (section 6.3)."""
    counts = session(lymphocyte=0.55, lymphocyte_atypical=0.15, monocyte=0.05)
    result = evaluate(counts, resolved(), n_localized=520, p_abn=0.40,
                      ood_score=10.0, quantifier=good_quantifier())
    assert result.label != "chronic_lymphoid_pattern", result.reasons
    assert "reactive_lymphoid_observation" in result.flags


def test_monocytosis_without_left_shift_is_not_cml():
    counts = session(monocyte=0.25, lymphocyte=0.20)
    result = evaluate(counts, resolved(), n_localized=520, p_abn=0.40,
                      ood_score=10.0, quantifier=good_quantifier())
    assert result.label != "chronic_myeloid_pattern", result.reasons
    assert "monocytic_predominance" in result.flags


def test_gates():
    thresholds = resolved()
    few = {"segmented_neutrophil": 60, "lymphocyte": 20}
    assert evaluate(few, thresholds, n_localized=80, p_abn=0.1,
                    ood_score=10.0).label == "insufficient_evidence"

    screening = {"segmented_neutrophil": 90, "lymphocyte": 50}
    result = evaluate(screening, thresholds, n_localized=145, p_abn=0.1, ood_score=10.0,
                      quantifier=good_quantifier())
    assert result.tier == "screening" and result.label == "non_leukemic"

    out = evaluate(session(lymphocyte=0.3), thresholds, n_localized=520, p_abn=0.9,
                   ood_score=10_000.0)
    assert out.label == "out_of_domain"


def test_tier_reference_is_reached_at_400():
    result = evaluate(session(lymphocyte=0.30), resolved(), n_localized=520, p_abn=0.1,
                      ood_score=10.0, quantifier=good_quantifier())
    assert result.tier == "reference"


def test_blast_at_22_percent_on_200_cells_is_indeterminate():
    """Section 3.3: at 200 cells, asserting 20 % needs 24 % observed. 22 % cannot be placed."""
    counts = {"myeloblast": 44, "lymphocyte": 60, "segmented_neutrophil": 96}
    result = evaluate(counts, resolved(), n_localized=210, p_abn=0.8, lineage_post=0.1,
                      ood_score=10.0)
    assert result.label == "indeterminate", result.reasons


def test_uncorrected_counts_would_fire_on_a_normal_smear():
    """Amendment 6a, the reason the quantifier exists.

    A normal smear whose smudge cells are entirely classifier false positives: with a
    quantifier the criterion is rejected, without one it is asserted.
    """
    poor = Quantifier(rates={name: ClassRates(0.90, 0.06, 2000, 20000)
                             for name in __import__("aster_block2.grid", fromlist=["QUANTITIES"]).QUANTITIES})
    counts = session(lymphocyte=0.55, smudge_cell=0.06, monocyte=0.05)
    with_correction = evaluate(counts, resolved(), n_localized=520, p_abn=0.40,
                               ood_score=10.0, quantifier=poor)
    without = evaluate(counts, resolved(), n_localized=520, p_abn=0.40, ood_score=10.0)
    assert without.verdicts.get("cll_smudge") == "assert"
    assert with_correction.verdicts.get("cll_smudge") != "assert"


def test_a_threshold_near_the_classifier_fpr_is_not_assertable():
    """A finding, locked in rather than worked around.

    `baso_frac >= 0.02` sits barely above a 3 % false-positive rate. A session showing 5 %
    basophils - which would be real basophilia - cannot assert the criterion, because the
    correction attributes most of the observation to classifier error. The grid returns
    indeterminate, which is the honest answer, and the paper must say that this criterion
    is only usable when the measured FPR is well below the threshold.
    """
    counts = session(myelocyte=0.12, metamyelocyte=0.08, promyelocyte=0.04,
                     basophil=0.05, band_neutrophil=0.10, lymphocyte=0.10)
    result = evaluate(counts, resolved(), n_localized=520, p_abn=0.40,
                      ood_score=10.0, quantifier=good_quantifier())
    assert result.label == "indeterminate"
    assert result.verdicts["cml_baso"] != "assert"

    # the same session read by a classifier with a 0.5 % FPR does fire
    from aster_block2.grid import QUANTITIES
    sharp = Quantifier(rates={n: ClassRates(0.92, 0.005, 2000, 20000) for n in QUANTITIES})
    better = evaluate(counts, resolved(), n_localized=520, p_abn=0.40,
                      ood_score=10.0, quantifier=sharp)
    assert better.label == "chronic_myeloid_pattern", better.reasons
