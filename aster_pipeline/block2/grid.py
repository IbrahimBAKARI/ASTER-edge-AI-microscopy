"""Executor for the FROZEN decision grid.

The rule logic is written out explicitly rather than interpreted generically: a
reviewer must be able to read this file next to PREREGISTRATION.md section 4 and check
that they say the same thing. Only the NUMBERS come from decision_grid.yaml, and the
[REF]/[FIT] ones are filled in by Phase 1 as dated amendments.

Nothing here may be changed without amending the pre-registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .proportion_test import Verdict, posterior_exceeds, test_proportion
from .quantify import Quantifier

GRID_PATH = Path(__file__).with_name("decision_grid.yaml")

CELL_CLASSES = [
    "myeloblast", "lymphoblast", "promyelocyte", "promyelocyte_abnormal",
    "myelocyte", "metamyelocyte", "band_neutrophil", "segmented_neutrophil",
    "basophil", "eosinophil", "monocyte", "lymphocyte", "lymphocyte_atypical",
    "smudge_cell", "erythroblast",
]
QUANTITIES = {
    "blast_frac": ["myeloblast", "lymphoblast", "promyelocyte_abnormal"],
    "ig_frac": ["promyelocyte", "myelocyte", "metamyelocyte"],
    "baso_frac": ["basophil"],
    "lymph_frac": ["lymphocyte"],
    "smudge_frac": ["smudge_cell"],
    "atypical_frac": ["lymphocyte_atypical"],
    "mono_frac": ["monocyte"],
    "abn_promy_frac": ["promyelocyte_abnormal"],
    "myeloblast_frac": ["myeloblast"],
    "myelocyte_frac": ["myelocyte"],
    "metamyelocyte_frac": ["metamyelocyte"],
}


@dataclass
class Thresholds:
    """Every number the grid needs. Values are None until Phase 1 resolves them."""

    gamma: float = 0.90
    reference_percentile: float = 99.0
    tier_screening: int = 100
    tier_pattern: int = 200
    tier_reference: int = 400
    blast_acute: float = 0.20           # [WHO]
    mono_cmml: float = 0.10             # [WHO]
    blast_normal: float = 0.05          # [CONV]
    ig_normal: float = 0.02             # [CONV]
    lymph_normal: float = 0.50          # [CONV]
    ig_cml_floor: float = 0.10          # [REF] floor
    baso_cml_floor: float = 0.02        # [REF] floor
    lymph_cll_floor: float = 0.50       # [REF] floor
    smudge_cll_floor: float = 0.02      # [REF] floor
    ig_cml: float | None = None         # [REF] resolved = max(floor, p99 reference)
    baso_cml: float | None = None
    lymph_cll: float | None = None
    smudge_cll: float | None = None
    atypical_reactive: float | None = None
    apl_abn_promy: float | None = None  # [FIT] 24 PML_RARA vs 105 other AML
    lineage_lo: float | None = None     # [FIT]
    lineage_hi: float | None = None     # [FIT]
    p_abn: float | None = None          # [FIT] specificity >= 0.95
    ood_mahalanobis: float | None = None  # [FIT] 99% in-domain pass rate
    min_classified_frac: float | None = None  # [FIT] 1st percentile of dev
    mil_temperature: float | None = None  # [FIT] temperature scaling on dev_cal

    def unresolved(self) -> list[str]:
        return [name for name, value in vars(self).items() if value is None]


@dataclass
class GridResult:
    label: str
    tier: str
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    quantities: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    n_localized: int = 0
    n_classified: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "tier": self.tier, "flags": self.flags,
            "reasons": self.reasons, "quantities": self.quantities,
            "verdicts": self.verdicts,
            "n_localized": self.n_localized, "n_classified": self.n_classified,
        }


def counts_to_quantities(counts: dict[str, int]) -> tuple[dict[str, int], int]:
    n_classified = sum(counts.get(name, 0) for name in CELL_CLASSES)
    numerators = {
        name: sum(counts.get(cls, 0) for cls in classes) for name, classes in QUANTITIES.items()
    }
    return numerators, n_classified


def evaluate(
    counts: dict[str, int],
    thresholds: Thresholds,
    *,
    n_localized: int,
    p_abn: float | None = None,
    lineage_post: float | None = None,
    ood_score: float | None = None,
    quantifier: Quantifier | None = None,
) -> GridResult:
    """Run the frozen grid on one session.

    counts       cell-class counts over the bag (``other_artifact`` may be present and
                 is excluded from N_c)
    n_localized  N, the number of WBC block 1 detected
    """
    numerators, n_classified = counts_to_quantities(counts)
    gamma = thresholds.gamma
    quantities = {
        name: (numerators[name] / n_classified if n_classified else 0.0) for name in numerators
    }
    result = GridResult(label="indeterminate", tier="none", quantities=quantities,
                        n_localized=n_localized, n_classified=n_classified)

    def check(name: str, theta: float | None, label: str | None = None) -> Verdict:
        """Three-way verdict on `name >= theta`, on CORRECTED counts.

        Counting a classifier's argmax predictions is a biased prevalence estimator: on a
        smear with no smudge cells, a 95 %-specific classifier still reports ~5 % of them,
        and a test on raw counts asserts it. The quantifier inverts the measured error
        rates and hands back an effective (k, n) carrying the widened uncertainty, so the
        frozen test consumes it unchanged. A quantity whose TPR and FPR cannot be
        separated is not assertable at all and returns INDETERMINATE.
        """
        if theta is None:
            raise ValueError(f"threshold for {name} is unresolved; run Phase 1 first")
        k, n_effective = numerators[name], n_classified
        if quantifier is not None:
            k, n_effective, quantifiable = quantifier.effective_counts(
                name, numerators[name], n_classified)
            if not quantifiable:
                result.verdicts[label or name] = Verdict.INDETERMINATE.value
                result.reasons.append(
                    f"{name}: the cell classifier cannot separate this population "
                    f"(TPR - FPR too small); no proportion claim is possible")
                return Verdict.INDETERMINATE
        verdict = test_proportion(k, n_effective, theta, gamma)
        result.verdicts[label or name] = verdict.value
        return verdict

    # ---- R0 gates ----------------------------------------------------------
    if ood_score is not None and thresholds.ood_mahalanobis is not None:
        if ood_score > thresholds.ood_mahalanobis:
            result.label = "out_of_domain"
            result.reasons.append(
                f"OOD score {ood_score:.2f} above the in-domain threshold "
                f"{thresholds.ood_mahalanobis:.2f}")
            return result
    if n_classified < thresholds.tier_screening:
        result.label = "insufficient_evidence"
        result.reasons.append(
            f"{n_classified} classified leukocytes, below the routine manual "
            f"differential of {thresholds.tier_screening} [STD]")
        return result
    if thresholds.min_classified_frac is not None and n_localized:
        if n_classified / n_localized < thresholds.min_classified_frac:
            result.label = "indeterminate"
            result.reasons.append(
                f"only {n_classified}/{n_localized} localized objects were classified as "
                f"leukocytes; the localizer output is not interpretable")
            return result

    if n_classified >= thresholds.tier_reference:
        result.tier = "reference"
    elif n_classified >= thresholds.tier_pattern:
        result.tier = "pattern"
    else:
        result.tier = "screening"

    # ---- observation flags -------------------------------------------------
    if check("mono_frac", thresholds.mono_cmml, "flag_mono") is Verdict.ASSERT:
        result.flags.append("monocytic_predominance")
    if thresholds.atypical_reactive is not None:
        if check("atypical_frac", thresholds.atypical_reactive, "flag_atypical") is Verdict.ASSERT:
            result.flags.append("reactive_lymphoid_observation")
    if counts.get("erythroblast", 0) >= 1:
        result.flags.append("nucleated_rbc_present")

    # ---- tier S: the MIL statement only ------------------------------------
    if result.tier == "screening":
        if p_abn is None or thresholds.p_abn is None:
            result.label = "indeterminate"
            result.reasons.append("no calibrated MIL probability available")
        else:
            below = p_abn < thresholds.p_abn
            result.label = "non_leukemic" if below else "suspicious_for_neoplasm"
            result.reasons.append(
                f"{n_classified} classified leukocytes: routine-differential tier, "
                f"screening statement only [STD]; P_abn={p_abn:.3f}")
        return result

    # ---- R1 acute blastic --------------------------------------------------
    acute = check("blast_frac", thresholds.blast_acute)
    if acute is Verdict.ASSERT:
        result.reasons.append(
            f"blasts {quantities['blast_frac']:.1%} of {n_classified} classified "
            f"leukocytes, at or above the 20% criterion [WHO]")
        # R1a APL, evaluated first
        apl = check("abn_promy_frac", thresholds.apl_abn_promy, "apl")
        if apl is Verdict.ASSERT and quantities["abn_promy_frac"] > quantities["myeloblast_frac"]:
            result.label = "acute_blastic__myeloid_oriented"
            result.flags.append("APL_suspicion")
            result.reasons.append(
                f"abnormal promyelocytes {quantities['abn_promy_frac']:.1%} predominate "
                f"over myeloblasts - urgent, confirm PML::RARA")
            return result
        # R1b lineage
        if lineage_post is None or thresholds.lineage_hi is None:
            result.label = "acute_blastic__lineage_indeterminate"
            result.reasons.append("no blast lineage posterior available")
        elif lineage_post >= thresholds.lineage_hi:
            result.label = "acute_blastic__lymphoid_oriented"
            result.reasons.append(f"blast lineage posterior {lineage_post:.2f} (lymphoid)")
        elif lineage_post <= thresholds.lineage_lo:
            result.label = "acute_blastic__myeloid_oriented"
            result.reasons.append(f"blast lineage posterior {lineage_post:.2f} (myeloid)")
        else:
            result.label = "acute_blastic__lineage_indeterminate"
            result.reasons.append(
                f"blast lineage posterior {lineage_post:.2f} inside the indeterminate "
                f"band [{thresholds.lineage_lo:.2f}, {thresholds.lineage_hi:.2f}] - "
                f"flow cytometry required")
        return result

    if acute is not Verdict.REJECT:
        result.label = "indeterminate"
        result.reasons.append(
            f"blast fraction {quantities['blast_frac']:.1%} cannot be placed with respect "
            f"to the 20% criterion at gamma={gamma} on {n_classified} cells")
        return result

    # ---- R2 chronic myeloid ------------------------------------------------
    ig = check("ig_frac", thresholds.ig_cml, "cml_ig")
    baso = check("baso_frac", thresholds.baso_cml, "cml_baso")
    cmml_exclusion = (
        check("mono_frac", thresholds.mono_cmml, "cmml_mono") is Verdict.ASSERT
        and check("ig_frac", thresholds.ig_cml_floor, "cmml_ig") is Verdict.REJECT
    )
    if ig is Verdict.ASSERT and baso is Verdict.ASSERT and not cmml_exclusion:
        result.label = "chronic_myeloid_pattern"
        result.reasons.append(
            f"left shift (immature granulocytes {quantities['ig_frac']:.1%}) with "
            f"basophilia ({quantities['baso_frac']:.1%}); blast excess rejected")
        return result

    # ---- R3 chronic lymphoid -----------------------------------------------
    lymph = check("lymph_frac", thresholds.lymph_cll, "cll_lymph")
    smudge = check("smudge_frac", thresholds.smudge_cll, "cll_smudge")
    reactive_exclusion = (
        thresholds.atypical_reactive is not None
        and check("atypical_frac", thresholds.atypical_reactive, "reactive_atypical")
        is Verdict.ASSERT
        and smudge is Verdict.REJECT
    )
    if lymph is Verdict.ASSERT and smudge is Verdict.ASSERT and not reactive_exclusion:
        result.label = "chronic_lymphoid_pattern"
        result.reasons.append(
            f"mature lymphocytosis ({quantities['lymph_frac']:.1%}) with smudge cells "
            f"({quantities['smudge_frac']:.1%}); blast excess rejected")
        return result

    # ---- R4 non-leukemic ---------------------------------------------------
    normal_checks = {
        "blast_frac": thresholds.blast_normal,
        "ig_frac": thresholds.ig_normal,
    }
    all_below = all(
        check(name, theta, f"normal_{name}") is Verdict.REJECT
        for name, theta in normal_checks.items()
    )
    lymph_not_high = lymph is Verdict.REJECT
    mil_ok = p_abn is not None and thresholds.p_abn is not None and p_abn < thresholds.p_abn
    if all_below and lymph_not_high and mil_ok:
        result.label = "non_leukemic"
        result.reasons.append(
            f"reference leukocyte composition and P_abn={p_abn:.3f} below the "
            f"specificity-0.95 operating point")
        return result

    # ---- R5 fallback -------------------------------------------------------
    result.label = "indeterminate"
    result.reasons.append("no rule reached the required confidence at gamma=%.2f" % gamma)
    if p_abn is not None:
        result.reasons.append(f"P_abn={p_abn:.3f} recorded, not used outside R1/R4")
    return result


def load_thresholds(path: Path | None = None) -> Thresholds:
    """Read the frozen numbers. [REF]/[FIT] placeholders stay None until Phase 1."""
    import yaml

    data = yaml.safe_load((path or GRID_PATH).read_text(encoding="utf-8"))
    thresholds = Thresholds()
    thresholds.gamma = float(data["inference"]["gamma"]["value"])
    thresholds.reference_percentile = float(data["reference_interval"]["percentile"]["value"])
    tiers = data["tiers"]
    thresholds.tier_screening = int(tiers["screening"]["min_classified_cells"]["value"])
    thresholds.tier_pattern = int(tiers["pattern"]["min_classified_cells"]["value"])
    thresholds.tier_reference = int(tiers["reference"]["min_classified_cells"]["value"])
    return thresholds
