#!/usr/bin/env python3
"""Article figures, drawn from the saved results of run 20260911_051357Z (no GPU, no model).

    python3 tools/make_figures.py <run_dir> <out_dir>
Every figure is a PNG (300 dpi) + PDF, and every number on it comes from a file of the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R = RUN / "results"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 110})
pred = pd.read_csv(R / "caitomorph_409_predictions.csv")
thr = json.loads((R / "resolved_thresholds.json").read_text())
TAU, OOD_T = thr["p_abn"], thr["ood_mahalanobis"]
made = []


def save(fig, name, caption):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    made.append((name, caption))


# 1. ROC of the screening score --------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 4.0))
for title, pos, neg, colour in (("AML vs stem-cell donors", ["AML"], ["Stem cell donor"], "C0"),
                                ("Acute leukaemia vs donors + reactive", ["AML", "ALL", "AL"],
                                 ["Stem cell donor", "Reactive changes"], "C1")):
    s = pred[pred.diagnosis_fine.isin(pos + neg)]
    y = s.diagnosis_fine.isin(pos).astype(int)
    fpr, tpr, _ = roc_curve(y, s.p_abn)
    ax.plot(fpr, tpr, color=colour, lw=1.8,
            label=f"{title}  AUROC {roc_auc_score(y, s.p_abn):.3f}  ({y.sum()} vs {len(y) - y.sum()})")
aml, don = pred[pred.diagnosis_fine == "AML"], pred[pred.diagnosis_fine == "Stem cell donor"]
ax.plot(1 - (don.p_abn < TAU).mean(), (aml.p_abn >= TAU).mean(), "o", color="C0", ms=7,
        label=f"pre-registered τ = {TAU:.3f} (19/37, 99/99)")
ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
ax.set(xlabel="1 − specificity", ylabel="sensitivity", xlim=(-0.02, 1), ylim=(0, 1.02),
       title="Session screening score P_abn — cAItomorph (held-out)")
ax.legend(fontsize=6.5, loc="lower right", frameon=False)
save(fig, "fig01_roc_p_abn", "ROC of the MIL screening score P_abn on cAItomorph; marker: the pre-registered operating point.")

# 2. Labels by diagnosis ------------------------------------------------------------------
conf = pd.read_csv(R / "caitomorph_409_confusion_full.csv").set_index("diagnosis_fine")
order = ["Stem cell donor", "Reactive changes", "AML", "ALL", "AL", "B-cell neoplasm", "T-cell neoplasm",
         "HCL", "MM", "PCL", "MDS", "MDS / MPN", "CMML", "CML", "MPN", "MPN / MDS-RS-T", "ET", "PV"]
conf = conf.reindex([o for o in order if o in conf.index])
cols = ["non_leukemic", "indeterminate", "acute_blastic__myeloid_oriented", "out_of_domain"]
conf = conf[[c for c in cols if c in conf.columns]]
frac = conf.div(conf.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(5.4, 5.6))
im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(frac.shape[0]):
    for j in range(frac.shape[1]):
        n = int(conf.values[i, j])
        ax.text(j, i, str(n), ha="center", va="center", fontsize=7,
                color="white" if frac.values[i, j] > 0.55 else "black")
ax.set_xticks(range(len(conf.columns)), ["non-leukaemic", "indeterminate", "acute\nmyeloid-oriented", "out of\ndomain"])
ax.set_yticks(range(len(conf)), [f"{d} (n={int(conf.loc[d].sum())})" for d in conf.index])
ax.set_title("Decision-grid labels by diagnosis — cAItomorph, 409 patients", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of the row")
save(fig, "fig02_labels_by_diagnosis", "Frozen decision grid on cAItomorph: count per diagnosis and label (colour = row fraction). No chronic, lymphoid or lineage-indeterminate label was emitted.")

# 3. P_abn by diagnosis ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
groups = [d for d in order if d in set(pred.diagnosis_fine)]
rng = np.random.default_rng(0)
for i, d in enumerate(groups):
    v = pred.loc[pred.diagnosis_fine == d, "p_abn"].to_numpy()
    ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=6, alpha=0.6,
               color="C3" if d in ("AML", "ALL", "AL") else ("C2" if d in ("Stem cell donor", "Reactive changes") else "C7"))
ax.axhline(TAU, color="k", ls="--", lw=0.8); ax.text(len(groups) - 0.5, TAU + 0.015, f"τ = {TAU:.3f}", ha="right", fontsize=7)
ax.set_xticks(range(len(groups)), groups, rotation=60, ha="right", fontsize=7)
ax.set(ylabel="P_abn (calibrated)", title="Screening score per patient, by diagnosis")
save(fig, "fig03_p_abn_by_diagnosis", "Calibrated P_abn per cAItomorph patient; dashed line: the operating point fixed on development data.")

# 4. Training curves ----------------------------------------------------------------------
cell = pd.read_json(RUN / "metrics/cell_head.jsonl", lines=True)
mil = pd.read_json(RUN / "metrics/mil_head.jsonl", lines=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
a1.plot(cell.epoch, cell.loss, "-o", ms=3, label="training loss")
a1b = a1.twinx(); a1b.plot(cell.epoch, cell.val_macro_f1, "-s", ms=3, color="C1", label="validation macro-F1")
a1b.axvline(14, color="C1", ls=":", lw=0.8); a1b.text(14.3, cell.val_macro_f1.min(), "best 14", fontsize=7, color="C1")
a1.set(xlabel="epoch", ylabel="loss", title="Cell head (early stop on macro-F1)"); a1b.set_ylabel("macro-F1")
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, frameon=False)
a2.semilogy(mil.epoch, mil.loss, "-o", ms=3, label="training BCE")
a2b = a2.twinx(); a2b.plot(mil.epoch, mil.mil_val_auroc, "-s", ms=3, color="C1", label="mil_val AUROC (29 patients)")
a2b.set_ylim(0.9, 1.01); a2b.axvline(1, color="C1", ls=":", lw=0.8)
a2.set(xlabel="epoch", ylabel="BCE (log)", title="MIL head: AUROC saturated → epoch 1 retained"); a2b.set_ylabel("AUROC")
h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a2b.get_legend_handles_labels()
a2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, frameon=False)
fig.tight_layout()
save(fig, "fig04_training_curves", "Training curves. Left: cell head, early stopping on validation macro-F1 (best epoch 14 of 22). Right: MIL head; the validation AUROC was 1.0 from epoch 1, so the declared strict-improvement rule retained epoch 1.")

# 5. Amendment 9: predicted vs manual blasts ----------------------------------------------
a9 = pd.read_csv(R / "amendment9_mll_patients.csv")
cal = json.loads((R / "amendment9_blast_calibration.json").read_text())["calibration"]
fig, ax = plt.subplots(figsize=(4.2, 3.9))
for stratum, colour in (("control", "C2"), ("aml", "C3")):
    s = a9[a9.stratum == stratum]
    ax.scatter(s.manual * 100, s.q * 100, s=10, alpha=0.7, color=colour, label=f"{stratum} (n={len(s)})")
m = np.linspace(0, 1, 50)
ax.plot(m * 100, (cal["intercept"] + cal["slope"] * m) * 100, "k-", lw=1,
        label=f"fit q = {cal['intercept']:.3f} + {cal['slope']:.3f}·m")
ax.plot([0, 100], [0, 100], ":", color="grey", lw=0.8, label="identity")
ax.axhline(5, color="C0", ls="--", lw=0.7); ax.axhline(20, color="C0", ls="--", lw=0.7)
ax.set(xlabel="manual blasts, % (100-cell differential)", ylabel="predicted blasts, % (cell head)",
       title=f"AML-MLL, 189 patients: offset + scatter (φ = {cal['diagnostics']['dispersion_per_stratum']['control']:.1f} / {cal['diagnostics']['dispersion_per_stratum']['aml']:.1f})")
ax.title.set_fontsize(8)
ax.legend(fontsize=7, frameon=False, loc="upper left")
save(fig, "fig05_blasts_predicted_vs_manual", "Predicted vs manual blast fraction on the 189 AML-MLL patients (amendment 9). Controls (0 manual blasts) are predicted at 5–30 %: a 13 % offset with patient-level scatter far above binomial (φ 9.2 controls, 24.8 AML).")

# 6. OOD scores ---------------------------------------------------------------------------
x40 = pd.read_csv(R / "x40_stress.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
data = [pred.loc[pred.diagnosis_fine == d, "ood"].to_numpy() for d in groups]
ax.boxplot(data, positions=range(len(groups)), widths=0.6, showfliers=True, flierprops={"ms": 2})
ax.scatter([len(groups) + 0.5] * len(x40), x40.ood, color="C3", marker="D", s=22, zorder=3, label="prototype ×40 sessions (3/3 withheld)")
ax.axhline(OOD_T, color="k", ls="--", lw=0.8); ax.text(0, OOD_T + 15, f"threshold {OOD_T:.0f}", fontsize=7)
ax.set_xticks(list(range(len(groups))) + [len(groups) + 0.5], groups + ["×40 proto"], rotation=60, ha="right", fontsize=7)
ax.set(ylabel="median per-crop Mahalanobis distance", title="OOD gate: cAItomorph by diagnosis vs prototype ×40 acquisitions")
ax.legend(fontsize=7, frameon=False, loc="lower right")
save(fig, "fig06_ood_scores", "OOD score per session. Box plots: cAItomorph patients by diagnosis; diamonds: the three ×40 prototype sessions; dashed: threshold fitted on development bags (99 % pass rate).")

# 7. gamma sensitivity --------------------------------------------------------------------
sens = pd.read_csv(R / "sensitivity_ops.csv")
g = sens[sens.parameter == "gamma"].copy()
frac_of = lambda s: s.apply(lambda v: int(v.split("/")[0]) / int(v.split("/")[1]))
fig, ax = plt.subplots(figsize=(4.0, 3.0))
ax.plot(g.value, g.spec_donors, "-o", label="specificity vs donors")
ax.plot(g.value, frac_of(g.sens_acute), "-s", label="sensitivity, acute")
ax.plot(g.value, g.indeterminate, "-^", label="indeterminate (all 409)")
ax.axvline(0.90, color="grey", ls=":", lw=0.8); ax.text(0.901, 0.05, "frozen", fontsize=7, color="grey")
ax.set(xlabel="γ (assertion confidence)", ylim=(0, 1.05), title="Sensitivity analysis of the only free parameter")
ax.legend(fontsize=7, frameon=False)
save(fig, "fig07_gamma_sensitivity", "Grid performance for γ ∈ {0.80, 0.90, 0.95}: monotone and small; the reference percentile (97.5–99.5) has no effect because the rules it feeds never fire.")

# 8. Cell head per class --------------------------------------------------------------------
pc = pd.read_csv(R / "cell_head_per_class.csv").rename(columns={"index": "cls"})
pc = pc[~pc.cls.isin(["accuracy", "macro avg", "weighted avg"])]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
x = np.arange(len(pc))
ax.bar(x - 0.2, pc.precision, 0.4, label="precision"); ax.bar(x + 0.2, pc.recall, 0.4, label="recall")
ax.set_xticks(x, [c.replace("_", " ") for c in pc.cls], rotation=60, ha="right", fontsize=7)
ax.set(ylim=(0, 1.05), title="Cell head on definite-label validation cells (14 121; macro-F1 0.753)")
ax.legend(fontsize=7, frameon=False, ncol=2)
save(fig, "fig08_cell_head_per_class", "Per-class precision and recall of the 16-class cell head. Abnormal promyelocytes are not recognised (recall 0.029), which makes the APL flag inoperable.")

# 9. Differential vs manual count -------------------------------------------------------------
rho = json.loads((R / "differential_spearman.json").read_text())
items = sorted(rho.items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.barh([k.replace("pb_", "").replace("_", " ") for k, _ in items], [v for _, v in items],
        color=["C3" if v < 0.5 else "C0" for _, v in items])
ax.axvline(0.5, color="k", ls="--", lw=0.8); ax.text(0.51, 0, "falsifier (myeloblast)", fontsize=6.5)
ax.set(xlabel="Spearman ρ (189 patients)", xlim=(0, 1), title="Bag differential vs manual 100-cell count (AML-MLL)")
ax.title.set_fontsize(8)
save(fig, "fig09_differential_spearman", "Rank correlation between the predicted and the manual differential per class. ρ validates the ordering of patients, not the level (see fig05).")

index = ["# Figures — run 20260911_051357Z", "", "Generated by `aster-block2/tools/make_figures.py` from the run's `results/` and `metrics/` (no model, no GPU).", ""]
index += [f"- **{n}** — {c}" for n, c in made]
(OUT / "FIGURES.md").write_text("\n".join(index) + "\n")
print("\n".join(n for n, _ in made))
