# Pre-registration audit trail

`results/PREREGISTRATION.sha256` always holds the **current** digests, so
`shasum -a 256 -c` passes at any time. This file holds the history: every superseded
digest and the justification for each change. Nothing is ever silently edited.

---

## Freeze — 2026-09-10

γ = 0.90, reference percentile = 99th. Frozen before the first cAItomorph inference.

```
7d7032bfd4e9e132d11d6238677741ecb9dab3094964de852cd3b02040a64ea4  PREREGISTRATION.md
43c56bcdb69316bc551327d0d9a7e6fcfa4c65eed3babe706ebfe38710bdf182  src/aster_block2/decision_grid.yaml
4fbda0bd8e9ecc5bd511d059b1a5976dcb37b1c25b05bc3cde35aebd213c7aea  src/aster_block2/proportion_test.py
```

---

## Amendment 1 — 2026-09-10, still before any cAItomorph inference

Three corrections made while wiring the corpus. Full text in `PREREGISTRATION.md`,
section "Amendment 1".

| # | Change | Moves a threshold? |
|---|---|---|
| 1a | PBC filenames carry per-cell labels (`PMY`/`MY`/`MMY`/`BNE`/`SNE`), not only the merged `ig` class. Left-shift training classes go from 15–109 cells to 1 030–1 742. | no — it strengthens a class §2 had declared weak |
| 1b | PBC cannot enter the `[REF]` reference intervals: they are per-patient proportions and PBC has no patient identifiers. Population corrected to the AML-MLL control patients of `dev_fit ∪ dev_cal` (≈ 24). | no — it corrects an unimplementable definition, and both terms of `max(floor, p99)` are now reported |
| 1c | `ALL_IDB_Dataset/` excluded: L1/L2 are ALL-IDB1 full fields and the boxes are single-class `Candidate_Cell` localisation, not cell types. | no — a training source is removed |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this
amendment.

Digests after amendment 1: see `results/PREREGISTRATION.sha256`.

---

## Amendment 2 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 2a | **Taleqani excluded**: its 224×224 `Original` images are downscaled microscopy *fields*, not cell crops (verified visually). The diagnosis is a field label. `lymphoblast` drops 2 882 → **130** cells. | no — a training source is removed, and the tier-B ALL claim is declared unsupportable at that scale |
| 2b | `granularity: cell \| field` is now **declared** per source in `sources.yaml` and required by `prepare_corpus.py`. Pixel dimensions had mis-classified two sources. | no |
| 2c | Every remaining corpus source inspected by eye and recorded as `granularity: cell  # verified visually 2026-09-10`. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

Digests after amendment 2: see `results/PREREGISTRATION.sha256`.

---

## Amendment 3 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 3a | **LeukemiaAttr wired in** as a cell-head source: 28 528 crops / 12 900 unique cells / 47 patients, cut from 640×640 fields with the block-1 convention. `lymphoblast` 130 → **2 234 cells across 19 ALL patients**, `lymphocyte_atypical` 11 → **656**, `promyelocyte_abnormal` 18 → **438**. | no threshold; `acute_blastic__lymphoid_oriented` returns from B− to **B** |
| 3b | Sharpness threshold **4.0 [OPS]**, applied per crop, chosen by visual binning; sensitivity at 2 and 6 in `results/leukemiaattr_retention.md`. Folder names do not predict sharpness (`L_100X_C1` measures 2.5). | new [OPS] value, declared and sensitivity-analysed |
| 3c | APML and CLL have **one patient per split**: those cells **train, never test**. `APL_suspicion` stays tier **A−**. §6.1 stays blocked (annotation is ~4.3 cells per field, incomplete). | no |
| 3d | LeukemiaAttr's own patient-disjoint split carried through; patient `6` forced to test (it straddled splits across subdomains); acquisitions of one physical cell grouped by `cell_uid_nomag`. | no |
| 3e | AML_APL / MILLIE inspected, **not** wired — the natural next step for a patient-level APL test. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

Digests after amendment 3: see `results/PREREGISTRATION.sha256`.

---

## Amendment 4 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 4a | **MILLIE / AML_APL wired in**: 8 291 labelled cells, 106 patients (APL 34 / AML 72). `smudge_cell` **15 → 1 570 across 56 patients** — §2 and amendment 3 had both declared this class unfixable by any available dataset, which was wrong. R3's trigger becomes measurable. | no — R3's thresholds, its `[REF]` policy and its tier **C** all stand: MILLIE holds no CLL patient |
| 4b | Declared mappings: `Blast_no_lineage_spec` → **partial label** {myeloblast, lymphoblast}; `Promyelocyte` → `promyelocyte` (normal — MILLIE does not label them abnormal, and 231/364 come from APL patients, so the question is left to evidence); `Promonocyte` → `monocyte`; thrombocytes/artifacts → `other_artifact`; `Plasma_cells` (6) skipped. 17 427 unlabelled `Unsigned_slides` not enumerated. | no |
| 4c | Split = the dataset's own Discovery (82) / Validation (24) cohorts, patient-level by construction. | no |
| 4d | `APL_suspicion` stays tier **A−**: training on these cells is not a test protocol. A patient-level APL test would need its own pre-declared endpoint. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

Digests after amendment 4: see `results/PREREGISTRATION.sha256`.

---

## Amendment 5 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 5a | **Early stopping** on both heads (cell head: 60 epochs max, patience 8, macro recall on `cell_val`; MIL head: 300 max, patience 30, AUROC on the new `mil_val`). Best weights restored, stopping epoch recorded. `aml_mll` patients re-split **0.45 train_mil / 0.15 mil_val / 0.20 dev_fit / 0.20 dev_cal** so model selection is paid for out of the training budget and never out of `dev_fit`/`dev_cal`. | no |
| 5b | Epoch budgets and patience are training hyper-parameters reported with the run, not `[OPS]` grid values. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

Digests after amendment 5: see `results/PREREGISTRATION.sha256`.

---

## Amendment 6 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 6a | **Prevalence is corrected, not counted.** Per-quantity Adjusted Classify and Count, uncertainty propagated into the frozen three-way test through the design effect. Raw counting made R3 assert `smudge_frac >= 2 %` on a smear with **zero** true smudge cells; corrected, it rejects. A quantity whose TPR−FPR < 0.05 is declared unquantifiable and its criterion returns indeterminate. | no threshold moves — the *estimator* feeding them changes |
| 6b | **Bag sampling fixed**: `features[:bag_size]` took the first fields scanned (spatial bias, and a breach of the deployed contract). `sampling.py` now reproduces `aster_pipeline/sampling.py` index-for-index. | no |
| 6c | **Early-stopping monitor** macro recall → macro **F1**; quantification MAE logged per epoch. Macro recall rewards over-predicting rare classes, which is the 6a failure. | no |
| 6d | **`other_artifact` cleaned**: LeukemiaAttr's `none` class (7 088 cells) contains recognisable leukocytes, not artifacts. Excluded; corpus 355 310 → 348 222. | no |
| 6e | Frozen encoder for the MIL head **declared** as a limitation with a candidate ablation, not corrected. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

Digests after amendment 6: see `results/PREREGISTRATION.sha256`.

---

## Amendment 7 — 2026-09-10, still before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 7a | **MILLIE split corrected**: only 56 patients carry labelled cells and all are Discovery, so the shipped cohorts leave zero validation cells. `smudge_cell` (1 555 of 1 570 from MILLIE) would then have had TPR = 0, been declared unquantifiable, and **R3 would have returned indeterminate by construction**. Replaced by a patient-level 0.80/0.20 split over the 56 labelled patients, stratified by diagnosis: 45/11 patients, and all 16 classes now have validation cells. | no |
| 7b | **5 truncated MILLIE JPEGs excluded**, counted and named rather than padded by `LOAD_TRUNCATED_IMAGES`. Corpus 348 222 → 348 217. | no |
| 7c | `lymphocyte_atypical` and `promyelocyte_abnormal` hold more validation than training cells (consequence of amendment 3c: one APML/CLL patient per split). Declared, not corrected — flipping the patients only reverses it. | no |

γ, the reference percentile, the tier sizes and every `[WHO]`, `[STD]` and `[CONV]` value
are unchanged. No `[REF]` or `[FIT]` value had been resolved at the time of this amendment.

---

## Amendment 8 — 2026-09-11, before any cAItomorph inference

| # | Change | Moves a threshold? |
|---|---|---|
| 8a | Lineage band fitted on **pseudo-bags of 40 predicted blasts** (= 200 [STD] × 20 % [WHO], derived, not chosen), aggregated as at inference, 5 % error cap per side — as §5.2 declared. The first implementation fitted single cells with an undeclared ±0.15 around the EER and gave [0.01, 0.31], which would have sent nearly every AML session to `lineage_indeterminate`. | a [FIT] value's *method* is brought in line with §5.2 |
| 8b | Three further scale mismatches in §7 fixed before any decision: `+inf` operating point from `argmax`; `τ_abn` raw vs calibrated; `[REF]` raw vs corrected fractions. | no declared value moves; fitted values are put on the scale they are compared at |

Digests after amendment 8: PREREGISTRATION.md 676e5c3b978c0a513c4dbdf6a3adc3301c8dd3396d8e67c371893dfa9edb27c5

---

## Amendment 9 — 2026-09-11, AFTER the primary cAItomorph inference (post hoc)

| # | Change | Moves a threshold? |
|---|---|---|
| 9a | Observed: R4 blocked on most stem-cell donors by `blast_frac < 5 %` (predicted blasts median 11.2 %). The same offset was visible before the test on the 60 AML-MLL controls (median 11.9 %, manual count 0/100) and was not acted on. Cause: cell-level TPR/FPR do not transfer to whole sessions of another domain. | no |
| 9b | `blast_frac` recalibrated **per session**: `observed = a + b·manual` fitted by OLS on the 189 AML-MLL patients (cell head never saw them), WHO blast equivalents as reference, bootstrap covariance of (a, b), quasi-binomial dispersion φ = max over control/AML strata, propagated through the design effect of 6a. `src/aster_block2/session_calibration.py`, now hashed. | no — the estimator feeding R1/R4's blast criterion changes |
| 9c | Declared before the run: applicability (lower IC95 of b ≥ 0.05), exact replay of the 409 primary labels, 5-fold CV on AML-MLL, cAItomorph once, every changed label listed, no success bar, no second variant. | no |
| 9d | Primary result unchanged and reported first. γ, percentile, tiers, rules, [WHO]/[STD]/[CONV] and every resolved [REF]/[FIT] value unchanged; `P_abn` untouched. | no |

Digests after amendment 9 (PREREGISTRATION.md 80cc1fccbb57f27d4fd13fdb1b380a92ac4286192fa2763d6b83ea25c2be54d7, session_calibration.py 2f646dd9a93d2fcc298d1fd34240f0c2a6dead7093f95690e12f526084a58ee0): see `results/PREREGISTRATION.sha256`.

