# ASTER block 2 — pre-registered decision grid

**Version:** 2.0 — **FROZEN 2026-09-10**, before any inspection of the cAItomorph test
set. Hashes in `results/PREREGISTRATION.sha256`.
**Written:** 2026-09-10, before any cAItomorph prediction was computed.
**Language:** English, so that §2–§6 can be lifted into the paper's Methods.

---

## 0. Why this document exists

The ASTER block-2 output vocabulary (AML-like, ALL-like, chronic myeloid pattern,
chronic lymphoid pattern, non-leukemic) is not backed by patient-level training data
for every class: two of the five classes have **zero** training patients. The only
material in the world containing those patients — cAItomorph — is the released test
set.

The validity of the design therefore rests on a strict separation between what is
learned, what is fitted, and what is declared:

| Component | Status | Basis |
|---|---|---|
| Cell classifier, blast lineage, abnormal-population score | **learned** | patient- or cell-level labels that exist |
| Operating points, APL threshold, lineage bands, OOD threshold | **fitted** | a development split disjoint from the model's training patients **and** from the test set |
| Chronic myeloid / chronic lymphoid criteria | **declared** | WHO/ICC criteria, conventional reference values, and the reference-interval rule of §5.3 |
| Every numeric value in §4 | **frozen before test** | this document |

A rule validated on 8 CML patients is a legitimate result reported with a wide
confidence interval. A classifier *tuned* on 8 CML patients is not, and its optimism
is not estimable. This document is what makes the first statement true.

---

## 1. Output vocabulary and claim tiers

### 1.1 Claim tiers anchored on documented counting standards

The system does not make the same claim on 50 cells and on 500. §3.3 shows why: with
50 classified leukocytes, asserting "blasts ≥ 20 %" requires 28 % observed blasts, so
the pattern layer is not resolvable at that session size.

The session sizes are **not** an ASTER invention. They are the counting standards of
manual cytology, adopted as they are **[STD]**:

| Tier | `N_c` | Documented basis | What is emitted |
|---|---|---|---|
| — | < 100 | below any recognised counting standard | `insufficient_evidence`; no differential-based claim |
| **S — screening** | ≥ 100 | routine manual leukocyte differential | `non_leukemic` / `suspicious_for_neoplasm` from the calibrated MIL head, plus the bag differential. No pattern label. |
| **P — pattern** | ≥ 200 | **manual differential at diagnosis is performed on 200 nucleated leukocytes** (EHA/ELN blast-reporting recommendation); the WHO 20 % blast threshold is applied to that count | the full grid of §4 |
| **R — reference** | ≥ 400 | **CLSI H20-A2 reference method: 400 cells** (200 on each of two slides, two observers) | the grid, annotated as reference-count-equivalent |

**What tier R does and does not replicate.** CLSI's 400 cells are split over two
slides and two observers precisely to control slide-preparation and observer
variability. An ASTER session is one specimen read by one model: only the **count** is
equivalent, not the replication design. The paper states this; tier R is never called
"the CLSI reference method".

**Consequence for the deployed operating point.** The current 50-WBC gate sits below
every recognised standard and is therefore raised. At the measured ×40 yield of
1.76 WBC/field:

| Tier | `N_c` | ×40 fields required |
|---|---|---|
| S | 100 | ≈ 57 |
| P | 200 | ≈ 114 |
| R | 400 | ≈ 227 |

At ~48 ms/field of localisation this is ~11 s of computation for tier R — the cost is
**operator acquisition time, not compute**. That reinforces rather than weakens the
paper's central finding.

On the test cohort the question does not arise: cAItomorph has a median of 500 cells
per patient, and only 10 of 409 patients fall below 400 — essentially the whole cohort
is evaluated at tier R.

### 1.2 Labels

| Label | Tier | Meaning |
|---|---|---|
| `non_leukemic` | S | reference or reactive leukocyte population |
| `suspicious_for_neoplasm` | S | abnormal population, pattern not resolved |
| `acute_blastic__myeloid_oriented` | P | blast excess, blast morphology oriented myeloid (AML-like) |
| `acute_blastic__lymphoid_oriented` | P | blast excess, blast morphology oriented lymphoid (ALL-like) |
| `acute_blastic__lineage_indeterminate` | P | blast excess, lineage not separable on morphology |
| `chronic_myeloid_pattern` | P | left shift with basophilia (CML-like) |
| `chronic_lymphoid_pattern` | P | mature lymphocytosis with smudge cells (CLL/lymphoproliferative-like) |
| `indeterminate` | — | criteria not met with sufficient statistical confidence |
| `insufficient_evidence` | — | fewer than 100 classified leukocytes [STD] |
| `out_of_domain` | — | encoder features outside the validated acquisition domain |

Secondary flags, emitted alongside a label, never alone:

| Flag | Meaning |
|---|---|
| `APL_suspicion` | abnormal/bilobed promyelocyte excess — **urgent**, confirm PML::RARA |
| `monocytic_predominance` | monocytosis without left shift (CMML-like observation) |
| `reactive_lymphoid_observation` | atypical lymphocytes without smudge cells |
| `nucleated_rbc_present` | erythroblasts among the localized objects |

Every label is a **morphological pattern statement**, not a diagnosis. Lineage
assignment and disease entity require immunophenotyping and molecular testing.

### 1.3 Evidence tier per output class (paper table)

| Output | Training | Validation | Tier |
|---|---|---|---|
| `non_leukemic` | 60 control patients | 99 donors + 42 reactive | **A** learned and tested patient-level |
| `acute_blastic__myeloid_oriented` | 129 AML patients | 37 AML + 2 AL | **A** |
| `APL_suspicion` | 24 PML_RARA patients + 186 abnormal-promyelocyte cells (LeukemiaAttr, **1 patient**) | (inside the 37 AML) | **A−** — the extra cells train it, they never test it (amendment 3) |
| `acute_blastic__lymphoid_oriented` | 2 001 cells / **15 ALL patients** (LeukemiaAttr) + 130 crops | 7 ALL | **B** — lineage learned cell-level on patient-structured data; wide CI on 7 test patients |
| `chronic_myeloid_pattern` | **none** | 8 CML (+36 MPN, 14 CMML, 22 MDS-MPN as neighbours) | **C** declared rule, tested only |
| `chronic_lymphoid_pattern` | **none** | 53 B-cell neoplasms, 3 HCL | **C** |

---

## 2. Cell classes (the learned differential)

15 morphological classes + 1 reject class.

| Class | AML-LMU (cells) | Other source |
|---|---|---|
| `myeloblast` | MYO 3268, MOB 26 | — |
| `lymphoblast` | — | LeukemiaAttr 2 234 cells / 19 ALL patients, ALL-IDB2 130 — amendment 3 |
| `promyelocyte` | PMO 70 | PBC `PMY` 592, MILLIE 364 |
| `promyelocyte_abnormal` | PMB 18 | LeukemiaAttr 438 — amendment 3 |
| `myelocyte` | MYB 42 | PBC `MY` 1137 |
| `metamyelocyte` | MMZ 15 | PBC `MMY` 1015 |
| `band_neutrophil` | NGB 109 | PBC `BNE` 1633 |
| `segmented_neutrophil` | NGS 8484 | PBC `SNE` 1646 |
| `basophil` | BAS 79 | PBC basophil |
| `eosinophil` | EOS 424 | PBC eosinophil |
| `monocyte` | MON 1789 | PBC monocyte |
| `lymphocyte` | LYT 3937 | PBC `LY` 1214, MILLIE 1 902 |
| `lymphocyte_atypical` | LYA 11 | LeukemiaAttr 1 206, MILLIE 162 |
| `smudge_cell` | KSC 15 | MILLIE 1 555 / **56 patients** — amendment 4 |
| `erythroblast` | EBO 78 | PBC erythroblast |
| `other_artifact` | — | PBC platelets 2348, negative crops, block-1 false positives |

**PBC partial labels.** Only the 151 files prefixed `IG` are partial (the true class is
known to lie in {promyelocyte, myelocyte, metamyelocyte}); they are trained with a
partial-label loss and never assigned to one of the three. The other 2 744 immature
granulocytes carry a definite prefix. See amendment 1.

**Declared weakness, revised.** With PBC wired in, the left-shift classes are no longer
the weak point: `metamyelocyte` 15 → 1 030, `myelocyte` 42 → 1 179, `promyelocyte`
70 → 662, `band_neutrophil` 109 → 1 742. The remaining thin classes are
`smudge_cell` (15), `lymphocyte_atypical` (11) and `promyelocyte_abnormal` (18), all
from AML-LMU alone — that is, the **CLL rule and the APL flag**, not the CML rule.
Per-class precision and recall are reported individually and those three carry the
uncertainty.

---

## 3. Session quantities and the three-way test

### 3.1 Definitions

Let the bag be the `N` localized WBC crops from block 1, and `N_c` the number not
classified `other_artifact`. All proportions are of `N_c`.

```
blast_frac      = (myeloblast + lymphoblast + promyelocyte_abnormal) / N_c
ig_frac         = (promyelocyte + myelocyte + metamyelocyte) / N_c
baso_frac       = basophil / N_c
lymph_frac      = lymphocyte / N_c
smudge_frac     = smudge_cell / N_c
atypical_frac   = lymphocyte_atypical / N_c
mono_frac       = monocyte / N_c
abn_promy_frac  = promyelocyte_abnormal / N_c
lineage_post    = Σ p(lymphoblast) / (Σ p(lymphoblast) + Σ p(myeloblast))  over blasts
P_abn           = calibrated attention-MIL probability of an abnormal population
```

**This is a bag differential, not a manual differential.** The denominator is the set
of objects a YOLO localizer detected; its per-class recall is not uniform and is
measured separately (§6.1). The paper names it as such throughout.

### 3.2 The three-way test (`src/aster_block2/proportion_test.py`)

No criterion compares a point estimate to a threshold. For a class count `k` of `n`,
under the Jeffreys posterior `Beta(k + ½, n − k + ½)`:

```
ASSERT        if  P(p ≥ θ | k, n) ≥ γ
REJECT        if  P(p < θ | k, n) ≥ γ
INDETERMINATE otherwise
```

A rule fires only when **every** criterion returns ASSERT. If any is INDETERMINATE
and none is REJECT, the session is `indeterminate`. Abstention is therefore a
property of the sample, not a tuned behaviour: 2 basophils in 100 cells gives
P = 0.55 for "≥ 2 %" and cannot assert basophilia; 6 in 100 gives P = 0.99 and can.

`γ = 0.90` **[OPS]**, sensitivity-analysed at 0.80 and 0.95.

### 3.3 Statistical resolution — observed count needed to assert

Computed with the module above at γ = 0.90:

| θ | N_c = 50 | 100 | 200 | 400 | 500 |
|---|---|---|---|---|---|
| 2 % | 3 (6.0 %) | 4 (4.0 %) | 7 (3.5 %) | 12 (3.0 %) | 15 (3.0 %) |
| 5 % | 5 (10.0 %) | 8 (8.0 %) | 15 (7.5 %) | 26 (6.5 %) | 32 (6.4 %) |
| 10 % | 8 (16.0 %) | 14 (14.0 %) | 26 (13.0 %) | 48 (12.0 %) | 59 (11.8 %) |
| 20 % | 14 (28.0 %) | 26 (26.0 %) | 48 (24.0 %) | 91 (22.8 %) | 112 (22.4 %) |
| 50 % | 30 (60.0 %) | 57 (57.0 %) | 110 (55.0 %) | 213 (53.2 %) | 265 (53.0 %) |

This table is the justification for the tier-P requirement of 200 cells, and it is a
paper figure: *the width of the abstention region as a function of session size*. It
is also why cAItomorph is the right cohort to validate the decision layer — median
**500 cells/patient**, only 10 of 409 patients below 400.

### 3.4 Minimum session sizes

| Rule family | Minimum `N_c` | Tag |
|---|---|---|
| any output at all | 100 | **[STD]** routine manual differential |
| tier-S screening statement | 100 | **[STD]** |
| acute rules (R1) | 200 | **[STD]** diagnostic differential, 200 nucleated leukocytes |
| chronic rules (R2, R3) | 200 | **[STD]** |
| `non_leukemic` (R4) | 200 | **[STD]** |
| reference-count annotation | 400 | **[STD]** CLSI H20-A2 |

Below the minimum, only the tier-S statement is emitted — never a pattern label,
never a negative pattern call. Below 100 classified leukocytes, nothing is emitted.

A pre-declared secondary analysis repeats §6.3 restricted to the patients reaching
tier R (≥ 400 cells), to show what the grid does at reference count.

---

## 4. The decision grid (frozen)

Provenance tags:

| Tag | Meaning | Who answers for the value |
|---|---|---|
| **[WHO]** | WHO/ICC classification criterion | the classification |
| **[STD]** | documented manual-cytology counting standard (§1.1) | CLSI H20-A2, EHA/ELN |
| **[CONV]** | conventional adult peripheral-blood reference value | the literature |
| **[REF]** | `max(clinical floor, 99th percentile of the reference population)` — §5.3 | the control data |
| **[FIT]** | fitted on the development split — §5.2 | the development data |
| **[OPS]** | residual discretionary choice | us, sensitivity-analysed |

Rules are evaluated in order; the first that fires wins.

### R0 — Gates
```
N_c < 100                       [STD] → insufficient_evidence  (below routine differential)
Mahalanobis OOD above threshold [FIT] → out_of_domain
N_c / N below threshold         [FIT] → indeterminate  (localizer output not interpretable)
100 <= N_c < 200                [STD] → tier-S statement only
N_c >= 400                      [STD] → tier-R annotation added
```

### R1 — Acute blastic pattern
```
blast_frac ≥ 0.20                              [WHO]
```
If asserted:

**R1a — APL suspicion (evaluated first)**
```
abn_promy_frac ≥ θ_APL                         [FIT]  (24 PML_RARA vs 105 other AML,
                                                       operating point at specificity ≥ 0.95)
AND abn_promy_frac > myeloblast_frac           [OPS]
   → acute_blastic__myeloid_oriented + APL_suspicion
```

**R1b — Blast lineage**
```
lineage_post ≥ θ_hi                            [FIT]  → acute_blastic__lymphoid_oriented
lineage_post ≤ θ_lo                            [FIT]  → acute_blastic__myeloid_oriented
otherwise                                             → acute_blastic__lineage_indeterminate
```
`θ_lo`, `θ_hi` are set on cell-level pseudo-bags (§5.2) at equal error rate, and the
resulting indeterminate band is reported, not minimised.

### R2 — Chronic myeloid pattern (CML-like)
Evaluated only when R1 is REJECTED.
```
ig_frac   ≥ max(0.10, p99_ref)                 [REF]  clinical floor: CML left shift
AND baso_frac ≥ max(0.02, p99_ref)             [REF]  clinical floor: [CONV] basophilia
AND NOT (mono_frac ≥ 0.10 AND ig_frac < 0.10)  [WHO]  CMML exclusion
   → chronic_myeloid_pattern
```
Recorded but **not used** in v1: `myelocyte_frac > metamyelocyte_frac` (the myelocyte
bulge), evaluated post hoc as a candidate refinement.

### R3 — Chronic lymphoid pattern (CLL/lymphoproliferative-like)
Evaluated only when R1 is REJECTED.
```
lymph_frac  ≥ max(0.50, p99_ref)               [REF]  clinical floor: [CONV] 20–45 % normal
AND smudge_frac ≥ max(0.02, p99_ref)           [REF]  clinical floor: Gumprecht cells
AND NOT (atypical_frac ≥ p99_ref AND smudge_frac < threshold)  [REF]  reactive exclusion
   → chronic_lymphoid_pattern
```

### R4 — Non-leukemic
```
blast_frac  < 0.05                             [CONV] blasts absent from normal blood
AND ig_frac < 0.02                             [CONV]
AND lymph_frac ≤ 0.50                          [CONV]
AND P_abn   < τ_abn                            [FIT]  specificity ≥ 0.95 on the calibration split
   → non_leukemic
```

### R5 — Fallback
```
anything else                                  → indeterminate
```
Observation flags attach to any label when their own criteria assert:
`monocytic_predominance` (`mono_frac ≥ 0.10` [WHO]),
`reactive_lymphoid_observation` (`atypical_frac ≥ p99_ref` [REF]),
`nucleated_rbc_present` (`erythroblast ≥ 1` cell [CONV]).

### 4.1 The learned score does not leak authority

`P_abn` is trained on AML versus control only. It appears in **R1 and R4 only**. For
chronic sessions it is recorded and reported but never used to assert a label: a high
`P_abn` on a CLL smear is out-of-training-distribution behaviour of the MIL head, not
evidence. Stated in the paper.

---

## 5. How each value is obtained

### 5.1 What is never fitted
Every **[WHO]** and **[CONV]** value. They are not varied in any analysis.

### 5.2 The development split (**[FIT]**)

A patient-level split of AML-MLL, **disjoint from the patients used to train the cell
head and the MIL head**, and never containing cAItomorph. Fitting a threshold on the
model's own training patients would make it look better than it is.

| Value | Development data | Criterion |
|---|---|---|
| `τ_abn` | AML-MLL dev patients | operating point at specificity ≥ 0.95 |
| `θ_APL` | 24 PML_RARA vs 105 other AML | specificity ≥ 0.95 among AML |
| `θ_lo`, `θ_hi` | cell-level pseudo-bags: lymphoblasts (ALL-IDB, Taleqani) vs myeloblasts (LMU) | equal error rate |
| OOD threshold | in-domain dev bags | 99 % in-domain pass rate |
| `N_c / N` floor | dev + ×40 sessions | 1st percentile of the dev distribution |
| MIL temperature | dev calibration split | temperature scaling |

**Pseudo-bag caveat.** ALL sources have no patient identifiers (verified: ALL-IDB
crops and Taleqani filenames carry none). Pseudo-bags are therefore built from
image-level pools, and `θ_lo`/`θ_hi` are declared as fitted on a construct, not on
patients. The tier-B label of §1.3 reflects this.

### 5.3 The reference-interval rule (**[REF]**)

Thresholds for the chronic rules cannot be fitted — there are no CML or CLL
development patients. They are instead set on the **reference population**, the way a
clinical laboratory sets a reference interval: from control **patients** only, never from
a case. The reference intervals are per-patient proportions, so a source without patient
identifiers cannot contribute to them — PBC therefore trains the cell head but takes no
part here (amendment 1). The population is the AML-MLL control patients of
`dev_fit` ∪ `dev_cal`, disjoint from the patients the models were trained on.

```
threshold = max( clinical floor , 99th percentile of the reference population )
```

The percentile is **[OPS]** (99th, sensitivity-analysed at 97.5 and 99.5). The
`max` with a clinical floor prevents the opposite failure: normal donors carry almost
no immature granulocytes, so a raw p99 would fire on any mild reactive left shift.

The resulting statement is empirical and interpretable: *this proportion exceeds what
99 % of reference smears show, and exceeds the clinical criterion.* No CML or CLL patient
is consumed.

**Declared in advance:** the reference population is about **24 control patients**. An
empirical 99th percentile on 24 observations is unstable and close to the maximum, so the
clinical floor is expected to dominate the `max(·, ·)` in most rules. Both terms are
reported for every threshold — the floor, the empirical percentile, the population size,
and which one was adopted — so the reader can see which mechanism actually decided.

**Optional upgrade.** If Raabin-WBC is confirmed to contain CLL cases, the smudge and
lymphocyte thresholds move from **[REF]** to **[FIT]** on real disease development
data, and cAItomorph stays untouched. No equivalent public development source is
known to us for CML; that rule stays **[REF]** and tested on 8 patients.

### 5.4 Residual discretionary values (**[OPS]**)

After §5.2 and §5.3, and once the session sizes are taken from the documented
standards of §1.1 **[STD]**, three values remain discretionary. All are
sensitivity-analysed and the result is reported whatever it shows:

| Value | Frozen at | Sensitivity range |
|---|---|---|
| `γ`, assertion confidence | 0.90 | 0.80, 0.95 |
| reference percentile | 99th | 97.5th, 99.5th |
| `abn_promy_frac > myeloblast_frac` (R1a) | as written | dropped / kept |

The session-size tiers are no longer among them: they are the counting standards of
manual cytology, and the tier-S/P/R comparison of §6.3 is a reporting axis rather than
a tuning knob.

A grid that only works at its frozen values is a fitted grid and will be reported as
such.

---

## 6. Pre-declared evaluation

### 6.1 Localizer per-class recall (prerequisite)
Block-1 recall measured per cell class before any differential is interpreted.
Without it, a low correlation in §6.2 cannot be attributed to the cell head.

### 6.2 Differential validity
Bag differential versus the AML-MLL per-patient manual WBC differential columns,
Spearman ρ per class with CI, reported per class, never as a single average.

### 6.3 Primary test — cAItomorph, 409 patients
Confusion matrix of the §1.2 vocabulary against `diagnosis_fine`, one row per fine
diagnosis (Stem cell donor 99, MM 56, B-cell neoplasm 53, Reactive changes 42,
MDS 38, AML 37, MPN 36, CMML 14, CML 8, ALL 7, MDS/MPN 6, HCL 3, ET 3, AL 2,
MPN/MDS-RS-T 2, T-cell 1, PV 1, PCL 1), with Clopper–Pearson intervals and the
abstention rate per row.

Endpoints by evidence tier (§1.3):
- **Tier A** — `non_leukemic` vs `acute_blastic__*`: sensitivity, and specificity
  reported **separately against Reactive changes and against stem-cell donors**.
- **Tier A−** — `APL_suspicion` within the AML rows.
- **Tier B** — blast lineage on 37 AML + 7 ALL + 2 AL: accuracy and indeterminate
  rate, wide CI expected.
- **Tier C** — chronic patterns, exploratory, wide CI, **excluded from the abstract**.

### 6.4 Acquisition-domain stress test
All available prototype ×40 sessions. Declared expectation: `out_of_domain` or
`indeterminate`. Reported against the current block-2 behaviour on the same session
(`AML`, 5/5 folds, calibrated p = 0.986).

### 6.5 What would falsify the design
Declared now:
- Spearman ρ < 0.5 on `myeloblast` in §6.2 → the cell head does not support the acute
  axis; the grid is withdrawn and only `P_abn` is reported.
- A chronic rule firing on more stem-cell donors than CML or B-cell patients → that
  rule is reported as **failed, not retuned**.
- ×40 sessions returning a confident class → the OOD gate has failed; block 2 stays
  scoped out of the deployment domain, as in the current paper.
- Tier-P abstention above 50 % on cAItomorph → the grid is reported as
  under-powered at the session sizes available, and the tier-S screening statement
  becomes the headline claim.

---

## 7. Freeze

**Frozen on 2026-09-10**, before the first cAItomorph inference was run.

The SHA-256 of this file, of `src/aster_block2/decision_grid.yaml` and of
`src/aster_block2/proportion_test.py` are recorded in
`results/PREREGISTRATION.sha256`. The hashes are held in a separate file by
construction: a document cannot contain its own hash.

The two discretionary values fixed at freeze:

| Value | Frozen at | Sensitivity analysis (still mandatory) |
|---|---|---|
| `γ`, assertion confidence | **0.90** | 0.80, 0.95 |
| reference percentile (**[REF]**) | **99th** | 97.5th, 99.5th |

Verification:

```bash
cd aster-block2 && shasum -a 256 -c results/PREREGISTRATION.sha256
```

Any later change is an amendment: a new dated entry appended to
`results/PREREGISTRATION.sha256` with its justification, and reported in the paper.
Values tagged **[REF]** and **[FIT]** are resolved during Phase 1 from the development
and control data only; writing those resolved numbers into `decision_grid.yaml` is an
expected amendment and is recorded as such. Nothing is ever resolved from cAItomorph.

---

## Amendment 1 — 2026-09-10, before any cAItomorph inference

Three corrections, all made while wiring the corpus and all **before** the test set was
read. The original freeze digests are preserved in `results/PREREGISTRATION.amendments.md`.

### 1a. PBC / Acevedo carries finer labels than assumed

The frozen §2 assumed PBC's immature granulocytes arrived only as the merged `ig` class.
Inspection of the files shows the labels are in the filenames:

| Prefix | Class | Count |
|---|---|---|
| `PMY` | promyelocyte | 592 |
| `MY` | myelocyte | 1 137 |
| `MMY` | metamyelocyte | 1 015 |
| `IG` | unspecified — stays a partial label | 151 |
| `BNE` | band neutrophil | 1 633 |
| `SNE` | segmented neutrophil | 1 646 |

**Effect:** the classes carrying the CML left-shift rule go from 15–109 training cells to
1 030–1 742. The declared weakness of §2 moves off the CML rule and onto the CLL rule
(`smudge_cell` 15, `lymphocyte_atypical` 11) and the APL flag (`promyelocyte_abnormal` 18),
which PBC does not supply. `[REF]` and `[FIT]` values are unaffected. This amendment
strengthens a class the pre-registration had declared weak; it does not move a threshold.

### 1b. PBC cannot contribute to the `[REF]` reference intervals

§5.3 named the reference population as "60 AML-MLL controls + PBC normal donors". The
reference intervals are **per-patient proportions**, and PBC has no patient identifiers,
so it cannot enter them. Corrected population: the AML-MLL **control patients of
`dev_fit` ∪ `dev_cal`**, ≈ 24 patients, disjoint from the MIL training patients.

Declared consequence: an empirical 99th percentile on ~24 observations is unstable and
close to the maximum, so the clinical floor is expected to dominate `max(floor, p99)`. For
every `[REF]` threshold the report gives the floor, the empirical percentile, the
population size, and which term was adopted.

### 1c. `ALL_IDB_Dataset/` excluded

`L1` and `L2` are ALL-IDB1 **full fields** (2592×1944 and 1712×1368), not crops, and their
493 boxes carry the single class `Candidate_Cell` — WBC localisation, not cell type. The
folder also holds a stray class list containing entries such as `car`, `dog`, `hamburger`,
and the `_0`/`_1` filename convention is applied inconsistently. Excluded from v1 rather
than injecting unresolved label noise into `lymphoblast`, the class the tier-B ALL claim
rests on. `lymphoblast` training therefore comes from ALL-IDB2 (130) and Taleqani (2 752).
See `datasets/DATASETS.md`.

---

## Amendment 2 — 2026-09-10, still before any cAItomorph inference

### 2a. Taleqani excluded: it is a field dataset, not cell crops

The phase-0 inventory recorded Taleqani as single-cell crops on the strength of its
224×224 dimension. **Visual inspection shows the opposite**: the `Original` images are
downscaled microscopy *fields* holding several leukocytes and many erythrocytes, with a
burned-in scale bar. `Segmented/` is not a mask of the labelled cell either — it is a
colour-threshold segmentation of every nucleus in the field, and it fails outright on many
images (mask coverage 0.0–0.1 % on the samples checked).

The diagnosis is therefore a **field** label. In an ALL field not every leukocyte is a
blast, so labelling each detected cell `lymphoblast` would inject exactly the label noise
refused for `ALL_IDB_Dataset` in amendment 1c. Source excluded.

**Consequence, stated plainly:** `lymphoblast` training collapses from 2 882 cells to
**130** (ALL-IDB2 `cancer` only), and `lymphocyte` loses 504 cells. The blast-lineage band
`θ_lo`/`θ_hi` of R1b is fitted on pseudo-bags built from those 130 crops against
AML-LMU myeloblasts. **The tier-B claim `acute_blastic__lymphoid_oriented` is not
supportable at that scale.** Unless a patient-level ALL source is added, R1b should be
expected to return `acute_blastic__lineage_indeterminate` for nearly every blastic
session, and the paper must say so rather than report a lineage accuracy on 130 crops.

### 2b. Granularity is now a declared property, not an inference

Two sources were mis-classified from their pixel dimensions alone (Taleqani here,
`ALL_IDB_Dataset` in amendment 1c). `datasets/sources.yaml` now requires
`granularity: cell | field` for every source, set only after looking at the images, and
`datasets/prepare_corpus.py` refuses to run a source that does not declare it. A source
declared `field` whose records carry no box is refused rather than resized.

The size-based guard (`crops.guard_is_single_cell`, 512 px) is kept as a second line of
defence, but it cannot catch a downscaled field and is no longer the primary check.

### 2c. Visual verification of every remaining source

Each source kept in the corpus was inspected by eye on 2026-09-10 and is a centred
single-cell crop: cAItomorph 144², AML-MLL 144², AML-LMU 400², ALL-IDB2 257²,
PBC 360×363. Recorded in `datasets/sources.yaml` as
`granularity: cell  # verified visually 2026-09-10`.

---

## Amendment 3 — 2026-09-10, still before any cAItomorph inference

### 3a. LeukemiaAttr added as a cell-head source

The source ships 640×640 microscopy **fields** with COCO boxes, so it enters through the
field path of amendment 2b: `datasets/extract_leukemiaattr.py` cuts every box with the
deployed block-1 convention (`aster_block2.crops`, byte-identical to
`aster_pipeline/crop_extraction.py`), and `datasets/clean_crops.py` keeps the crops whose
measured sharpness reaches the threshold below. **28 528 crops, 12 900 unique cells,
47 patients.**

What it changes, and it is the reason for the amendment:

| Class | Before | After |
|---|---|---|
| `lymphoblast` | 130 crops, **no patient identifier** | **2 234 cells across 19 ALL patients** |
| `lymphocyte_atypical` | 11 | **656** |
| `promyelocyte_abnormal` | 18 | **438** |

`acute_blastic__lymphoid_oriented` therefore returns from **B−** to **B**: amendment 2a had
declared it unsupportable on 130 crops, and it is now learned on patient-structured data.
The R1b lineage band is fitted on those cells instead of on pseudo-bags built from 130
crops.

### 3b. The sharpness threshold, and why a filter rather than a folder rule

The 11 usable subdomains re-acquire the same physical cells at different sharpness and
with different cameras. Measured at the 224 block-2 input size (variance of the Laplacian),
subdomain medians span 1.1 to 15.3 — and the folder name does not predict it: `L_100X_C1`
measures 2.5, as blurry as the 40× subdomains.

The threshold is **4.0**, chosen by inspecting crops binned by sharpness
(`_work/bandes_nettete.png`): below 4 the cell is still recognisable as a cell, but the
fine chromatin texture is gone — and that texture is exactly what separates a lymphoblast
from a lymphocyte, or an abnormal promyelocyte from a normal one. A label neither a model
nor a human can verify is noise, not supervision. It is applied **per crop**, which
recovers 3 857 sharp crops from the 40× subdomains and discards 5 511 blurred ones from
the 100× subdomains.

Cost, stated: 27.2 % of crops and 43 % of unique cells survive. The gains in the table
above are net of that cost.

**[OPS]**, sensitivity-analysed at 2 and 6 — the per-class cell counts at every threshold
are in `results/leukemiaattr_retention.md`.

### 3c. Two limits carried into the paper

1. **APML and CLL have one patient on each side of the split.** `promyelocyte_abnormal`
   and `lymphocyte_atypical` consequently hold as many test cells as training cells, all
   from a single patient per split. **These cells train; they never test.** No test metric
   for `APL_suspicion` or for the reactive-lymphoid exclusion may be reported from this
   source — those tests stay cAItomorph and the AML-MLL PML_RARA patients. The evidence
   tier of `APL_suspicion` therefore stays **A−**.
2. **The annotation is incomplete**: about 4.3 cells are annotated per field while more
   leukocytes are visible. Usable for labelled crops; **not** usable to measure localizer
   recall, so §6.1 remains blocked.

### 3d. Splits

LeukemiaAttr's own train/test split is patient-disjoint and is carried through rather than
redrawn — but it is disjoint only *within* a subdomain in the shipped data: patient `6`
appears in train in one subdomain and in test in another. Any patient seen in test anywhere
is assigned to test everywhere (34 train / 13 test patients). `cell_uid_nomag` groups the
acquisitions of one physical cell, so no two acquisitions of the same cell can land on
opposite sides.

### 3e. Not added

**AML_APL / MILLIE** (106 patients, 34 APL / 72 AML) is inspected and documented but not
wired in this amendment. It is the natural next step for the APL flag, since it is the
only patient-level APL cohort that could *test* it.

---

## Amendment 4 — 2026-09-10, still before any cAItomorph inference

### 4a. MILLIE / AML_APL added — the smudge-cell gap is closed

Manescu 2023. **8 291 labelled single cells across 106 patients** (APL 34 / AML 72),
filed per patient under `Signed_slides/<class>/`. Images are 360×363 centred single cells,
sharpness 29–152 — the PBC band, and far above LeukemiaAttr. Verified visually, not
inferred from the dimension.

Amendment 3 and §2 both declared `smudge_cell` the weakest class of the design and stated
that no available dataset could fix it. **That was wrong**: MILLIE carries
**1 555 smudge cells across 56 patients**.

| Class | Before | After |
|---|---|---|
| `smudge_cell` | **15** (AML-LMU only) | **1 570 across 56 patients** |
| `lymphocyte` | 5 689 | 7 591 |
| `lymphocyte_atypical` | 1 217 | 1 379 |
| `promyelocyte` | 662 | 1 026 |
| `erythroblast` | 1 629 | 1 845 |

`chronic_lymphoid_pattern` (R3) rests on `smudge_frac`. Its trigger was effectively
undetectable; it is now trainable. **The rule itself is unchanged** — thresholds, `[REF]`
policy and evidence tier **C** all stand, because MILLIE contains no CLL patient: it makes
the criterion measurable, it does not make the pattern learned.

### 4b. Declared mappings

`Blast_no_lineage_spec` (1 865 cells) is a **partial label** over
{`myeloblast`, `lymphoblast`}. The annotation states the lineage was not specified; the
cohort is AML/APL, but inferring myeloid lineage from the cohort would be inference, not
annotation.

`Promyelocyte` (364 cells, 231 of them from APL patients) maps to `promyelocyte`, the
**normal** class — MILLIE does not label them abnormal. The patient diagnosis is carried in
the manifest so a later analysis can settle it on evidence rather than on assumption.
`Promonocyte` → `monocyte`, as for LeukemiaAttr. `Giant_thrombocyte`,
`Thrombocyte_aggregation` and `Arifact` → `other_artifact`. `Plasma_cells` (6 images) has
no counterpart in the frozen vocabulary and is **skipped** rather than filed as an artifact.

The 17 427 `Unsigned_slides` carry no label and are **not enumerated**: no declared use.

### 4c. Split

The dataset's own **Discovery (82) / Validation (24)** cohorts are carried through rather
than redrawn. Patient-level by construction.

### 4d. What this does NOT change

`APL_suspicion` stays tier **A−**. MILLIE's 34 APL patients could *test* the flag, and that
is the natural next step, but wiring the cells for training does not by itself constitute a
test protocol: that would need its own pre-declared endpoint, and it is not declared here.
Any APL test metric still comes from cAItomorph and the AML-MLL PML_RARA patients.

---

## Amendment 5 — 2026-09-10, still before any cAItomorph inference

### 5a. Early stopping, and the split it is paid for from

Both heads now train to a large epoch budget and stop early on a monitored metric, instead
of running a fixed number of epochs chosen by hand:

| Head | Max epochs | Patience | Monitored on | Metric |
|---|---|---|---|---|
| cell head | 60 | 8 | `cell_val` | **macro** recall over definite-label cells |
| MIL head | 300 | 30 | **`mil_val`** (new) | AUROC |

The best weights are restored, and the stopping epoch is recorded in
`results/{cell,mil}_head_training.json`.

**Why a new split.** Early stopping is model selection. Monitoring it on `dev_fit` would
choose the stopping epoch on the same patients that later fix `τ_abn`, `θ_APL` and the OOD
threshold, and those operating points would look better than they are. Model selection is
therefore paid for out of the **training** budget: `aml_mll` patients are now split
**0.45 `train_mil` / 0.15 `mil_val` / 0.20 `dev_fit` / 0.20 `dev_cal`**, all patient-level
and stratified by `bag_label`. `dev_fit` and `dev_cal` keep exactly the roles §5.2 gives
them.

**Why macro recall for the cell head.** Overall accuracy is dominated by
`segmented_neutrophil` (15 253 cells) and would tolerate a model that never predicts
`smudge_cell` (1 570), `lymphocyte_atypical` (1 379) or `promyelocyte_abnormal` (784) — the
classes R1a and R3 depend on. Macro recall makes a rare class count as much as a common
one.

### 5b. What this does not change

No `[WHO]`, `[STD]`, `[CONV]`, `[REF]` or `[FIT]` value moves. The epoch budgets and the
patience values are training hyper-parameters, not decision thresholds: they are reported
with the run, not sensitivity-analysed as `[OPS]` grid values.

---

## Amendment 6 — 2026-09-10, still before any cAItomorph inference

Four corrections found by re-examining the design rather than the data. The first is a
threat to the validity of R3 and is the reason for this amendment.

### 6a. Counting a classifier is not measuring a prevalence

Every criterion in §4 is a proportion, and until now those proportions were estimated by
counting the cell classifier's `argmax` predictions. That is "Classify and Count", the
known-worst prevalence estimator: its bias depends on the classifier's error rates and on
the true prevalence, and for rare classes it is severe.

**The failure it produces.** Take a smudge-cell classifier at 95 % specificity — a good
figure. On a **normal** smear containing no smudge cells at all, 5 % of the other
leukocytes are counted as smudge. `smudge_frac >= 0.02` is crossed, and the three-way test
of §3.2 — which treats those counts as clean multinomial draws from the true proportion —
**asserts** it with high confidence. R3 fires on normal blood because of classifier error,
not biology. The test quantifies sampling uncertainty; it never quantified classification
error.

**Correction.** Per-quantity Adjusted Classify and Count. TPR and FPR are measured for each
grid quantity on the definite-label validation cells — per *quantity*, because
`blast_frac` is a sum of three classes and the error rate of "is this a blast" is not the
sum of three per-class rates. Then

```
observed = TPR · true + FPR · (1 − true)      →      true = (observed − FPR) / (TPR − FPR)
```

with the variance propagated by the delta method and expressed as an effective `(k, n)`
through the design effect `n_eff = n · Var_raw / Var_corrected`, so the **frozen** test of
`proportion_test.py` consumes it unchanged and the classifier's uncertainty widens the
interval instead of vanishing.

Simulated at 95 % specificity, 200 classified leukocytes:

| True prevalence | Raw verdict | Corrected verdict |
|---|---|---|
| 0 % | **assert** ← false positive | **reject** |
| 1 % | indeterminate | **reject** |
| 3 % | assert | assert |
| 30 % | assert | assert |

**A quantity whose TPR and FPR cannot be separated (< 0.05) is not assertable at all**:
the criterion returns `indeterminate` and the rule cannot fire. That is reported, never
retuned.

### 6b. The bag was not sampled, it was truncated

`inference.py` took the first `bag_size` crops in file order — the first fields scanned. A
smear is not homogeneous along the scan path (feathered edge versus body), so that is a
spatial bias, not a sample, and it also broke the deployed contract. `sampling.py` now
reproduces `aster_pipeline/sampling.py` exactly: `default_rng(seed + repeat · 10007)`,
verified index-for-index.

### 6c. The early-stopping monitor was wrong

Amendment 5a chose macro **recall**. That rewards over-predicting rare classes — which is
precisely the failure of 6a. The monitor is now macro **F1**, and the quantification MAE
(mean absolute error on class proportions) is logged every epoch as the diagnostic closest
to what the grid actually reads.

### 6d. `other_artifact` was being trained on real leukocytes

7 088 of its cells came from LeukemiaAttr's COCO class `none`, which means "not one of the
13 annotated classes", **not** "artifact". A random sample shows plainly recognisable
neutrophils, monocytes and lymphocytes among them. Training the reject class on those
teaches the model to discard real cells, which would shrink `N_c`, bias the differential
non-randomly and corrupt the `min_classified_frac` gate. Excluded: the corpus drops from
355 310 to **348 222** images, and `other_artifact` now rests on PBC platelets (2 348) and
MILLIE artifacts (21) — thin but clean, and extensible with block-1 false positives from
the ×40 sessions, which are artifacts by construction.

### 6e. Declared limitation, not corrected

The encoder is **frozen** when the MIL head trains: the head can only use features the cell
head needed. That is the price of encoding each crop once, which is what pays for the cell
head, the attention head and the OOD test inside the deployed latency budget. Declared as a
limitation and listed as a candidate ablation (unfreeze `layer4`), not silently accepted.

---

## Amendment 7 — 2026-09-10, still before any cAItomorph inference

### 7a. MILLIE is split on its labelled patients, not on the shipped cohorts

Amendment 4c carried MILLIE's own Discovery (82) / Validation (24) cohorts through. Building
the corpus showed why that cannot work: **only 56 patients carry `Signed_slides`, and all
56 are Discovery.** The 24 Validation patients hold `Unsigned_slides` only, which are
unlabelled and not enumerated.

The consequence would have been silent and severe. Every labelled MILLIE cell would have
landed in `cell_train` with **no validation cell at all** — and `smudge_cell` comes from
MILLIE for 1 555 of its 1 570 cells. The quantifier of amendment 6a estimates TPR on the
validation split; with no validation cell it would have measured TPR = 0, declared
`smudge_frac` unquantifiable, and **R3 would have returned `indeterminate` by construction**
for every session. The chronic-lymphoid rule would have looked broken while the real cause
was a split.

Corrected: the 56 labelled patients get their own patient-level 0.80 / 0.20 split,
stratified by diagnosis. Result: 45 train / 11 validation patients, and every one of the 16
cell classes now has validation cells — `smudge_cell` 1 384 / 185.

### 7b. Five corrupt source files excluded

Five MILLIE JPEGs are truncated (`Patient_24`, `30`, `36`, `37`, `52`). Pillow can pad a
truncated JPEG with `LOAD_TRUNCATED_IMAGES`, which would have put half-decoded cells into
the training set under valid labels with nothing downstream noticing. They are excluded,
counted, and named in `_work/corpus/corrupt_files.txt`. Corpus: 348 222 → **348 217**.

### 7c. Known imbalance, declared not corrected

`lymphocyte_atypical` (623 train / 756 validation) and `promyelocyte_abnormal`
(262 / 522) hold more validation than training cells. This follows from amendment 3c: APML
and CLL have one LeukemiaAttr patient per split, and the test-side patient happens to be
imaged in the sharper subdomains, so more of its cells survive the sharpness filter.
Flipping the two patients would only reverse the imbalance. Both sides keep hundreds of
cells, which is what the quantifier needs; the asymmetry is reported with the results.

---

## 8. Standards cited for the [STD] and [WHO] values

To be verified against the primary documents before submission — the values below
were taken from the secondary literature and must be confirmed page-in-hand.

1. **CLSI H20-A2**, *Reference Leukocyte (WBC) Differential Count (Proportional) and
   Evaluation of Instrumental Methods*, Approved Standard, 2nd ed., 2007 — reference
   manual differential = **400 cells**, 200 on each of two slides by two observers.
   <https://clsi.org/shop/standards/h20/>
2. **EHA/ELN expert panel**, *Reporting blast percentage for response assessment in
   acute leukemias*, Haematologica — at diagnosis the manual peripheral-blood
   differential is performed on **200 nucleated leukocytes** whenever possible.
   <https://haematologica.org/article/view/12257>
3. **WHO classification of myeloid neoplasms and acute leukaemia** — the ≥ 20 % blast
   criterion (WHO 2001 onward; retained in WHO 2016/2017 and, with entity-specific
   exceptions, in WHO 2022 and ICC 2022). Note for the discussion: **ICC 2022 lowers
   the threshold to 10 % for defined genetic entities and WHO 2022 removes the blast
   cut-off for several of them** — the ASTER grid uses the classical 20 % morphological
   criterion and says so explicitly.
   <https://ashpublications.org/blood/article/127/20/2391/35255/>
4. **ICSH / Kratz et al.**, *Digital morphology analyzers in hematology: ICSH review
   and recommendations*, Int J Lab Hematol 41(4):437–447, 2019 — already cited as
   reference [2] of the paper; the pre-classification-then-verification model that
   ASTER's rule layer follows.
