# Amendment 9 — outcome (2026-09-11)

Run once, as declared in PREREGISTRATION.md amendment 9. Raw outputs in the Drive run
directory: `results/amendment9_*.{json,csv}`.

## Fit on the 189 AML-MLL patients (60 control, 129 AML)

| | value | IC95 |
|---|---|---|
| observed blast fraction, median | control 0.119, AML 0.486 | |
| `a` session false-positive rate | 0.1327 | [0.1193, 0.1474] |
| `b` session TPR − FPR | 0.6236 | [0.5834, 0.6605] |
| dispersion φ | 24.84 (AML), 9.25 (control) — adopted 24.84 | |
| regression-dilution ratio | 0.990 | |
| 9c-1 applicable | yes | |

## 9c-3 — 5-fold CV on AML-MLL

| | cell-level quantifier | amendment 9 |
|---|---|---|
| controls: `blast ≥ 5 %` REJECTED | 8/60 = 0.13 [0.07, 0.24] | 0/60 = 0.00 [0.00, 0.06] |
| controls: `blast ≥ 20 %` ASSERTED (harm) | 2/60 = 0.03 [0.01, 0.11] | 0/60 = 0.00 [0.00, 0.06] |
| AML manual ≥ 20 %: `blast ≥ 20 %` ASSERTED | 124/127 = 0.98 [0.93, 0.99] | 93/127 = 0.73 [0.65, 0.80] |

## cAItomorph, 409 patients (9c-2 replay: 409/409 reproduced)

| | frozen (primary) | amendment 9 |
|---|---|---|
| `non_leukemic`, donors | 29/99 = 0.293 [0.212, 0.389] | 0/99 = 0.000 [0.000, 0.037] |
| `non_leukemic`, reactive | 23/42 = 0.548 [0.399, 0.688] | 0/42 = 0.000 [0.000, 0.084] |
| specificity vs donors | 95/99 = 0.960 [0.901, 0.984] | 99/99 = 1.000 [0.963, 1.000] |
| specificity vs reactive | 39/42 = 0.929 [0.810, 0.975] | 42/42 = 1.000 [0.916, 1.000] |
| sensitivity acute (AML/ALL/AL) | 22/46 = 0.478 [0.341, 0.619] | 10/46 = 0.217 [0.123, 0.356] |
| indeterminate | 182/409 = 0.445 [0.398, 0.493] | 345/409 = 0.844 [0.805, 0.876] |

163 labels changed: 128 `non_leukemic → indeterminate`, 35 `acute_blastic → indeterminate`.
No donor or reactive patient newly labelled leukaemic.

## Reading

1. **The offset is real but is not the obstacle.** The session-level false-blast rate is
   13 %, precisely estimated. What blocks a `< 5 %` blast claim is the **patient-to-patient
   scatter** of that rate: φ ≈ 9 on controls, ≈ 25 on AML, far above binomial counting.
   At 400 cells a control then carries ~16 effective cells even with the control-only φ;
   rejecting `blast ≥ 5 %` needs φ of the order of 5 or less.
2. **Amendment 9 is not adopted.** It removes every false leukaemic label (specificity
   1.000) but abstains on 84 % of patients and halves acute sensitivity. The deployed
   estimator stays the frozen cell-level quantifier; the primary result stands as reported.
3. **Limitation carried into the paper.** The frozen three-way test models counting and
   cell-level classifier error, not patient-level scatter. Its nominal γ = 0.90 is therefore
   not the confidence actually achieved per patient on blast claims; the primary's
   empirical sensitivity/specificity are the measured performance, the γ is not.
4. **Design target for the next cell head.** Patient-level dispersion of blast calls on
   normal blood, φ ≲ 5, measured on controls — reachable only by reducing domain-driven
   variability (target-domain cell labels, unfrozen encoder — amendment 6e — or stain
   normalisation), not by a statistical correction.
