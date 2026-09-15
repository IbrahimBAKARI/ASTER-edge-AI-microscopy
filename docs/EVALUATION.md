# EVALUATION.md — measured performance of the prototype

Every number below is read from a file in `results/` (or in
`block2_development/training_run_20260911_051357Z/`), named next to it.
Configuration: the **frozen, pre-registered block 2** (decision grid frozen
2026-09-10, thresholds resolved 2026-09-11, amendments 1–8 adopted before the
test, amendment 9 analysed post hoc and not adopted). Out-of-domain threshold
**576.11**. Measurement environment: [`ENVIRONMENT.md`](ENVIRONMENT.md)
(NVIDIA Jetson Orin Nano 8 GB Super, JetPack 6.2, L4T R36.4.7, `nvpmodel`
MAXN_SUPER).

**Naming.** The fields acquired with the add-on through the ×40 dry objective
are called *prototype fields* in the article. In file and directory names they
carry the historical tag `x40` (for example `results/x40_testsplit/`,
`evaluation/x40_yield.py`); see [`../results/README.md`](../results/README.md)
for the mapping of every batch.

---

## 1. WBC localizer (block 1)

YOLO11n, single class, recipe `exp4_domain_aug` with `nonone` labels (real WBC
boxes only), fine-tuned to the prototype fields by mixed replay
(`localizer_adaptation/`). Deployed at **imgsz 960, conf 0.18, NMS IoU 0.50,
crop padding 10 %**.

### 1.1 Domain transfer before adaptation — `results/domain_transfer_12runs.json`, `results/localizer_transfer_478/x40_eval.csv`

Twelve recipes (four training-data regimes × three treatments of the `None`
boxes), trained on the public LeukemiaAttri data only:

| | Range over the 12 recipes |
|---|---|
| mAP@50, each recipe's public test split | 0.914 – 0.939 |
| mAP@50, the 478 prototype fields | 0.023 – 0.386 |
| Recall, the 478 prototype fields | ≤ 0.348 |

Spearman correlation between public-test and prototype mAP@50: ρ = 0.51
(p = 0.09). Per-recipe values: `results/derived/domain_transfer_12runs_with_x40_478.csv`.

### 1.2 Adaptation, held-out prototype test split (104 fields, 171 boxes) — `results/x40_testsplit/`, `results/x40_threshold/selection.json`

| | Value |
|---|---|
| Four candidate recipes before adaptation, mAP@50 | 0.306 – 0.396 (exp4/nonone: 0.371) |
| Deployed model after mixed replay, mAP@50 (conf 0.001) — `x40_testsplit/deployed_after_adaptation.json` | **0.982** |
| At the operating point (conf 0.1812, NMS 0.50), Ultralytics validator: P / R / mAP@50 / mAP@50-95 | 0.9211 / 0.9556 / 0.9565 / 0.6436 |
| Source-domain check, LeukemiaAttri test mAP@50 before → after | 0.934 → 0.925 (after: notebook run on Colab, see note) |

The two recipes that ranked first on the prototype fields (exp2/nonone and
exp4/nonone) were both adapted; exp4/nonone was retained
(`localizer_adaptation/README.md`). The "before" LeukemiaAttri value is in
`domain_transfer_12runs.json`; the "after" LeukemiaAttri value was printed by the Colab
adaptation notebook, whose run state is not part of this release.

### 1.3 Operating point — `results/x40_threshold/selection.json`, `curves.csv`

Maximum mean F1 on the 43-field prototype **validation** split (never the test
split), imgsz 960, NMS IoU ∈ {0.50, 0.60, 0.70}: **conf 0.18 / NMS IoU 0.50**
(best conf 0.1812; validation F1 0.9565, P 0.9296, R 0.985).

### 1.4 Leukocyte yield at native resolution — `results/leukocyte_yield/` (`evaluation/x40_yield.py`)

104 held-out test fields; ground truth 171 boxes, 1.64 WBC per field. Greedy
matching at IoU ≥ 0.5.

| conf | Detections | TP | FP | P | R | F1 | WBC/field (mean · sd) | Fields for 50 WBC |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 196 | 169 | 27 | 0.8622 | 0.9883 | 0.9210 | 1.885 · 1.211 | 26.5 |
| 0.15 | 185 | 166 | 19 | 0.8973 | 0.9708 | 0.9326 | 1.779 · 1.065 | 28.1 |
| **0.18** | **183** | **164** | **19** | **0.8962** | **0.9591** | **0.9266** | **1.760 · 1.079** | **28.4** |
| 0.25 | 180 | 164 | 16 | 0.9111 | 0.9591 | 0.9345 | 1.731 · 1.031 | 28.9 |
| 0.50 | 167 | 158 | 9 | 0.9461 | 0.9240 | 0.9349 | 1.606 · 0.882 | 31.1 |

At the mean yield of 1.76 WBC per field, the session tiers of block 2 need
about 57 fields (100 classified leukocytes), 114 fields (200) and 227 fields
(400) — `results/derived/session_cost_by_tier.json`.

Recall by triage verdict, before → after adaptation
(`results/derived/detections_x40test_by_triage.csv`): keep 0.394 → 0.958,
review 0.408 → 0.939, reject 0.490 → 0.980.

### 1.5 PyTorch ↔ TensorRT parity — `results/embedded_campaign/*/yolo_isolated*.json`

- Box parity, **deployed weights**, operating point, 9 ALL-IDB L2 fields:
  95 PyTorch boxes, 94 FP16 boxes, **93 matched, mean IoU 0.881**.

### 1.6 Isolated localizer latency — `results/embedded_campaign/`

| Median ms per field (imgsz 960) | DVFS | Pinned clocks |
|---|---|---|
| PyTorch `.pt` | 64.3 | 64.3 |
| TensorRT FP16 `.engine` | 48.7 | 48.5 |
| Speed-up | ×1.32 | ×1.33 |

The DVFS file `yolo_isolated_100runs.json` was transcribed from the console
output (its `_note` field says why); the pinned-clock file is the raw output.

---

## 2. Session decision (block 2)

### 2.1 Primary test — cAItomorph, 409 patients, pre-registered

Reference run: `block2_development/training_run_20260911_051357Z/results/caitomorph_409_predictions.csv`
(Colab, run `20260911_051357Z`). Re-scored on the Jetson with both backends:
`results/block2_v2_benchmarks/cai409_{trt,torch}/`. Cohort: stem-cell donors 99,
multiple myeloma 56, B-cell neoplasms 53, reactive changes 42, MDS 38, AML 37,
MPN 36, CMML 14, CML 8, ALL 7, 19 rarer diagnoses. Intervals: 95 % Wilson
(proportions), bootstrap (AUROC).

| Endpoint | Value |
|---|---|
| AUROC of *P*<sub>abn</sub>, AML vs donors | **0.973** (0.937 – 0.998) |
| AUROC of *P*<sub>abn</sub>, acute leukemia vs donors + reactive | 0.967 (0.935 – 0.991) |
| *P*<sub>abn</sub> ≥ τ = 0.933: AML / donors | 19/37 / 0/99 |
| Grid specificity, donors | 95/99 = 0.960 (0.901 – 0.984) |
| Grid specificity, reactive changes | 39/42 = 0.929 (0.810 – 0.975) |
| Grid sensitivity, acute leukemia (AML, ALL, AL) | 22/46 = 0.478 (0.341 – 0.619) |
| Grid sensitivity, AML | 17/37 = 0.459 (0.310 – 0.616) |
| `indeterminate` / `out_of_domain` | 182/409 = 0.445 / 46/409 = 0.112 |
| Chronic-pattern rules fired | 0/409 (declared inert on this cohort, not validated) |

Device reproduction:

| | TensorRT FP16 | PyTorch fp32 |
|---|---|---|
| Label agreement with the reference run | **409/409** | **409/409** |
| max \|Δ*P*<sub>abn</sub>\| | 0.0021 | 0.0006 |

Per-diagnosis labels: `results/derived/caitomorph_labels_by_group_frozen.csv`
and `block2_development/training_run_20260911_051357Z/results/caitomorph_409_confusion_full.csv`.
γ sensitivity (0.80 / 0.95, declared): `…/results/sensitivity_ops.csv`.

### 2.2 Golden-session parity — `models/mil/verify_report_20260912.json`

`block2_development/jetson/verify_on_jetson.py` on the device: **PASS on both
backends**. Crop tensors bit-identical; encoder cosine vs the reference ≥ 0.9999998
(PyTorch fp32) and ≥ 0.999998 (TensorRT FP16); label, tier and *P*<sub>abn</sub>
of the six golden sessions identical on both backends.

### 2.3 Out-of-domain gate on other acquisition domains

Complete deployed pipeline (block 1 + block 2). The prototype material is
non-leukemic; the gate is evaluated before the count gate.

| Session (fields / detected WBC) | Where | Label | OOD score (threshold 576.11) | *P*<sub>abn</sub> |
|---|---|---|---|---|
| Prototype test batch (104 / 183) | Jetson, TensorRT FP16 — `block2_v2_benchmarks/x40_stress_test/` | out_of_domain | 1024.0 | 0.966 |
| Prototype batch A (102 / 211; 200 scored) | Jetson — `x40_stress_train_part_aa/` | out_of_domain | 1150.6 | 0.972 |
| Prototype batch B (101 / 149) | Jetson — `x40_stress_train_part_ab/` | out_of_domain | 855.5 | 0.945 |
| Prototype validation batch (43 / 71) | CPU, PyTorch — `cpu_reruns_frozen/prototype_validation_batch/` | out_of_domain | 1096.9 | 0.957 |
| LeukemiaAttri AML, held-out (60 / 152) | CPU — `cpu_reruns_frozen/leukemiaattri_AML_heldout/` | out_of_domain | 1474.2 | 0.985 |
| LeukemiaAttri ALL, held-out (60 / 111) | CPU — `…/leukemiaattri_ALL_heldout/` | out_of_domain | 1602.2 | 0.990 |
| LeukemiaAttri APL, held-out (60 / 203) | CPU — `…/leukemiaattri_APL_heldout/` | out_of_domain | 1386.2 | 0.988 |
| LeukemiaAttri CML, held-out (60 / 363) | CPU — `…/leukemiaattri_CML_heldout/` | out_of_domain | 2823.5 | 0.002 |
| ALL-IDB L2 technical fixture (9 / 95) | CPU — `…/allidb_fixture/` | out_of_domain | 684.7 | 0.996 |

- 351 prototype fields were available to block 2 (train 204, validation 43,
  test 104); one training field has a broken JPEG stream and was excluded
  (350 read). Batches of 104, 102, 101 and 43 fields, limited by device memory.
- Batch A: the Jetson benchmark harness caps a session at 200 crops
  (`--crops-per-session 200`); the CPU re-run scores all 211 (OOD 1165.0).
- The Jetson record of the validation batch (`x40_stress_val/`) contains no
  session result (71 WBCs, below the 100-leukocyte screening tier in that
  harness); its row above comes from the deployed pipeline on CPU.
- The ALL-IDB fixture is the 9-field technical session of § 3; on the Jetson it
  was `out_of_domain` at every one of its 100 timing runs.
- CPU re-runs of the three Jetson batches agree with the Jetson within 1.3 %
  on the OOD score (`cpu_reruns_frozen/cpu_reruns_summary.csv`).
- Field lists: `fields.txt` in each `cpu_reruns_frozen/` directory; source
  datasets in [`DATA.md`](DATA.md).

### 2.4 Amendment 9 — post hoc, not adopted

`block2_development/results/AMENDMENT9_OUTCOME.md`,
`…/training_run_20260911_051357Z/results/amendment9_*.{json,csv}`.
Session-level blast recalibration on AML-MLL: intercept 0.133 (0.119 – 0.147),
slope 0.624, overdispersion 9.25 (controls) and 24.8 (AML). Applied once to the
409 patients: donor specificity 1.000, acute sensitivity 10/46, abstention
84.4 %. Not adopted; the primary result above is unchanged.

### 2.5 Block 2 latency on the device

Median ms, TensorRT FP16 (`block2_development/jetson/e2e/benchmark_block2_e2e.py`):

| Stage | 104-field prototype session, 183 crops, 15 runs — `block2_v2_benchmarks/timing_x40_test/` | 9-field ALL-IDB fixture, 95 crops, 100 runs — `timing_allidb/` |
|---|---|---|
| ResNet18 encoder (one pass per crop) | 158.9 | 98.9 |
| Cell head + gated-attention MIL head | 5.8 | 4.7 |
| Decision grid (all rules, three-way test) | 158.4 | 76.1 |
| **Block 2 total** | **323.5** | **180.6** |

### 2.6 Real 104-field prototype session, end to end — `block2_v2_benchmarks/timing_x40_test/`, `energy_x40_test/`

Native 4032 × 3040 fields; block 1 on the PyTorch path in this campaign.

| Stage | Median (15 runs) |
|---|---|
| Localization | 17.75 s |
| Crop extraction | 10.81 s |
| Block 2 | 0.32 s |
| Persistence | 9.69 s |
| **Total** | **39.2 s** (≈ 0.38 s per field) |

Energy (300 s window): mean board power 6.90 W, net energy 65.3 J per session,
peak RAM 6.38 GB, maximum GPU / CPU temperature 54.1 / 54.0 °C.

---

## 3. Nine-field technical session on the device

Fixture: `benchmark_session.txt` (9 ALL-IDB L2 fields, 95 detected WBCs), a
technical fixture, not a clinical session. Block 2 returns `out_of_domain` on it
at every run; the whole block-2 pass is still executed and timed.

### 3.1 Deployed pipeline — `results/block2_v2_benchmarks/deployed_latency/`, `deployed_energy/`

Median of 100 runs, block 1 on the PyTorch path in this campaign (no YOLO
engine), block 2 TensorRT FP16:

| Stage | ms |
|---|---|
| Localization (9 fields) | 775.0 |
| Crop extraction | 333.0 |
| Crop → tensor | 224.4 |
| Block 2: encoder / heads / grid | 105.7 / 4.9 / 77.4 |
| Output persistence | 356.2 |
| **Total** | **1892.0** (P95 1971.7) |

Power, 300 s per mode, VDD_IN rail:

| Mode | Mean power |
|---|---|
| Idle | 5.23 W |
| Camera acquisition (1280 × 720 at 30 fps) | 5.68 W |
| Analysis loop | 7.70 W (+2.48 W over idle) |
| Net energy per analysis | 4.61 J |
| Peak RAM · max GPU / CPU temperature | 2.97 GB · 54.9 / 55.2 °C |

## 4. Limitations and open items

- The prototype material is non-leukemic educational smears from one source,
  acquired on one day: the gate's refusal is measured, sensitivity on the
  target domain is not measurable.
- The operating point (conf 0.18) was selected on the prototype validation
  split, not on an independent source-domain split.
- In the campaigns of § 2.6 and § 3.1 block 1 ran on the PyTorch path; the
  TensorRT localizer numbers of § 1.6 are isolated measurements.
- Chronic-pattern rules never fired on the 409 patients: inert on this cohort,
  not validated.
- µm-per-pixel calibration of the optical path (stage micrometer) is not yet
  measured.
