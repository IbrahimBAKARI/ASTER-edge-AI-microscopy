# ASTER — detachable edge-AI add-on for conventional microscopes

Code, models and measured results of the article

> I. Bakari Sani, I. Debbarh, N. Machkour, I. Battas, "A Detachable Edge-AI Add-On for
> Conventional Microscopes: Acquisition-Domain Adaptation and Pre-Registered,
> Abstention-Aware Session-Level Leukemia Screening," manuscript for *IEEE Access*, 2026.

> **Research prototype.** The models support leukemia *screening* and
> *morphological pattern* statements at session level. They are not a diagnosis,
> not a medical device, and must not be used for any clinical decision.

---

## What the prototype does

```
microscopy fields (4032 × 3040, ×40 dry objective)        one acquisition session
  └─ block 1  YOLO11n single-class WBC localizer, adapted to the add-on's own fields
              imgsz 960, conf 0.18, NMS IoU 0.50, crop padding +10 % per side
  └─ block 2  one ResNet18 encoder pass per crop → 16-class cell head
              → prevalence-corrected proportions (Adjusted Classify-and-Count)
              → gated-attention MIL screening score P_abn
              → Mahalanobis out-of-domain gate (threshold 576.11)
              → frozen decision grid: three-way Jeffreys test, γ = 0.90,
                tiers of 100 / 200 / 400 classified leukocytes
  └─ result.json (status, label, tier, flags, reasons, evidence) + overlays + crops + manifest
```

Hardware: NVIDIA Jetson Orin Nano 8 GB (Super), USB C-mount camera (Arducam IMX477
HQ), interchangeable ocular / C-mount adapters, 7-inch touch display, NVMe SSD.
The host microscope is not modified.

**Configuration.** This release contains the frozen, pre-registered block 2
(grid frozen 2026-09-10; amendments 1–8 before the test; amendment 9 post hoc,
not adopted). Every result in `results/` and `docs/EVALUATION.md` was obtained
with exactly the models in `models/`.

---

## Repository map

| Path | Content |
|---|---|
| `run.py`, `config/`, `aster_pipeline/`, `backend/`, `YOLO/`, `Interface/`, `tests/` | the deployed pipeline: entry point, single parameter file (`config/inference.yaml`), blocks 1 and 2, service layer, PyQt5 touch interface (native-resolution capture, [`Interface/CAMERA.md`](Interface/CAMERA.md)), tests |
| `models/` | localizer weights, block-2 encoder / heads / grid / thresholds / OOD statistics, checksums — [`models/MODELS.md`](models/MODELS.md) |
| `localizer_adaptation/` | how the localizer was adapted to the prototype fields (Colab notebook, split builder, evaluator) |
| `evaluation/` | leukocyte yield of the deployed localizer |
| `scripts/` | on-device benchmarks (latency, energy), TensorRT engine builders, operating-point selection, campaign procedures |
| `block2_development/` | block 2: pre-registration and amendments, training code and notebook, record of the training run, technical report, Jetson verification kit — [`block2_development/README.md`](block2_development/README.md) |
| `results/` | every measured result — [`results/README.md`](results/README.md) |
| `data_splits/` | every split (prototype fields, block-2 corpus) — [`data_splits/README.md`](data_splits/README.md) |
| `docs/` | [`EVALUATION.md`](docs/EVALUATION.md) (all measured numbers), [`ARCHITECTURE_CONTRACT.md`](docs/ARCHITECTURE_CONTRACT.md) (block 1 → block 2 contract), [`DATA.md`](docs/DATA.md) (data sources and availability), [`ENVIRONMENT.md`](docs/ENVIRONMENT.md) (Jetson stack) |

Rebuilding anything: [`REPRODUCE.md`](REPRODUCE.md).

---

## Quick start (PyTorch path, CPU or CUDA)

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt      # outside a Jetson: remove the two index lines first (REPRODUCE.md § 0)
cd models && shasum -a 256 -c SHA256SUMS.txt && cd ..
.venv/bin/python run.py --input <folder of session fields> --session-id demo \
    --output results_sessions/demo --device cpu
.venv/bin/python -m pytest tests/ -q
```

Statuses: `completed` (a label, possibly `indeterminate`), `insufficient_evidence`
(< 100 classified leukocytes), `out_of_domain` (outside the validated acquisition
domain), `no_decision` (no WBC detected). The TensorRT FP16 path needs engines
built on the device: `scripts/export_yolo_trt.sh`, `scripts/build_block2_trt.sh`.

---

## Key results

Details and intervals: [`docs/EVALUATION.md`](docs/EVALUATION.md).

| Quantity | Value |
|---|---|
| Localizer, 12 recipes trained on public data: public test vs 478 prototype fields (mAP@50) | 0.914–0.939 vs 0.023–0.386 |
| Localizer after mixed-replay adaptation, 104 held-out prototype test fields (mAP@50) | 0.371 → **0.982**; public test 0.934 → 0.925 |
| Leukocyte yield at conf 0.18 | 1.76 WBC per field (57 / 114 / 227 fields for 100 / 200 / 400 leukocytes) |
| Block 2, cAItomorph held-out test (409 patients), AUROC of P_abn, AML vs donors | **0.973** (0.937–0.998) |
| Decision grid: specificity donors / reactive; sensitivity acute leukemia | 95/99, 39/42; 22/46 (44.5 % `indeterminate`) |
| Jetson vs cloud run, 409 patients | 409/409 identical labels (TensorRT FP16 and PyTorch) |
| Out-of-domain gate on the add-on's own non-leukemic fields | every session withheld (`out_of_domain`) |
| Block 2 on the device, 104-field session (183 crops) | 0.32 s of a 39.2 s analysis |
| Board power: idle / camera / analysis loop | 5.23 / 5.68 / 7.70 W |

---

## Data availability

- **Public datasets** (LeukemiaAttri, AML-Cytomorphology_MLL_Helmholtz,
  AML-Cytomorphology_LMU, PBC, MILLIE, ALL-IDB, cAItomorph) are distributed by
  their authors under their own terms and are **not** redistributed here; their
  identifiers are in [`docs/DATA.md`](docs/DATA.md) and the construction of every
  split in [`data_splits/`](data_splits/).
- **Prototype fields acquired with the add-on** (478 fields and their WBC
  annotations; non-leukemic educational smears, no patient data): **available
  from the authors on reasonable request.** Their split, triage verdicts, capture
  times and every result computed on them are included.

## Licence

GNU Affero General Public License v3.0 ([`LICENSE`](LICENSE)). The localizer
weights are derived from Ultralytics YOLO11 (AGPL-3.0). Public datasets keep
their own terms.

## Citation

See [`CITATION.cff`](CITATION.cff).
