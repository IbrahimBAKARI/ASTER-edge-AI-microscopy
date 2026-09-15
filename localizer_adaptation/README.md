# Localizer adaptation to the prototype fields — mixed-replay fine-tune

Adapt the WBC localiser to the prototype fields (**×40 dry** objective; tag `x40` in file names) without
losing source-domain (LeukemiaAttri) performance, and stay robust to staining variation.

## Data

The prototype-field dataset (`ASTER_X40_DATASET`, see `docs/DATA.md`; available from the
authors on reasonable request) — built by `make_x40_split.py` from the 478 annotated fields.
Triage verdicts: keep (`GARDER` in the raw triage files), review (`A_REVOIR`), reject (`REJET`).

| split | fields | boxes | content |
|---|---|---|---|
| train | 204 | 334 | keep + review only (clean material) |
| val | 43 | 67 | keep + review |
| test | 104 | 171 | **all verdicts incl. reject** — representative of deployment, held out |

Split unit = acquisition cluster (fields > 90 s apart). Whole clusters go to one
split → no field is seconds away from a field in another split. Seed 42.
Details + cluster→split map: `x40_SPLIT.md`; per-field list: `data_splits/prototype_fields_split.csv`.
127 reject fields (very blurry / heavy precipitate) from train/val clusters are
left out of training; they stay available for a robustness experiment.

## Step 1 — baseline on the held-out test (before fine-tune)

```bash
python localizer_adaptation/eval_on_x40_test.py --device cuda
```
→ `results/x40_testsplit/x40_testsplit.csv`. This is the number the fine-tune
must beat, on the exact same 104 fields.

## Step 2 — fine-tune (Colab; LeukemiaAttri lives there, Drive folder `LLD/`, see `docs/DATA.md` § 2)

**Executable source of truth: `05_mixedreplay_x40_colab.ipynb`** (committed).
It mounts LeukemiaAttri + the ×40 dataset, builds the mixed data lists, measures before,
fine-tunes **both** `exp2_multisharpness/nonone` and `exp4_domain_aug/nonone`,
and measures after. `localizer_adaptation/finetune_x40_mixedreplay.py` is the readable
distillation of the notebook's training step (cell 6); run it only if you have
the mixed `data.yaml` already built.

```bash
python localizer_adaptation/finetune_x40_mixedreplay.py \
    --data /content/mix/exp4_domain_aug__nonone_mix.yaml \
    --base <exp4_domain_aug/nonone base best.pt> \
    --project runs --name mixedreplay__exp4_domain_aug__nonone --device 0
```

Key choices (notebook cells 1–2 and 6):
- **bases = exp2_multisharpness/nonone AND exp4_domain_aug/nonone** — the two best
  ×40 candidates off-the-shelf (mAP50 0.396 / 0.371). **Winner = exp4/nonone**
  (×40-test 0.371 → 0.982, LeukemiaAttri test kept) — the deployed model.
- **mixed replay**, `REPLAY_RATIO=4`: each base replays a seed-42 sample of *its
  own* original train list (exp2 → multi-sharpness, exp4 → multi-domain), sized
  4× the ×40 train set → ~20 % ×40 per batch. Sees both domains every batch.
- **no `freeze`** (full fine-tune), **no `cos_lr`** (linear decay), `imgsz=960`,
  `lr0=0.001`, 60 ep, **patience 10**, `degrees=15`.
- **`hsv_h=0.5`, `hsv_s=0.7`** (vs default 0.015 / 0.7) — the ×40 gap is a colour
  gap (RBC = WBC = purple). Heavy hue jitter forces the model onto morphology,
  making it robust to over-staining now *and* to a well-stained slide later.
- The Colab run also diverged from the earlier script draft in `args.yaml`:
  `single_cls: false` (not true).

## Step 3 — evaluate

The notebook does this itself (cells 5 "before" and 6 "after"), writing
`state/mixedreplay_results.json`. To redo locally on the held-out splits:

```bash
# held-out x40 test (the deployment number)
python localizer_adaptation/eval_on_x40_test.py --weights <best.pt> --tag x40_finetuned --device cuda
# LeukemiaAttri test must not regress
python -c "from ultralytics import YOLO; YOLO('<best.pt>').val(data='<lld_data.yaml>', split='test', classes=[0])"
```

Real numbers: **LeukemiaAttri test 0.934 → 0.925 (kept)**,
**×40-test 0.371 → 0.982 (adapted)** for exp4/nonone (the 0.982 is re-computed in
`results/x40_testsplit/deployed_after_adaptation.json`). Success bar the notebook
checks: ×40-test mAP50 > 0.396 and recall ≥ 0.6, LeukemiaAttri test mAP50 ≥ 0.92.

## Then — done (2026-09-09)

- Threshold re-selected on the x40 **val** split (`scripts/select_threshold_x40val.py`):
  **conf 0.18 / NMS IoU 0.50**. `results/x40_threshold/selection.json`.
- `evaluation/x40_yield.py` on the mixed-replay `.pt` (104 held-out ×40 test fields,
  engine parity already established): at conf 0.18, **P 0.896 / R 0.959 / F1 0.927,
  1.76 WBC/field, ~28 fields for 50 WBC**. `results/leukocyte_yield/`.
- Still open: re-staining decision (yield is workable, so likely not needed).
