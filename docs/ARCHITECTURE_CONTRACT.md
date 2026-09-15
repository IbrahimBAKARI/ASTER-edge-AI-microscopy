# Block 1 → block 2 contract

Every value below is read from the deployed code (`aster_pipeline/`) and from
`config/inference.yaml`, the single source of runtime parameters. Functions are
named rather than line numbers, so the document stays valid when the code moves.

---

## 0. Data flow

```
RGB field (file)
  └─ block 1: YoloWBCDetector.detect()                   aster_pipeline/yolo_detector.py
       ├─ Ultralytics YOLO.predict(imgsz=960, conf=0.18, iou=0.50, classes=[0]), one field per call
       ├─ expand_and_clip_box(box, W, H, padding=0.10)   aster_pipeline/crop_extraction.py
       ├─ extract_crop(image, crop_box) → PIL image at native pixels
       └─ persistence: crop PNG + annotated overlay + one row of crop_manifest.csv
  └─ block 2: LeukemiaPipeline.run() → SessionScorer.score()   aster_pipeline/pipeline.py, aster_pipeline/block2/
       ├─ build_eval_transform(224) on every crop:
       │     SquarePad(median of the 4 corners) → Resize((224,224)) → ToTensor(÷255) → Normalize(ImageNet)
       ├─ ResNet18 encoder: ONE pass per unique crop (TensorRT FP16 or PyTorch fp32) → features [N, 512]
       ├─ 16-class cell head on the features → per-class counts
       ├─ Adjusted Classify-and-Count quantifier → prevalence-corrected proportions
       ├─ gated-attention MIL head on a seeded bag (≤ 200 crops, seed 42) → P_abn (temperature 0.25)
       ├─ Mahalanobis out-of-domain gate (median squared distance over the crops)
       └─ frozen decision grid (three-way Jeffreys test, γ = 0.90, tiers 100/200/400
          classified leukocytes) → status + label + tier + flags
```

The frozen grid, its provenance tags and its amendments are in
[`../block2_development/PREREGISTRATION.md`](../block2_development/PREREGISTRATION.md)
and [`../block2_development/results/PREREGISTRATION.amendments.md`](../block2_development/results/PREREGISTRATION.amendments.md).

---

## 1. Block 1 output

### 1.1 Detection

| Parameter | Deployed value | Source |
|---|---|---|
| Backend | Ultralytics `YOLO(weights, task="detect").predict(...)`; `.pt` or TensorRT `.engine` | `YoloWBCDetector` |
| `imgsz` | **960** | `yolo.image_size` |
| Pre-processing | Ultralytics letterbox (stride 32, fill 114, RGB, ÷255, NCHW), identical for `.pt` and `.engine` | Ultralytics |
| `conf` | **0.18** | `yolo.confidence` |
| NMS `iou` | **0.50** | `yolo.iou` |
| `classes` | `[0]` (single class `wbc`; the class names of a `.pt` are checked at load) | `yolo.class_id` |
| Batch | one field per `predict` call | `YoloWBCDetector.detect` |
| Coordinates | `box.xyxy` in pixels of the original field | Ultralytics |

Each returned box is re-checked (`score ≥ conf` and `class_id == 0`) before a
crop is cut, so no box below the threshold reaches block 2.

### 1.2 Crop definition — `expand_and_clip_box()`

For a predicted box `(x1, y1, x2, y2)` in native pixels, box width `bw`, height
`bh`, field `W × H`, `padding = 0.10`:

```
px = 0.10 * bw ; py = 0.10 * bh          # added on EACH side
x1' = int(max(0,     x1 - px))
y1' = int(max(0,     y1 - py))
x2' = int(min(W - 1, x2 + px))
y2' = int(min(H - 1, y2 + py))
```

- The box grows by about 20 % in width and in height before clipping.
- A cell cut by the field border is simply clipped; no padding compensates the
  missing part at this stage (square padding happens in block 2).
- Output size: native pixels, variable from crop to crop; no resampling here.
- `extract_crop()` uses `PIL.Image.crop` (right/bottom exclusive). A degenerate
  box (`x2' ≤ x1'` or `y2' ≤ y1'`) is dropped: no crop, no overlay, no manifest
  row, not counted.

### 1.3 Persisted artefacts

| Artefact | Path | Format |
|---|---|---|
| Crops | `<out>/crops/field{NNN}_{stem}_wbc_{NNNN}.png` | PNG, compression level 1 (pixel-exact) |
| Overlays | `<out>/annotated/{NNN}_{stem}_annotated.jpg` | JPEG quality 92 (visualisation only) |
| Manifest | `<out>/crop_manifest.csv` | `crop_id, source_image, crop_path, score, class_id, x1, y1, x2, y2, padding` (x/y = crop box after expansion and clipping) |
| Result | `<out>/result.json` | § 3 |

`output.lossless_archival: true` writes overlays and crops as PNG level 6.

---

## 2. Block 2 input

### 2.1 Crop → tensor — `aster_pipeline/transforms.py::build_eval_transform(224)`

1. **`SquarePad`**: centred square padding to `max(w, h)`; the fill colour is the
   per-channel **median of the four corner pixels** of the crop.
2. **`Resize((224, 224))`**: bilinear, on the already square image.
3. **`ToTensor()`**: HWC → CHW, `uint8 → float32`, ÷ 255.
4. **`Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`** (ImageNet).

Result: `[3, 224, 224]` float32, RGB. No other normalisation and no test-time
augmentation. The same transform is used in training
(`block2_development/src/aster_block2/preprocess.py`); the parity is tested by
`block2_development/tests/test_preprocess_parity.py`.

### 2.2 Encoding and bag

- All retained crops are stacked (`[N, 3, 224, 224]`) and encoded **once**
  (`SessionScorer.encode`, chunks of `encode_chunk` = 64) → `[N, 512]`.
- The cell head reads the `N` feature vectors → per-class counts; `N_c` is the
  number of crops classified as a leukocyte class (`other_artifact` excluded).
- `deterministic_bag(N, bag_size=200, seed=42)` (`aster_pipeline/block2/sampling.py`,
  `np.random.default_rng`) draws **one** bag of `min(200, N)` indices without
  replacement for the gated-attention MIL head → raw probability → temperature
  scaling (T = 0.25) → `P_abn`.
- Evidence thresholds are expressed on `N_c`, never on the raw detection count.

---

## 3. Statuses, gates and labels

The grid's rule R0 is evaluated in this order: out-of-domain gate, then count
gate, then the classified-fraction floor.

| Condition | `status` | `label` | `final_label` (legacy) |
|---|---|---|---|
| 0 detected WBC | `no_decision` | `no_decision` | `no_decision` |
| OOD score (median squared Mahalanobis distance) > **576.11** | `out_of_domain` | `out_of_domain` | `no_decision` |
| `N_c` < 100 | `insufficient_evidence` | `insufficient_evidence` | `no_decision` |
| `N_c / N` < 0.9915 | `completed` | `indeterminate` | `no_decision` |
| 100 ≤ `N_c` < 200 (tier *screening*) | `completed` | `non_leukemic` / `suspicious_for_neoplasm` (from `P_abn` vs τ = 0.933) | derived |
| `N_c` ≥ 200 (tier *pattern*; ≥ 400 *reference*) | `completed` | full grid, rules R1–R5 | derived |
| uncaught exception | `failed` | — | `no_decision` |

Grid labels (`PREREGISTRATION.md` §1.2): `non_leukemic`,
`suspicious_for_neoplasm`, `acute_blastic__myeloid_oriented`,
`acute_blastic__lymphoid_oriented`, `acute_blastic__lineage_indeterminate`,
`chronic_myeloid_pattern`, `chronic_lymphoid_pattern`, `indeterminate`,
`insufficient_evidence`, `out_of_domain`. Flags: `APL_suspicion`,
`monocytic_predominance`, `reactive_lymphoid_observation`,
`nucleated_rbc_present`.

`legacy_final_label()` (`aster_pipeline/schemas.py`) maps this vocabulary onto
the three values read by `Interface/model_integration.py`: only
`acute_blastic__myeloid_oriented` → `AML`, only `non_leukemic` →
`non_leukemic_control`, everything else → `no_decision`.

`result.json` fields: `session_id, status, label, tier,
final_calibrated_probability` (= `P_abn`), `flags[], reasons[],
population_profile{}, quantities{}, verdicts{}, uncertainty{p_abn_raw, p_abn,
ood_score, ood_threshold}, evidence{top_attended, attention_weights},
number_of_fields, number_of_detected_wbc, number_of_classified_leukocytes,
device, inference_backend, grid_sha256, warnings[], timing_ms{}, final_label`.

---

## 4. Resolved thresholds — `models/mil/thresholds.json`

Resolved on development and control data only (never on the test cohort),
2026-09-11:

| Key | Value | Tag |
|---|---|---|
| `p_abn` (τ) | 0.9334 | FIT |
| `mil_temperature` | 0.25 | FIT |
| `ood_mahalanobis` | 576.1118 | FIT (99 % in-domain pass rate) |
| `min_classified_frac` | 0.9915 | FIT |
| `apl_abn_promy` | 0.0147 | FIT |
| `lineage_lo` = `lineage_hi` | 0.3113 | FIT (equal error rate) |
| `ig_cml` / `baso_cml` | 0.10 / 0.0258 | REF |
| `lymph_cll` / `smudge_cll` | 0.50 / 0.1373 | REF |
| `atypical_reactive` | 0.10 | REF |

---

## 5. Open points

1. **Operating point `conf 0.18 / iou 0.50`**: selected by maximum mean F1 on
   the prototype **validation** split (43 fields), never on the test split
   (`results/x40_threshold/selection.json`); not re-validated on an independent
   source-domain split.
2. **Crop framing, training vs production**: block-2 training crops were cut
   by the reference crop functions of `block2_development/src/aster_block2/crops.py`
   (same expansion and clipping convention), from sources with their own
   framing; production crops come from the YOLO boxes above.
3. **Scope**: screening and morphological pattern statements, not a diagnosis;
   lineage and subtype are deferred to immunophenotyping and molecular testing.
   44.5 % of the 409 test patients are `indeterminate` by design
   ([`EVALUATION.md`](EVALUATION.md) § 2.1).
