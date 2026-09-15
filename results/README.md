# results/ — measured artefacts

Every number in [`../docs/EVALUATION.md`](../docs/EVALUATION.md) is read from
one of these files. All block-2 results use the frozen, pre-registered
configuration (OOD threshold 576.11).

**Placeholders.** Absolute paths of the machines that produced these records
were replaced by placeholders — `<REPO>` (this repository), `<DATA_ROOT>`,
`<ALL_IDB>`, `<PROTOTYPE_FIELDS>`, `<PROTOTYPE_DATASET>`, `<BLOCK2_HANDOFF>`
(the block-2 device kit folder), `<TRAINING_WORKDIR>`, `<FIELDS>` (the input
folder of a session), `<OUTPUT>`, `<WORKDIR>`, `<HOME>`. Nothing else in the
records was changed; the `fields_sha256` and `sha256` values of the manifests hash
file contents and are unaffected.
In `crop_manifest.csv` files, `source_image` is `<FIELDS>/<file name>` and
`crop_path` is `crops/<file name>`. The CPU re-run directories carry English
names, also used in `cpu_reruns_frozen/cpu_reruns_summary.csv` and as the `session_id`
of their `result.json`.

## Naming of the prototype batches

Files and directories use the historical tag `x40` for the fields acquired with
the add-on through the ×40 dry objective (*prototype fields* in the article).

| Directory | Article name |
|---|---|
| `block2_v2_benchmarks/x40_stress_test/`, `cpu_reruns_frozen/prototype_test_batch/` | prototype test batch (test split, 104 fields, 183 WBCs) |
| `block2_v2_benchmarks/x40_stress_train_part_aa/`, `cpu_reruns_frozen/prototype_batch_A/` | prototype batch A (training split, 102 fields) |
| `block2_v2_benchmarks/x40_stress_train_part_ab/`, `cpu_reruns_frozen/prototype_batch_B/` | prototype batch B (training split, 101 readable fields) |
| `block2_v2_benchmarks/x40_stress_val/`, `cpu_reruns_frozen/prototype_validation_batch/` | prototype validation batch (validation split, 43 fields) |
| `x40_testsplit/`, `x40_threshold/`, `leukocyte_yield/` | prototype test / validation fields (localizer) |

## Contents

| Path | Content | `docs/EVALUATION.md` |
|---|---|---|
| `domain_transfer_12runs.json` | 12 YOLO11n recipes (4 training-data regimes × 3 `None`-box treatments) on the public test split of each recipe (`lld_test`) | § 1.1 |
| `localizer_transfer_478/x40_eval.csv` | the same 12 recipes on the 478 prototype fields | § 1.1 |
| `x40_testsplit/x40_testsplit.{csv,json}`, `x40_testsplit/deployed_after_adaptation.json` | candidate recipes before adaptation and the deployed model after adaptation, on the 104 held-out prototype test fields | § 1.2 |
| `x40_threshold/selection.json`, `curves.csv` | operating-point selection on the 43-field prototype validation split → conf 0.18 / NMS IoU 0.50 | § 1.3 |
| `leukocyte_yield/summary.json`, `threshold_sweep.csv`, `detections.csv` | `evaluation/x40_yield.py` on the 104 test fields (predictions, scores, matches; no ground-truth box) | § 1.4 |
| `embedded_campaign/dvfs/`, `embedded_campaign/pinned/` | isolated localizer, PyTorch vs TensorRT FP16 at imgsz 960 (latency and box parity), under frequency scaling and with pinned clocks | § 1.5, § 1.6 |
| `block2_v2_benchmarks/cai409_{trt,torch}/` | 409-patient cAItomorph re-scored on the Jetson, both backends | § 2.1 |
| `block2_v2_benchmarks/x40_stress_*/` | out-of-domain stress test on the Jetson (TensorRT FP16), prototype batches | § 2.3 |
| `block2_v2_benchmarks/timing_x40_test/`, `timing_allidb/`, `energy_x40_test/` | block-2 and end-to-end latency, energy (block-2 device benchmark) | § 2.5, § 2.6 |
| `block2_v2_benchmarks/deployed_latency/`, `deployed_energy/` | the deployed pipeline's own benchmark scripts, 9-field fixture | § 3.1 |
| `cpu_reruns_frozen/` | complete deployed pipeline on CPU (PyTorch), frozen configuration: the prototype batches and four held-out LeukemiaAttri sessions; `fields.txt` lists the input fields of each session, `cpu_reruns_summary.csv` the outcomes | § 2.3 |
| `prototype_acquisition/acquisition_log.csv` | file name, capture time, resolution and triage verdict of the 478 prototype fields (no image) | § 1.4, `docs/DATA.md` |
| `derived/` | values computed from the files above for the article: the 12 recipes on both domains (`domain_transfer_12runs_with_x40_478.csv`), recall by triage verdict (`detections_x40test_by_triage.csv`), fields per session tier (`session_cost_by_tier.json`), observed counts needed for an assertion by the three-way Jeffreys test (`jeffreys_assert_counts.json`), cAItomorph labels by diagnostic group (`caitomorph_labels_by_group_frozen.csv`) | § 1.1, § 1.4, § 2.1 |

Each `*manifest*.json` records the command, the clock state, the image list
and the package versions of its run.
