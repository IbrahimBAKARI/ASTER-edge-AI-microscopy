# Run 20260911_051357Z — what feeds what

**7 alert(s)** — read `ALERTS.md` first.

Started 2026-09-11T05:13:57.311533+00:00, ended 2026-09-11T16:20:59.442457+00:00.
Failed sections: none.

| Artifact | Feeds |
|---|---|
| `results/cell_head_per_class.csv` | per-class precision/recall — paper, cell-head table |
| `results/differential_spearman.json` | differential validated vs AML-MLL — §6.2, and falsifier 6.5 |
| `results/caitomorph_409_predictions.csv` | the primary test, one row per patient |
| `results/caitomorph_409_confusion.csv` | confusion matrix by `diagnosis_fine` — main results table |
| `results/caitomorph_tierR_confusion.csv` | secondary analysis at reference count — §3.4 |
| `results/x40_stress.csv` | ×40 stress test — the figure that justifies the redesign |
| `results/reference_intervals.json` | each `[REF]` threshold: floor, percentile, which was adopted |
| `results/resolved_thresholds.json` | every `[FIT]` value and the split it was fitted on |
| `results/sensitivity*.csv` | the `[OPS]` sensitivity analyses |
| `metrics/*.jsonl` | training curves |
| `logs/*.log`, `logs/*.traceback.txt` | what actually happened, including failures |
| `env/`, `manifest.json` | reproducibility: versions, GPU, seeds, pre-registration digests |
| `bundle/` + `bundle/SHA256SUMS.txt` | **integration/PATCH.md** — what goes on the Jetson |

Download runs/<run_id>/bundle/ and follow integration/PATCH.md. Build the TensorRT engine ON THE JETSON - engines are not portable. REPORT_INPUTS.md maps every artifact to the paper section it feeds.
