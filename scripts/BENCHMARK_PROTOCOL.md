# Benchmark protocol — deployed pipeline on the Jetson

The benchmark scripts measure a **technical session assembled from previously
evaluated microscopy fields** (default: `benchmark_session.txt`, 9 ALL-IDB L2
fields). It is a throughput and resource benchmark, not a clinical session and
not a diagnostic-performance study.

## Pipeline identity

`scripts/benchmark_current_pipeline.py` and `scripts/benchmark_energy.py` call
`aster_pipeline.pipeline.LeukemiaPipeline.run(..., benchmark=True)`, the same
method reached by `run.py`, `backend/service.py` and the touch interface
(`Interface/leukemia_ui.py` → `Interface/model_integration.py` →
`backend.ASTERBackend`). There is no copied inference implementation. The method
runs:

- image discovery;
- `YoloWBCDetector.detect` — localization, crop extraction, overlays, crops;
- `build_eval_transform` — crop → tensor;
- `SessionScorer.score` (block 2) — encoder, cell and MIL heads, OOD gate,
  decision grid;
- result and manifest persistence.

Camera acquisition is outside both timing commands (it is one of the three
power modes of `benchmark_energy.py`).

## Input list

A UTF-8 text file with one image path per line; relative paths resolve against
the list file's directory; blank lines and `#` lines are ignored. Use the same
list for timing and energy.

```bash
.venv/bin/python scripts/benchmark_current_pipeline.py --images benchmark_session.txt \
    --warmup 5 --runs 100 --device cuda --mil-backend tensorrt \
    --output benchmarks/current_pipeline/timing
.venv/bin/python scripts/benchmark_energy.py --images benchmark_session.txt \
    --duration 300 --interval-ms 1000 --device cuda --mil-backend tensorrt \
    --output benchmarks/current_pipeline/energy_three_modes
```

`--mil-backend tensorrt --device cuda` makes a campaign fail instead of falling
back to PyTorch.

## Timing boundaries

Instrumentation is enabled only by `benchmark=True`. Named stages:
`session_image_loading_ms`, `yolo_inference_ms`, `wbc_crop_extraction_ms`,
`crop_preprocessing_ms`, `block2_encode_ms`, `block2_heads_ms`,
`block2_grid_ms`, `result_rendering_ms`, `output_persistence_ms`,
`total_analysis_ms`, plus `artifact_flush_ms` (time until every queued artefact
is written). CUDA is synchronised around GPU stages in benchmark mode only.
The summary also reports total minus named stages as orchestration overhead.

## Validation and repetition

One full validation analysis runs first; fewer than `--min-wbc` (default 50)
detected WBCs, or a status other than `completed` or `out_of_domain`, aborts.
`out_of_domain` is accepted because block 2 still executes its whole pass (the
ALL-IDB fixture lies outside the validated acquisition domain).
`insufficient_evidence` is not accepted. Model initialisation is outside the
measured runs; warm-up runs are excluded from `timing_runs.csv`. Inputs are
copied once into a fixed workspace so that import and clean-up never enter the
clocks. Each manifest hashes every source image and model file found.

## Energy

`benchmark_energy.py` refuses to run without `tegrastats`. The three modes
(idle, camera acquisition, full-analysis loop) use the same rail, sampling
interval and duration. Every raw line is time-stamped; the preferred rail is
`VDD_IN`. Energy is integrated trapezoidally over the real time stamps. Reported:
gross energy per analysis, `active_fraction`, `net_power_w = P_full − P_idle`
and `net_energy_per_analysis_j`.

## Outputs

Timing: `timing_runs.csv`, `timing_summary.json`, `benchmark_manifest.json`,
`latency_breakdown.png`, `total_latency_distribution.png`.
Energy: per-mode `tegrastats_raw.log`, `energy_samples.csv`,
`energy_summary.json`, plus `energy_three_modes_summary.json` and
`benchmark_manifest.json`.

## Limits

File-system cache, temperature, DVFS, display load and background services
affect the measurements; keep them constant within a campaign. The energy
window can exceed the requested duration by one final analysis.
