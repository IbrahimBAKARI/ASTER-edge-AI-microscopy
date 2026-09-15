# End-to-end tests on the Jetson

Run from `e2e/` with the deployed venv active, MAXN_SUPER + `jetson_clocks` for any
timing/energy number. `--deployed` defaults to the repository root (block 1 is
imported from there: YOLO engine, crop contract, config); `--kit` defaults to `../kit`
(`block2_development/jetson/kit`, made by `../make_kit_from_repo.sh`). `fields/`, `../data/corpus`
and the golden crops are not distributed (`docs/DATA.md`).

| Goal | Command | Expected / compare with |
|---|---|---|
| ×40 stress test on the device | `python3 benchmark_block2_e2e.py --mode session --fields fields/x40_prototype --crops-per-session 200 --output out/x40_stress` | 3/3 `out_of_domain` (Colab: OOD 957–1195 vs 576.1; P_abn 0.95–0.97) |
| One realistic session (tier R) | `python3 benchmark_block2_e2e.py --mode session --fields fields/x40_prototype --output out/x40_session` | label + `result_000.json` (v2 schema) |
| Latency, realistic session | `python3 benchmark_block2_e2e.py --mode timing --runs 100 --warmup 5 --fields fields/x40_prototype --output out/timing_x40` | per-stage medians / P95 |
| Latency, deployed fixture | `python3 benchmark_block2_e2e.py --mode timing --runs 100 --warmup 5 --fields fields/benchmark_allidb_L2 --output out/timing_allidb` | deployed pipeline on the same 9 fields |
| Energy + temperature | `python3 benchmark_block2_e2e.py --mode energy --duration 300 --idle-seconds 60 --fields fields/x40_prototype --output out/energy_x40` | W mean/max, J/analysis, GPU/CPU °C max, RAM |
| fp32 reference path | add `--backend torch` | same labels as TensorRT |
| 409 patients re-scored | `python3 verify_caitomorph_409.py --corpus ../data/corpus --manifest ../data/corpus/manifest.csv --expected ../run/aster_block2_drive/runs/20260911_051357Z/results/caitomorph_409_predictions.csv --output out/cai409_trt` | 409/409 labels, AUROC 0.973 [0.937, 0.998] |

Notes
- Crops stay in memory; overlay writing is timed but discarded (persistence is the
  product's I/O, measured by the deployed benchmark).
- The ALL-IDB fixture yields ~95 crops: below tier S (100) → it measures latency only.
  On the Mac CPU it gave `out_of_domain` (OOD 684.7), P_abn 0.996.
- Energy uses the deployed method verbatim (tegrastats, VDD_IN, trapezoidal integration);
  the idle window is measured in the same campaign, so the incremental J/analysis is valid.
- Temperatures are the tegrastats `*@xxC` sensors (GPU, CPU, SoC…), max over the window.
