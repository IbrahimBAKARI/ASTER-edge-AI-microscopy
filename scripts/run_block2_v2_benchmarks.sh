#!/usr/bin/env bash
# Block 2 v2 device benchmark campaign -- run this AFTER closing VS Code / freeing RAM.
#
# Every step below writes its output to disk under $OUT before the next step starts, so
# the campaign can be interrupted (memory pressure, a crash, a reboot) and resumed by
# re-running just the remaining steps -- nothing already written is lost or needs to be
# redone. Steps are independent; comment any of them out to skip/resume selectively.
#
# Usage:
#   cd <repository root>
#   HANDOFF=<block-2 device kit folder> \
#   bash scripts/run_block2_v2_benchmarks.sh 2>&1 | tee results/block2_v2_benchmarks/campaign.log
set -euo pipefail

# Paths used on the device (placeholders; adapt to your layout):
#   REPO      this repository
#   HANDOFF   a folder laid out as the block-2 device kit: kit/ (jetson/ scripts + models/),
#             e2e/ (jetson/e2e/ scripts + fields/), data/corpus/ (the prepared corpus,
#             not distributed) and run/.../results/caitomorph_409_predictions.csv
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HANDOFF="${HANDOFF:?set HANDOFF to the block-2 device kit folder}"
PY="${ASTER_PY:-$REPO/.venv/bin/python3}"
OUT="$REPO/results/block2_v2_benchmarks"
mkdir -p "$OUT"

echo "== fixed clocks (record + set) =="
{ sudo nvpmodel -q || true; sudo nvpmodel -m 2; sudo jetson_clocks; sudo jetson_clocks --show; } \
    > "$OUT/00_clocks.log" 2>&1 || true
cat "$OUT/00_clocks.log"

echo
echo "== 1/6: 409-patient cAItomorph corpus, TensorRT backend =="
mkdir -p "$OUT/cai409_trt"
cd "$HANDOFF/e2e"
"$PY" verify_caitomorph_409.py \
    --corpus ../data/corpus --manifest ../data/corpus/manifest.csv \
    --expected ../run/aster_block2_drive/runs/20260911_051357Z/results/caitomorph_409_predictions.csv \
    --kit ../kit --backend tensorrt --output "$OUT/cai409_trt"

echo
echo "== 2/6: 409-patient cAItomorph corpus, PyTorch fp32 reference =="
mkdir -p "$OUT/cai409_torch"
"$PY" verify_caitomorph_409.py \
    --corpus ../data/corpus --manifest ../data/corpus/manifest.csv \
    --expected ../run/aster_block2_drive/runs/20260911_051357Z/results/caitomorph_409_predictions.csv \
    --kit ../kit --backend torch --output "$OUT/cai409_torch"

echo
echo "== 3/6: x40 domain stress test (expect 3/3 out_of_domain) =="
mkdir -p "$OUT/x40_stress"
"$PY" benchmark_block2_e2e.py --mode session --deployed "$REPO" --kit ../kit \
    --fields fields/x40_prototype --crops-per-session 200 --output "$OUT/x40_stress"

echo
echo "== 4/6: latency, x40_prototype (realistic tier-R session, ~614 crops) =="
mkdir -p "$OUT/timing_x40"
"$PY" benchmark_block2_e2e.py --mode timing --deployed "$REPO" --kit ../kit \
    --fields fields/x40_prototype --runs 100 --warmup 5 --output "$OUT/timing_x40"

echo
echo "== 5/6: latency, benchmark_allidb_L2 (deployed fixture, ~95 crops) =="
mkdir -p "$OUT/timing_allidb"
"$PY" benchmark_block2_e2e.py --mode timing --deployed "$REPO" --kit ../kit \
    --fields fields/benchmark_allidb_L2 --runs 100 --warmup 5 --output "$OUT/timing_allidb"

echo
echo "== 6/6: energy, x40_prototype (~6-7 minutes: 60s idle + 300s full analysis) =="
echo "    (this is the long one -- safe to close VS Code now if you haven't; the"
echo "     e2e script and everything before it already wrote their output to disk)"
mkdir -p "$OUT/energy_x40"
"$PY" benchmark_block2_e2e.py --mode energy --deployed "$REPO" --kit ../kit \
    --fields fields/x40_prototype --duration 300 --idle-seconds 60 --output "$OUT/energy_x40"

echo
echo "== 7/7: the deployed pipeline's own benchmark scripts, through the REAL integrated =="
echo "==      pipeline (aster_pipeline.pipeline.LeukemiaPipeline, block2 v2) on the      =="
echo "==      ALL-IDB fixture already used for the published latency/energy numbers      =="
cd "$REPO"
mkdir -p "$OUT/deployed_latency" "$OUT/deployed_energy"
"$PY" scripts/benchmark_current_pipeline.py \
    --images benchmark_session.txt --device cuda --mil-backend tensorrt \
    --output "$OUT/deployed_latency"
"$PY" scripts/benchmark_energy.py \
    --images benchmark_session.txt --device cuda --mil-backend tensorrt \
    --output "$OUT/deployed_energy"

echo
echo "All steps done. Results under: $OUT"
