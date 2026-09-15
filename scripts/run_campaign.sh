#!/usr/bin/env bash
# Replays the whole latency + energy campaign with PINNED clocks.
#
#   cd <REPO>
#   scripts/run_campaign.sh
#
# Asks for the sudo password twice (pin, then restore the clocks). Everything else
# runs as a normal user. About 25 min. Do not use the machine meanwhile (no IDE or
# browser: the full pipeline needs about 4 GB of free unified memory).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M)
OUT=benchmarks/campaign_$STAMP
mkdir -p "$OUT"

echo "############################################################"
echo "#  Pinned-clock campaign  ->  $OUT"
echo "############################################################"

# --- 0. pin the clocks -----------------------------------------------------
echo; echo ">>> [0/4] nvpmodel MAXN_SUPER + jetson_clocks (sudo)"
sudo nvpmodel -m 2 || true
sudo jetson_clocks --store "$OUT/jetsonclocks_prev.conf" 2>/dev/null || true   # for --restore
sudo jetson_clocks
sudo jetson_clocks --show | tee "$OUT/jetson_clocks_state.txt" | grep -Ei 'GPU|EMC|governor' || true
# clock_state() cannot read jetson_clocks --show without root -> give it the dump
export ASTER_CLOCKS_STATE_FILE="$PWD/$OUT/jetson_clocks_state.txt"

restore() {
    echo; echo ">>> restoring the clocks (sudo)"
    sudo jetson_clocks --restore "$OUT/jetsonclocks_prev.conf" 2>/dev/null \
        || sudo jetson_clocks --restore 2>/dev/null \
        || echo "  (--restore unavailable: 'sudo reboot' to return to DVFS)"
}
trap restore EXIT

# --- 1. RAM / CUDA ----------------------------------------------------------
echo; echo ">>> [1/4] RAM + CUDA check"
free -m | awk 'NR==2 {print "  RAM available:", $7, "MB  (target >= 4000)"}'
$PY - <<'EOF'
import torch
x = torch.zeros(int(1e9//4), device="cuda"); torch.cuda.synchronize()
print("  CUDA OK,", round(torch.cuda.mem_get_info()[0]/1e9, 2), "GB free")
EOF

# --- 2. isolated localizer ---------------------------------------------------
echo; echo ">>> [2/4] isolated localizer  PyTorch vs FP16  (100 runs)"
$PY scripts/benchmark_yolo_isolated.py --runs 100 --warmup 5 --out "$OUT/yolo_isolated"

# --- 3. full session ------------------------------------------------------
echo; echo ">>> [3/4] full session  (100 runs, ~4 min)"
$PY scripts/benchmark_current_pipeline.py --runs 100 --warmup 5 --mil-backend tensorrt \
    --output "$OUT/timing"

# --- 4. energy, 3 modes --------------------------------------------------
echo; echo ">>> [4/4] energy idle / acquisition / full_analysis  (300 s x3, ~17 min)"
CAM_OK=$($PY - <<'EOF'
try:
    import cv2; c=cv2.VideoCapture(0); print("1" if c.isOpened() else "0"); c.release()
except Exception:
    print("0")
EOF
)
if [ "$CAM_OK" = "1" ]; then
    $PY scripts/benchmark_energy.py --duration 300 --camera-index 0 --output "$OUT/energy_three_modes"
else
    echo "  (no camera -> --skip-acquisition)"
    $PY scripts/benchmark_energy.py --duration 300 --skip-acquisition --output "$OUT/energy_three_modes"
fi

# --- done --------------------------------------------------------------
echo
echo "############################################################"
echo "#  DONE.  Results in: $OUT"
echo "############################################################"
find "$OUT" -name '*summary*.json' -o -name 'yolo_isolated.json' | sort
echo
echo "Copy these files into results/ (see results/README.md)."
