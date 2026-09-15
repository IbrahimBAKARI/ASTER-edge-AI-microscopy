#!/usr/bin/env bash
# Build the deployed YOLO11n WBC localiser engine ON THE TARGET JETSON.
#
# Source  : models/yolo/wbc_detector.pt   (YOLO11n exp4_domain_aug/nonone, x40 mixed-replay)
# Products: models/yolo/wbc_detector.onnx        (opset 19, static B1, imgsz 960)
#           models/yolo/wbc_detector.engine      (TensorRT FP16, imgsz 960)
#
# Never reuse an engine built on another machine / another TensorRT version.
# Run from the repo root:  scripts/export_yolo_trt.sh
# For pinned-clock latency numbers: `sudo jetson_clocks` before the benchmarks
# (the export itself is clock-insensitive).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
YOLO="${YOLO:-$ROOT/.venv/bin/yolo}"
IMGSZ="${IMGSZ:-960}"
PT="models/yolo/wbc_detector.pt"
ONNX="models/yolo/wbc_detector.onnx"
ENGINE="models/yolo/wbc_detector.engine"

[ -f "$PT" ] || { echo "missing $PT" >&2; exit 1; }

# `yolo export` auto-installs the onnx toolchain; without a constraint it drags
# in onnx>=1.17 -> ml_dtypes>=0.5.4 -> NumPy 2, which breaks the Jetson torch
# 2.8.0 wheel (built against NumPy 1.x). Pin the compatible set first.
echo "== 0/3  pin the onnx export toolchain to a NumPy 1.x set =="
"$PY" -m pip install --quiet "numpy>=1.26,<2" "onnx==1.16.2" "ml-dtypes<0.5.0" "onnxslim>=0.1.48"
"$PY" -c "import numpy,onnx,onnxslim; print('numpy',numpy.__version__,'| onnx',onnx.__version__)"

echo "== 1/3  ONNX export (imgsz $IMGSZ, opset 19, static batch 1, no NMS) =="
"$YOLO" export model="$PT" format=onnx imgsz="$IMGSZ" opset=19 \
    simplify=True batch=1 dynamic=False nms=False device=cpu
# ultralytics writes "${PT%.pt}.onnx", which is already $ONNX here.
[ -f "$ONNX" ] || { echo "ONNX not produced at $ONNX" >&2; exit 1; }

echo "== 2/3  TensorRT FP16 engine (imgsz $IMGSZ) =="
# Ultralytics-native engine export: the plan carries letterbox / imgsz / class
# metadata, so aster_pipeline.YoloWBCDetector loads it exactly like the .pt.
"$YOLO" export model="$PT" format=engine imgsz="$IMGSZ" half=True \
    batch=1 dynamic=False simplify=True device=0 workspace=4
[ -f "$ENGINE" ] || { echo "engine not produced at $ENGINE" >&2; exit 1; }

# the engine step can also auto-install; make sure NumPy is still 1.x for torch
"$PY" -m pip install --quiet "numpy>=1.26,<2" "onnx==1.16.2" "ml-dtypes<0.5.0"
"$PY" -c "import numpy,torch; torch.from_numpy(numpy.zeros((2,2),'float32')); print('torch<->numpy OK, numpy',numpy.__version__)"

echo "== 3/3  checksums =="
sha256sum "$PT" "$ONNX" "$ENGINE" | tee models/yolo/SHA256SUMS.txt

echo
echo "Done. config/inference.yaml already points yolo.tensorrt_weights at $ENGINE."
echo "Sanity check (from the repo root):"
echo "  .venv/bin/python scripts/benchmark_yolo_isolated.py --runs 20 --warmup 3"
