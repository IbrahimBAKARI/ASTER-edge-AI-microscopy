#!/usr/bin/env bash
# Build the block-2 encoder engine ON THE TARGET JETSON (a TensorRT engine is tied to the
# GPU and the TensorRT version that built it - never copy one from another machine).
#
#   ./build_engine.sh          -> models/encoder_fp16.engine   (deployment default)
#   ./build_engine.sh fp32     -> models/encoder_fp32.engine   (fallback if fp16 fails verification)
#
# Source: models/encoder.onnx, static batch 8, input 'images' [8,3,224,224] - the same
# convention as the deployed max_fold*_encoder engines (scripts/export_mil_trt.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PRECISION="${1:-fp16}"
case "$PRECISION" in fp16) FLAG="--fp16" ;; fp32) FLAG="" ;;
  *) echo "usage: $0 [fp16|fp32]" >&2; exit 2 ;; esac

ONNX="models/encoder.onnx"
ENGINE="models/encoder_${PRECISION}.engine"
CACHE="models/timing_${PRECISION}.cache"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec || true)"
[ -n "$TRTEXEC" ] || { echo "trtexec not found (set TRTEXEC=...)" >&2; exit 1; }

# The ONNX must be the one this run exported - checked against config.json before building.
python3 - "$ONNX" <<'PY'
import hashlib, json, sys, pathlib
onnx = pathlib.Path(sys.argv[1])
expected = json.loads(pathlib.Path("models/config.json").read_text())["onnx_sha256"]
actual = hashlib.sha256(onnx.read_bytes()).hexdigest()
if actual != expected:
    sys.exit(f"{onnx} sha256 {actual[:12]} differs from config.json {expected[:12]} - kit corrupted")
print(f"ONNX verified ({actual[:12]})")
PY

echo "== $ONNX -> $ENGINE ($PRECISION) =="
# Same settings as the deployed build: workspace 512 MiB, timing cache, no timing inference.
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" $FLAG \
    --memPoolSize=workspace:512 --timingCacheFile="$CACHE" --skipInference

python3 - "$ENGINE" "$PRECISION" <<'PY'
import hashlib, json, pathlib, sys
import tensorrt as trt
engine, precision = pathlib.Path(sys.argv[1]), sys.argv[2]
release = pathlib.Path("/etc/nv_tegra_release")
meta = {"engine": engine.name, "precision": precision, "static_batch": 8,
        "input": "images", "input_shape": [8, 3, 224, 224],
        "tensorrt": trt.__version__,
        "l4t": release.read_text().splitlines()[0] if release.exists() else None,
        "sha256": hashlib.sha256(engine.read_bytes()).hexdigest()}
engine.with_suffix(".engine.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY
echo
echo "Next: python3 verify_on_jetson.py --engine $ENGINE"
