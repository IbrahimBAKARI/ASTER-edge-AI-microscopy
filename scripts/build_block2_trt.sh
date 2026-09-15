#!/usr/bin/env bash
# Build the block-2 encoder engine ON THE TARGET JETSON (a TensorRT engine is tied to
# the GPU and the TensorRT version that built it -- never copy one from another
# machine). Adapted from block2_development/jetson/build_engine.sh for this repo's
# models/mil/ layout; run from the repo root.
#
#   scripts/build_block2_trt.sh          -> models/mil/encoder_fp16.engine   (deployed)
#   scripts/build_block2_trt.sh fp32     -> models/mil/encoder_fp32.engine   (fallback if
#                                            fp16 fails aster_pipeline/block2's parity)
#
# Source: models/mil/encoder.onnx, static batch 8, input 'images' [8,3,224,224] --
# the same convention as the deployed YOLO engine (scripts/export_yolo_trt.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/models/mil"

PRECISION="${1:-fp16}"
case "$PRECISION" in fp16) FLAG="--fp16" ;; fp32) FLAG="" ;;
  *) echo "usage: $0 [fp16|fp32]" >&2; exit 2 ;; esac

ONNX="encoder.onnx"
ENGINE="encoder_${PRECISION}.engine"
CACHE="timing_${PRECISION}.cache"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec || true)"
[ -n "$TRTEXEC" ] || { echo "trtexec not found (set TRTEXEC=...)" >&2; exit 1; }
[ -f "$ONNX" ] || { echo "missing $ROOT/models/mil/$ONNX" >&2; exit 1; }

# The ONNX must be the one config.json declares -- checked before building.
python3 - "$ONNX" <<'PY'
import hashlib, json, sys, pathlib
onnx = pathlib.Path(sys.argv[1])
expected = json.loads(pathlib.Path("config.json").read_text())["onnx_sha256"]
actual = hashlib.sha256(onnx.read_bytes()).hexdigest()
if actual != expected:
    sys.exit(f"{onnx} sha256 {actual[:12]} differs from config.json {expected[:12]} - bundle corrupted")
print(f"ONNX verified ({actual[:12]})")
PY

echo "== $ONNX -> $ENGINE ($PRECISION) =="
# Same settings as block2_development/jetson/build_engine.sh: workspace 512 MiB, timing cache, no
# timing inference during the build.
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
echo "Done. config/inference.yaml already points block2.encoder at models/mil/$ONNX;"
echo "aster_pipeline.config.Block2Config.encoder_engine derives models/mil/$ENGINE next to it."
echo "Sanity check: run.py on any session, or block2_development/jetson/verify_on_jetson.py"
echo "(kit from block2_development/jetson/make_kit_from_repo.sh) against this engine."
