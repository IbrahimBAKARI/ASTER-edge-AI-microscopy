#!/usr/bin/env bash
# Assemble the block-2 device kit from this repository (instead of from a Colab download):
#   block2_development/jetson/make_kit_from_repo.sh [output folder]   (default: block2_development/jetson/kit)
# Layout expected by verify_on_jetson.py, load_kit.py and e2e/*.py:
#   kit/{aster_block2/, models/, golden/, build_engine.sh, verify_on_jetson.py, load_kit.py, trt_encoder.py}
# golden/sessions/ (the crops of the six golden cAItomorph sessions) and golden/encoder_parity.npz
# (16 of those crops as tensors, with their Colab features) are derived from the cAItomorph
# images and are not redistributed; golden/expected.json lists the crops and the expected outputs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV="$(dirname "$HERE")"
ROOT="$(dirname "$DEV")"
OUT="${1:-$HERE/kit}"

mkdir -p "$OUT/aster_block2" "$OUT/models" "$OUT/golden"
for f in __init__.py calibration.py grid.py inference.py models.py ood.py preprocess.py \
         proportion_test.py quantify.py sampling.py schema.py crops.py decision_grid.yaml; do
    cp "$DEV/src/aster_block2/$f" "$OUT/aster_block2/"
done
for f in encoder.onnx heads.pt thresholds.json decision_grid.yaml quantifier.json ood_stats.npz config.json; do
    cp "$ROOT/models/mil/$f" "$OUT/models/"
done
cp "$HERE/golden/expected.json" "$OUT/golden/"
cp "$HERE"/{README_JETSON.md,build_engine.sh,verify_on_jetson.py,load_kit.py,trt_encoder.py} "$OUT/"
chmod +x "$OUT/build_engine.sh"
(cd "$ROOT/models" && shasum -a 256 -c SHA256SUMS.txt --quiet) && echo "models verified against models/SHA256SUMS.txt"
echo "kit: $OUT"
[ -d "$OUT/golden/sessions" ] || echo "note: golden/sessions/ and golden/encoder_parity.npz absent -> verify_on_jetson.py needs them (cAItomorph-derived)"
