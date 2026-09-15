#!/usr/bin/env bash
# Mac side: turn the `jetson_kit/` folder downloaded from the Drive run directory into the
# self-contained kit copied to the Jetson (models + golden set from Colab, code from here).
#
#   jetson/assemble_kit.sh ~/Downloads/jetson_kit
#   -> _work/aster_block2_jetson_kit/  and  _work/aster_block2_jetson_kit.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SRC="$(cd "${1:?usage: $0 <downloaded jetson_kit folder>}" && pwd)"
OUT="$REPO/_work/aster_block2_jetson_kit"

echo "== verifying the Colab files =="
(cd "$SRC" && shasum -a 256 -c COLAB_SHA256SUMS.txt --quiet) && echo "all Colab files intact"

rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/aster_block2"
cp -R "$SRC/models" "$SRC/golden" "$OUT/"
cp "$SRC/COLAB_SHA256SUMS.txt" "$OUT/"
# the package exactly as it ran on Colab (runlog is Colab-only, session_calibration unused)
for f in __init__.py calibration.py grid.py inference.py models.py ood.py preprocess.py \
         proportion_test.py quantify.py sampling.py schema.py crops.py decision_grid.yaml; do
    cp "$REPO/src/aster_block2/$f" "$OUT/aster_block2/"
done
cp "$HERE"/{README_JETSON.md,build_engine.sh,verify_on_jetson.py,load_kit.py,trt_encoder.py} "$OUT/"
chmod +x "$OUT/build_engine.sh"

(cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.txt)
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "kit: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "tar: $OUT.tar.gz ($(du -sh "$OUT.tar.gz" | cut -f1))"
echo "next: scp \"$OUT.tar.gz\" <user>@<jetson>:~/   then follow README_JETSON.md"
