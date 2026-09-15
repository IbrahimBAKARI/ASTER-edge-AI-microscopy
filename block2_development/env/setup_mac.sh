#!/usr/bin/env bash
# Creates the local venv used by the Mac-side scripts. Run once, from aster-block2/.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r env/requirements-mac.txt

echo "ready. Use it with:"
echo "  .venv/bin/python datasets/prepare_corpus.py --dry-run"
