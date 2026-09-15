# Environments

The training baseline is Python 3.11 with PyTorch 2.5.1 / torchvision 0.20.1. The exact
versions are deliberately pinned. Every notebook must save `python --version`,
`nvidia-smi`, and `pip freeze` beside its run manifest before doing work.

Local CPU validation:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r env/requirements-cpu.txt
```

Colab:

```bash
pip install -r env/requirements-colab.txt
```

**The Colab lock is deliberately loose.** The image ships its own Python, torch, numpy and
CUDA and they move (2026-09: Python 3.14, torch 2.11+cu128). Pinning against it makes pip
refuse the entire requirement set, and forcing numpy or torch onto the image breaks CUDA.
Only what the image lacks is installed; the versions actually used are recorded per run in
`runs/<id>/env/pip_freeze.txt` and `manifest.json` (here: `training_run_20260911_051357Z/`).

If Colab's CUDA image cannot install this lock unchanged, create a new dated lock and
record the change in the run manifest; do not silently relax versions.

