# REPRODUCE.md — rebuild the results from this repository

Every number of the article, including the data plotted in its figures, comes
from the files in `results/` and `block2_development/training_run_20260911_051357Z/`
and can be checked without any image. Re-running a measurement additionally needs the images, which are not
distributed: public datasets come from their authors, the prototype fields are
available from the authors on reasonable request ([`docs/DATA.md`](docs/DATA.md)).
Put them under `external_data/` (git-ignored) or point the environment
variables below at your copies.

| Variable | Default | Content |
|---|---|---|
| `ASTER_X40_DATASET` | `external_data/prototype_fields_dataset` | prototype fields split into `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml` |
| `ASTER_X40_RAW` | `external_data/prototype_fields_raw` | raw annotated prototype fields with `triage/` |
| — | `external_data/ALL_IDB/L2/` | the ALL-IDB L2 images of `benchmark_session.txt` |

Target for the embedded numbers: NVIDIA Jetson Orin Nano 8 GB (Super),
JetPack 6.2, stack in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). The pipeline
and the tests also run on any Linux or macOS machine (CPU).

---

## 0. Environment

```bash
python3.10 -m venv --system-site-packages .venv    # Jetson: PyQt5 comes from apt (python3-pyqt5)
.venv/bin/pip install -r requirements.txt           # runtime and tests
cd models && shasum -a 256 -c SHA256SUMS.txt && cd ..
```

`requirements.txt` targets the Jetson (JetPack 6, CUDA 12.6: torch and
torchvision come from the Jetson AI Lab index named in its first lines). On
another machine, delete the two `--index-url`/`--extra-index-url` lines and the
PyPI wheels of torch 2.8.0 / torchvision 0.23.0 are used. The exact frozen set
of the Jetson is `docs/requirements-frozen.txt`.

## 1. Run the deployed pipeline

```bash
.venv/bin/python run.py --input <folder of the session fields> \
    --session-id s1 --output results_sessions/s1 --device cpu     # or cuda / auto
.venv/bin/python -m pytest tests/ -q
```

`config/inference.yaml` is the only parameter file; its `workspace_root: ..`
resolves every model path against the repository root. The CPU re-runs of
`results/cpu_reruns_frozen/` were produced this way, one folder per session
(`fields.txt` lists its fields).

## 2. TensorRT engines (on the Jetson only)

```bash
scripts/export_yolo_trt.sh      # models/yolo/wbc_detector.{onnx,engine}  (imgsz 960, FP16)
scripts/build_block2_trt.sh     # models/mil/encoder_fp16.engine          (batch 8, FP16)
```

Engines are never portable across devices. Missing engines → the pipeline logs
a warning and runs the PyTorch path.

## 3. Localizer adaptation (how `models/yolo/wbc_detector.pt` was made)

Source of truth: [`localizer_adaptation/05_mixedreplay_x40_colab.ipynb`](localizer_adaptation/05_mixedreplay_x40_colab.ipynb)
(Colab); recipe and hyper-parameters in [`localizer_adaptation/README.md`](localizer_adaptation/README.md).

```bash
python localizer_adaptation/make_x40_split.py      # prototype split (needs ASTER_X40_RAW)
python localizer_adaptation/eval_on_x40_test.py --weights models/yolo/wbc_detector.pt \
    --tag deployed --device cpu                     # held-out 104 test fields
```

## 4. Operating point and leukocyte yield

```bash
python scripts/select_threshold_x40val.py --weights models/yolo/wbc_detector.pt   # validation split only
.venv/bin/python evaluation/x40_yield.py                                           # -> results/leukocyte_yield/
```

## 5. Block 2

Training, fitting and the pre-registered test: [`block2_development/README.md`](block2_development/README.md).
The rules are frozen; verify the digests first:

```bash
cd block2_development && shasum -a 256 -c results/PREREGISTRATION.sha256 && cd ..
```

Device verification (golden sessions, 409-patient re-scoring, stress test,
latency, energy): `block2_development/jetson/README_JETSON.md` and
`scripts/run_block2_v2_benchmarks.sh`.

## 6. Embedded measurement campaign (Jetson)

[`scripts/HEADLESS_CAMPAIGN.md`](scripts/HEADLESS_CAMPAIGN.md) and
[`scripts/BENCHMARK_PROTOCOL.md`](scripts/BENCHMARK_PROTOCOL.md);
one-shot pinned-clock campaign: `scripts/run_campaign.sh`.
