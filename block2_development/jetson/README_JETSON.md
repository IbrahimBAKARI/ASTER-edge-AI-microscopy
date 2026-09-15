# ASTER block 2 — Jetson kit

Target: **Jetson Orin Nano 8 GB Super, JetPack 6.2.x, TensorRT 10.3, Python 3.10** — the
stack pinned in `docs/ENVIRONMENT.md` (repository root).

> **In this repository** the kit is assembled by `make_kit_from_repo.sh` (code from
> `block2_development/src/aster_block2/`, models from `models/mil/`, golden references from
> `golden/expected.json`). The crops of the six golden cAItomorph sessions (`golden/sessions/`)
> and `golden/encoder_parity.npz` are derived from cAItomorph images and are not redistributed. The Mac/Colab steps below describe how the kit was built originally.

## What is in the kit

| Path | Role | Built where |
|---|---|---|
| `models/encoder.onnx` | ResNet18 encoder, static batch 8, input `images` [8,3,224,224], opset 17 | Colab |
| `models/heads.pt` | cell head + attention MIL (+ fp32 encoder weights: reference path and fallback) | Colab |
| `models/thresholds.json` | every resolved [REF]/[FIT] value, full precision | Colab |
| `models/decision_grid.yaml` | frozen grid + Phase-1 amendment | Colab |
| `models/quantifier.json` | cell-level error rates (exact, not the rounded display copy) | Colab |
| `models/ood_stats.npz` | Mahalanobis mean, precision, threshold | Colab |
| `models/config.json` | bag size, seed, classes, run id, ONNX sha256 | Colab |
| `golden/` | 6 cAItomorph sessions + their Colab outputs, 16 crops with fp32 features | Colab |
| `aster_block2/` | the scoring code, byte-identical to the code that ran on Colab | Mac repo |
| `build_engine.sh` | ONNX → TensorRT engine, **on the Jetson** | — |
| `verify_on_jetson.py` | Jetson vs Colab, link by link | — |
| `load_kit.py`, `trt_encoder.py` | `load_scorer(backend="tensorrt" | "torch")` for the integration | — |

## Steps

**On the Mac**

1. In Drive, download `runs/<run_id>/jetson_kit/` (the whole folder).
2. `aster-block2/jetson/assemble_kit.sh ~/Downloads/jetson_kit`
   It checks every Colab file against `COLAB_SHA256SUMS.txt`, adds the code, and writes
   `_work/aster_block2_jetson_kit.tar.gz`.
3. `scp _work/aster_block2_jetson_kit.tar.gz <user>@<jetson>:~/`

**On the Jetson**

```bash
tar xzf aster_block2_jetson_kit.tar.gz && cd aster_block2_jetson_kit
source <repository root>/.venv/bin/activate          # torch 2.8, tensorrt 10.3, numpy 1.26
sha256sum -c SHA256SUMS.txt --quiet && echo "kit intact"
sudo nvpmodel -m 2 && sudo jetson_clocks               # MAXN_SUPER, as for the paper's latencies
./build_engine.sh                                      # ~1 min, writes models/encoder_fp16.engine
python3 verify_on_jetson.py                            # PASS/FAIL, writes verify_report.json
```

Adjust the `source` line to wherever the deployed pipeline's venv lives.

## Reading `verify_on_jetson.py`

| Check | Isolates | Criterion |
|---|---|---|
| bag sampler | numpy version on the Jetson | same 200 indices as Colab |
| crop → tensor | Pillow / torchvision | max diff ≤ 1e-5 |
| encoder PyTorch fp32 | the code and the weights | min cosine ≥ 0.99999 |
| encoder TensorRT | the engine | min cosine ≥ 0.999, cell argmax ≥ 15/16 |
| golden sessions | the whole chain, both backends | same label; counts within 2 % of N_c; raw P_abn within 0.01 |
| latency | — | reported, not judged |

If only the **TensorRT fp16** labels differ, the engine's precision is the cause:

```bash
./build_engine.sh fp32 && python3 verify_on_jetson.py --engine models/encoder_fp32.engine
```

and deploy whichever engine passes (the latency of each is in its `verify_report.json`).
If the **PyTorch** path fails too, the cause is not TensorRT — send `verify_report.json`.

## After a PASS

The kit is verified. In this repository block 2 is already wired into the deployed
pipeline (`aster_pipeline/block2/`, loaded by `aster_pipeline/block2/loader.py`, the
equivalent of `load_kit.load_scorer()`).
