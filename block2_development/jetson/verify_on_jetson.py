"""Verify block 2 v2 on the Jetson against the Colab run that produced the kit.

    python3 verify_on_jetson.py                                   # fp16 engine
    python3 verify_on_jetson.py --engine models/encoder_fp32.engine
    python3 verify_on_jetson.py --torch-only                      # no TensorRT (code/weights check)

Checks, in order - each one isolates one link of the chain:
  1. environment      python, torch, CUDA, TensorRT, Pillow WebP support
  2. bag sampler      numpy draws the same bag indices as on Colab (bag = MIL input)
  3. preprocessing    the Jetson crop->tensor transform reproduces the Colab tensors
  4. encoder          PyTorch fp32 and TensorRT features vs the Colab fp32 features
  5. golden sessions  full scorer (encoder -> heads -> OOD gate -> grid), both backends,
                      label / counts / P_abn / OOD score vs the Colab outputs
  6. latency          TensorRT session timings
Writes verify_report.json; exit code 0 only if every required check passes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from load_kit import load_config, load_scorer  # noqa: E402
from aster_block2.preprocess import build_eval_transform  # noqa: E402
from aster_block2.sampling import deterministic_bag  # noqa: E402

# Pass criteria. FP16 moves features by ~1e-3; what must not move is the DECISION.
COSINE_FP32 = 0.99999      # Jetson PyTorch fp32 vs Colab fp32: same maths, different GPU
COSINE_TRT = 0.999         # TensorRT fp16 vs Colab fp32
P_ABN_RAW_TOL = 0.01       # raw MIL probability; the calibrated one amplifies it (T < 1)
COUNT_TOL = 0.02           # sum |delta counts| / n_classified


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=KIT / "models/encoder_fp16.engine")
    parser.add_argument("--torch-only", action="store_true")
    args = parser.parse_args()

    config = load_config()
    golden = KIT / "golden"
    expected = json.loads((golden / "expected.json").read_text())
    report, failures = {"config": config}, []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        report.setdefault("checks", []).append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    # 1 -----------------------------------------------------------------------
    print("1. environment")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(), "numpy": np.__version__,
           "pillow_webp": features.check("webp")}
    try:
        import tensorrt as trt
        env["tensorrt"] = trt.__version__
    except ImportError:
        env["tensorrt"] = None
    print("  ", env)
    report["environment"] = env
    check("pillow can read WebP", env["pillow_webp"], "golden crops are lossless WebP")
    use_trt = not args.torch_only
    if use_trt and (env["tensorrt"] is None or not env["cuda"] or not args.engine.exists()):
        check("tensorrt engine available", False,
              f"tensorrt={env['tensorrt']} cuda={env['cuda']} engine={args.engine} - run ./build_engine.sh")
        use_trt = False

    # 2 -----------------------------------------------------------------------
    print("2. bag sampler")
    for session, exp in expected.items():
        bag = deterministic_bag(len(exp["crops"]), config["bag_size"], config["seed"])
        check(f"bag {session}", bag == exp["bag"], f"{len(bag)} indices, numpy {np.__version__}")

    # 3 + 4 -------------------------------------------------------------------
    print("3. preprocessing, 4. encoder")
    parity = np.load(golden / "encoder_parity.npz")
    first = next(iter(expected))
    transform = build_eval_transform(config["image_size"])
    names = expected[first]["crops"][:len(parity["images"])]
    images = torch.stack([transform(Image.open(golden / "sessions" / first / n).convert("RGB")) for n in names])
    diff = float(np.abs(images.numpy() - parity["images"]).max())
    check("crop -> tensor", diff <= 1e-5, f"max |jetson - colab| = {diff:.2e}")
    images = torch.from_numpy(parity["images"])     # from here on, identical inputs

    torch_scorer = load_scorer(backend="torch")
    with torch.inference_mode():
        f32 = torch_scorer.encode(images).numpy()
    cos = float(cosine_rows(f32, parity["features"]).min())
    check("encoder PyTorch fp32", cos >= COSINE_FP32, f"min cosine vs Colab = {cos:.7f}")

    trt_scorer = None
    if use_trt:
        trt_scorer = load_scorer(backend="tensorrt", engine=args.engine)
        with torch.inference_mode():
            f16 = trt_scorer.encode(images).numpy()
            probabilities = trt_scorer.cell_head(torch.from_numpy(f16).to(trt_scorer.device))["probabilities"].cpu().numpy()
        cos = float(cosine_rows(f16, parity["features"]).min())
        check(f"encoder TensorRT ({args.engine.name})", cos >= COSINE_TRT, f"min cosine vs Colab = {cos:.6f}")
        agree = int((probabilities.argmax(1) == parity["probabilities"].argmax(1)).sum())
        check("cell class, TensorRT", agree >= len(images) - 1, f"{agree}/{len(images)} argmax identical")

    # 5 -----------------------------------------------------------------------
    print("5. golden sessions")
    rows = []
    backends = [("torch", torch_scorer)] + ([("tensorrt", trt_scorer)] if trt_scorer else [])
    for session, exp in expected.items():
        paths = [golden / "sessions" / session / n for n in exp["crops"]]
        for backend, scorer in backends:
            start = time.perf_counter()
            result, detail = scorer.score(paths, session_id=session)
            elapsed = (time.perf_counter() - start) * 1000
            n = max(exp["n_classified"], 1)
            count_delta = sum(abs(detail["counts"].get(k, 0) - v) for k, v in exp["counts"].items()) / n
            p_delta = abs(result.uncertainty["p_abn_raw"] - exp["p_abn_raw"])
            row = {"session": session, "backend": backend, "expected": exp["label"], "got": result.label,
                   "tier": result.tier, "count_delta": round(count_delta, 4),
                   "p_abn_raw_delta": round(p_delta, 5),
                   "p_abn": round(result.uncertainty["p_abn"], 4), "p_abn_expected": round(exp["p_abn"], 4),
                   "ood": round(result.uncertainty["ood_score"], 2), "ood_expected": round(exp["ood"], 2),
                   "timing_ms": {k: round(v, 1) for k, v in result.timing_ms.items()},
                   "wall_ms": round(elapsed, 1), "n_crops": len(paths)}
            rows.append(row)
            check(f"{session} [{backend}] label", result.label == exp["label"],
                  f"{result.label} (Colab: {exp['label']})")
            check(f"{session} [{backend}] counts", count_delta <= COUNT_TOL,
                  f"sum|delta| = {count_delta:.2%} of N_c")
            check(f"{session} [{backend}] P_abn", p_delta <= P_ABN_RAW_TOL,
                  f"raw delta {p_delta:.4f}, calibrated {result.uncertainty['p_abn']:.4f} vs {exp['p_abn']:.4f}")
    report["sessions"] = rows

    # 6 -----------------------------------------------------------------------
    if trt_scorer:
        print("6. latency (TensorRT, warm)")
        session, exp = next(iter(expected.items()))
        paths = [golden / "sessions" / session / n for n in exp["crops"]]
        trt_scorer.score(paths)                                  # warm-up
        timings = [trt_scorer.score(paths)[0].timing_ms for _ in range(5)]
        summary = {k: round(float(np.median([t[k] for t in timings])), 1) for k in timings[0]}
        print(f"   {len(paths)} crops: {summary}  (median of 5, image decoding included in encode_ms)")
        report["latency_ms"] = {"n_crops": len(paths), **summary}

    (KIT / "verify_report.json").write_text(json.dumps(report, indent=2, default=str))
    print()
    if failures:
        print(f"FAIL - {len(failures)} check(s): {failures}")
        if any("label" in f and "tensorrt" in f for f in failures) and "fp16" in args.engine.name:
            print("A TensorRT fp16 label differs from Colab. Rebuild in fp32 and verify again:\n"
                  "  ./build_engine.sh fp32 && python3 verify_on_jetson.py --engine models/encoder_fp32.engine")
        return 1
    scope = "PyTorch path only - TensorRT NOT checked" if args.torch_only else "PyTorch and TensorRT"
    print(f"PASS ({scope}) - block 2 on this machine reproduces the Colab run. Report: verify_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
