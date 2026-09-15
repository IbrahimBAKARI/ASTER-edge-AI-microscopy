#!/usr/bin/env python3
"""Isolated localiser latency: PyTorch (.pt) vs TensorRT FP16 (.engine).

Same fields, warm-ups excluded, N timed runs, one field at a time (the deployed
batch-1 path). Reports per-field latency (mean/std/median/p95) and the
PyTorch->FP16 speed-up, plus detection parity at the deployed operating point
(matched-box count and mean IoU of matched boxes).

conf / iou / imgsz are read from config/inference.yaml so this never drifts from
the deployed settings. Run on the target Jetson; `sudo jetson_clocks` first for
pinned-clock numbers.

  python scripts/benchmark_yolo_isolated.py --runs 100 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_common import ROOT, clock_state, read_images, sha256, write_json
from aster_pipeline.config import load_config


def summ(xs: list[float]) -> dict:
    a = sorted(map(float, xs))
    n = len(a)
    return {"mean": statistics.fmean(a), "std": statistics.pstdev(a),
            "median": statistics.median(a), "p95": a[min(n - 1, int(0.95 * n))],
            "min": a[0], "max": a[-1], "n": n}


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bench(weights: Path, images: list[Path], runs: int, warmup: int, cfg, device: str) -> tuple[dict, list]:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def one_pass():
        per_img = []
        for im in images:
            r = model.predict(str(im), imgsz=cfg.image_size, conf=cfg.confidence,
                              iou=cfg.iou, classes=[cfg.class_id], device=device,
                              verbose=False)[0]
            per_img.append(sorted(
                ([*map(float, b.xyxy.reshape(-1).tolist()), float(b.conf.reshape(-1)[0])]
                 for b in ([] if r.boxes is None else list(r.boxes))),
                key=lambda x: -x[4]))
        return per_img

    for _ in range(warmup):
        one_pass()
    sync()
    lat, dets = [], None
    for _ in range(runs):
        t = time.perf_counter()
        dets = one_pass()
        sync()
        lat.append((time.perf_counter() - t) * 1000.0 / len(images))
    return summ(lat), dets


def parity(pt_dets, fp16_dets) -> dict:
    matched, iou_sum, pt_total, fp16_total = 0, 0.0, 0, 0
    for a, b in zip(pt_dets, fp16_dets):
        pt_total += len(a)
        fp16_total += len(b)
        used = set()
        for da in a:
            best, bj = 0.0, -1
            for j, db in enumerate(b):
                if j in used:
                    continue
                v = iou(da[:4], db[:4])
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj)
                matched += 1
                iou_sum += best
    return {"pytorch_boxes": pt_total, "fp16_boxes": fp16_total, "matched_boxes": matched,
            "mean_iou_matched": round(iou_sum / matched, 4) if matched else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "benchmark_session.txt"))
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--config", default=str(ROOT / "config/inference.yaml"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--out", default=str(ROOT / "benchmarks/yolo_isolated"))
    args = ap.parse_args()

    cfg = load_config(args.config).yolo
    images = read_images(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "n_images": len(images), "runs": args.runs, "warmup_excluded": args.warmup,
        "device": args.device,
        "operating_point": {"conf": cfg.confidence, "iou_nms": cfg.iou, "imgsz": cfg.image_size},
        "clock_state": clock_state(),
        "weights": {"pytorch": str(cfg.weights), "pytorch_sha256": sha256(cfg.weights)},
    }

    pt_lat, pt_dets = bench(cfg.weights, images, args.runs, args.warmup, cfg, args.device)
    result["latency_ms_per_field"] = {"pytorch": pt_lat}

    engine = cfg.tensorrt_weights
    if engine and engine.is_file() and args.device == "cuda":
        fp16_lat, fp16_dets = bench(engine, images, args.runs, args.warmup, cfg, args.device)
        result["latency_ms_per_field"]["tensorrt_fp16"] = fp16_lat
        result["weights"]["tensorrt_fp16"] = str(engine)
        result["weights"]["tensorrt_fp16_sha256"] = sha256(engine)
        result["speedup_median"] = round(pt_lat["median"] / fp16_lat["median"], 3)
        result["speedup_mean"] = round(pt_lat["mean"] / fp16_lat["mean"], 3)
        result["detection_parity_at_operating_point"] = parity(pt_dets, fp16_dets)
    elif args.device != "cuda":
        result["tensorrt_fp16"] = "skipped: TensorRT needs --device cuda"
    else:
        result["tensorrt_fp16"] = f"engine not found ({engine}); build with scripts/export_yolo_trt.sh"

    # A tiny run is a smoke test, not a measurement -- never let it overwrite the
    # canonical 100-run file.
    name = "yolo_isolated.json" if args.runs >= 10 else "yolo_isolated_smoke.json"
    write_json(out / name, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
