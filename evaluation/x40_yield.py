#!/usr/bin/env python3
"""Leukocyte yield on the held-out x40 test split.

Runs the deployed localiser on the 104 x40 test fields, dumps every prediction
with its confidence + greedy IoU>=0.5 match to a ground-truth box, then sweeps
the confidence threshold and reports per threshold: precision, recall, F1, WBC
per field, and the number of fields needed to reach 50 WBC at the mean yield.

The .pt is scored on CPU by default -- the .pt<->FP16 engine parity is already
established (93/95 boxes matched, mean IoU 0.88), so the yield curve is the same. Pass --weights <engine> --device cuda to redo it
on the exact deployed plan.

  python evaluation/x40_yield.py                 # mixed-replay .pt, CPU, imgsz 960
  python evaluation/x40_yield.py --weights models/yolo/wbc_detector.engine --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X40 = Path(os.environ.get("ASTER_X40_DATASET",
          ROOT / "external_data/prototype_fields_dataset"))  # not distributed; set ASTER_X40_DATASET to your copy
DEFAULT_W = ROOT / "models/yolo/wbc_detector.pt"  # the deployed x40 mixed-replay localiser


def load_gt(stem, w, h):
    p = X40 / "labels/test" / f"{stem}.txt"
    out = []
    if p.is_file():
        for line in p.read_text().split("\n"):
            f = line.split()
            if len(f) >= 5:
                cx, cy, bw, bh = (float(x) for x in f[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--iou-nms", type=float, default=0.50)
    ap.add_argument("--iou-match", type=float, default=0.50)
    ap.add_argument("--out", default=str(ROOT / "results/leukocyte_yield"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    images = sorted((X40 / "images/test").glob("*.jpg"))

    dets, per_field_gt = [], {}
    for img_path in images:
        stem = img_path.stem
        w, h = Image.open(img_path).size
        gt = load_gt(stem, w, h)
        per_field_gt[stem] = len(gt)
        r = model.predict(str(img_path), imgsz=args.imgsz, conf=0.01, iou=args.iou_nms,
                          classes=[0], device=args.device, verbose=False)[0]
        preds = sorted(
            ((float(b.conf.reshape(-1)[0]), tuple(float(v) for v in b.xyxy.reshape(-1).tolist()))
             for b in ([] if r.boxes is None else list(r.boxes))),
            key=lambda x: -x[0])
        used = set()
        for score, box in preds:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gt):
                if j in used:
                    continue
                v = iou(box, g)
                if v > best_iou:
                    best_iou, best_j = v, j
            tp = best_iou >= args.iou_match
            if tp:
                used.add(best_j)
            dets.append({"field": stem, "score": score, "tp": int(tp), "iou": round(best_iou, 3)})

    total_gt = sum(per_field_gt.values())
    n_fields = len(images)
    with (out / "detections.csv").open("w", newline="") as fh:
        wc = csv.DictWriter(fh, fieldnames=["field", "score", "tp", "iou"])
        wc.writeheader(); wc.writerows(dets)

    rows = []
    for thr in [round(x, 2) for x in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)]:
        kept = [d for d in dets if d["score"] >= thr]
        tp = sum(d["tp"] for d in kept)
        prec = tp / len(kept) if kept else 0.0
        rec = tp / total_gt if total_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_field = {s: 0 for s in per_field_gt}
        for d in kept:
            per_field[d["field"]] += 1
        counts = list(per_field.values())
        mean_y = statistics.fmean(counts)
        rows.append({
            "threshold": thr, "detections": len(kept), "tp": tp, "fp": len(kept) - tp,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "wbc_per_field_mean": round(mean_y, 3),
            "wbc_per_field_sd": round(statistics.pstdev(counts), 3),
            "wbc_per_field_median": statistics.median(counts),
            "fields_zero_wbc": sum(1 for c in counts if c == 0),
            "fields_for_50_wbc_at_mean": round(50 / mean_y, 1) if mean_y > 0 else None,
        })
    with (out / "threshold_sweep.csv").open("w", newline="") as fh:
        wc = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wc.writeheader(); wc.writerows(rows)

    summary = {
        "weights": args.weights, "device": args.device, "imgsz": args.imgsz,
        "split": "x40 test (held out)", "n_fields": n_fields, "total_gt_boxes": total_gt,
        "gt_wbc_per_field_mean": round(total_gt / n_fields, 3),
        "operating_point_conf_0.18": next(r for r in rows if r["threshold"] == 0.18),
        "sweep": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["operating_point_conf_0.18"], indent=2))
    print(f"\nwrote {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
