#!/usr/bin/env python3
"""Select the deployment confidence / NMS-IoU on the x40 VALIDATION split.

The deployed operating point (YOLO ``conf`` and NMS ``iou``) must be chosen on a
split that is never used to report a number. Here that split is the 43-field x40
validation set (``aster_x40_dataset/images/val``); the 104-field test set stays
untouched. This is *not* the independent LLD validation split the mission asked
for -- flag that when reporting.

  ASTER_X40_DATASET=external_data/prototype_fields_dataset \
  python scripts/select_threshold_x40val.py \
      --weights models/yolo/wbc_detector.pt --imgsz 960 --device cpu

The prototype-field dataset is not distributed (available from the authors on
reasonable request); see docs/DATA.md.

Writes results/x40_threshold/{selection.json,curves.csv} under the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the deployed localiser = the mixed-replay exp4_domain_aug/nonone best.pt
DEFAULT_WEIGHTS = ROOT / "models/yolo/wbc_detector.pt"
DEFAULT_DATA = Path(os.environ.get("ASTER_X40_DATASET",
                                   ROOT / "external_data/prototype_fields_dataset")) / "data.yaml"


def sweep(weights: Path, data: Path, imgsz: int, device: str, split: str,
          nms_ious: tuple[float, ...], out: Path) -> tuple[dict, list[dict]]:
    from ultralytics import YOLO
    import numpy as np

    summary: list[dict] = []
    curve: list[dict] = []
    for nms_iou in nms_ious:
        model = YOLO(str(weights), task="detect")
        r = model.val(
            data=str(data), split=split, imgsz=imgsz, conf=0.001, iou=nms_iou,
            classes=[0], device=device, workers=0, plots=False, verbose=False,
            project=str(out / "_ultra"), name=f"{split}_iou{nms_iou:.2f}", exist_ok=True,
        )
        px = np.asarray(r.box.px, dtype=float)                 # confidence grid
        f1 = np.asarray(r.box.f1_curve, dtype=float).mean(0)   # mean F1 over classes
        p = np.asarray(r.box.p_curve, dtype=float).mean(0)
        rc = np.asarray(r.box.r_curve, dtype=float).mean(0)
        best = int(f1.argmax())
        summary.append({
            "nms_iou": nms_iou,
            "best_conf": round(float(px[best]), 4),
            "f1_at_best": round(float(f1[best]), 4),
            "precision_at_best": round(float(p[best]), 4),
            "recall_at_best": round(float(rc[best]), 4),
            "map50": round(float(r.box.map50), 4),
            "map50_95": round(float(r.box.map), 4),
        })
        step = max(1, len(px) // 200)
        for i in range(0, len(px), step):
            curve.append({
                "nms_iou": nms_iou, "conf": round(float(px[i]), 4),
                "precision": round(float(p[i]), 4), "recall": round(float(rc[i]), 4),
                "f1": round(float(f1[i]), 4),
            })
    summary.sort(key=lambda x: -x["f1_at_best"])
    return {"per_nms_iou": summary, "selected": summary[0]}, curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nms-ious", default="0.50,0.60,0.70")
    ap.add_argument("--out", type=Path, default=ROOT / "results/x40_threshold")
    ap.add_argument("--also-test", action="store_true",
                    help="also run the held-out test split at the selected point (sanity)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    nms_ious = tuple(float(x) for x in args.nms_ious.split(","))

    result, curve = sweep(args.weights, args.data, args.imgsz, args.device, "val", nms_ious, args.out)
    result["weights"] = str(args.weights)
    result["imgsz"] = args.imgsz
    result["split_for_selection"] = "val (43 x40 fields)"
    result["caveat"] = ("selected on the x40 validation split, NOT an independent "
                        "LLD validation split; report as such")

    if args.also_test:
        from ultralytics import YOLO
        sel = result["selected"]
        m = YOLO(str(args.weights), task="detect")
        t = m.val(data=str(args.data), split="test", imgsz=args.imgsz,
                  conf=sel["best_conf"], iou=sel["nms_iou"], classes=[0],
                  device=args.device, workers=0, plots=False, verbose=False,
                  project=str(args.out / "_ultra"), name="test_at_selected", exist_ok=True)
        result["test_at_selected_point"] = {
            "conf": sel["best_conf"], "nms_iou": sel["nms_iou"],
            "precision": round(float(t.box.mp), 4), "recall": round(float(t.box.mr), 4),
            "map50": round(float(t.box.map50), 4), "map50_95": round(float(t.box.map), 4),
        }

    (args.out / "selection.json").write_text(json.dumps(result, indent=2))
    with (args.out / "curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["nms_iou", "conf", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(curve)

    sel = result["selected"]
    print(json.dumps(result, indent=2))
    print(f"\nSELECTED  conf={sel['best_conf']}  nms_iou={sel['nms_iou']}  "
          f"(val F1 {sel['f1_at_best']}, P {sel['precision_at_best']}, R {sel['recall_at_best']})")
    print(f"wrote {args.out/'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
