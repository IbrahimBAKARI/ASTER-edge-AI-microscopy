#!/usr/bin/env python3
"""Score models on the held-out x40 TEST split (training/make_x40_split.py output).

Use it for the 'before fine-tune' baseline and, later, for the fine-tuned model,
so before/after are on the exact same held-out fields.

  python training/eval_on_x40_test.py --device cuda
  python training/eval_on_x40_test.py --weights runs/x40_ft/weights/best.pt --tag x40_finetuned
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# training runs of the localizer recipes (not distributed; pass --weights explicitly otherwise)
RUNS = Path(os.environ.get("ASTER_YOLO_RUNS", ROOT / "YOLO11/state/runs"))
# prototype-field dataset: not distributed (available from the authors on reasonable request)
DATA = Path(os.environ.get("ASTER_X40_DATASET", ROOT / "external_data/prototype_fields_dataset")) / "data.yaml"

DEFAULT = {
    "exp2_multisharpness__nonone": RUNS / "exp2_multisharpness__nonone/weights/best.pt",
    "exp4_domain_aug__nonone": RUNS / "exp4_domain_aug__nonone/weights/best.pt",
    "exp4_domain_aug__2class": RUNS / "exp4_domain_aug__2class/weights/best.pt",
    "exp3_multidomain__nonone": RUNS / "exp3_multidomain__nonone/weights/best.pt",
}


def val(tag: str, w: Path, device: str, out: Path) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(w), task="detect").val(
        data=str(DATA), split="test", imgsz=640, batch=1, device=device,
        classes=[0], workers=0, plots=False, verbose=False,
        project=str(out), name=tag, exist_ok=True)
    return {"model": tag, "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4), "map50": round(float(r.box.map50), 4),
            "map50_95": round(float(r.box.map), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="one extra checkpoint (e.g. the fine-tuned model)")
    ap.add_argument("--tag", default="custom")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "results/x40_testsplit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = list(DEFAULT.items())
    if args.weights:
        targets.append((args.tag, Path(args.weights)))

    rows = []
    for tag, w in targets:
        if not Path(w).is_file():
            print(f"skip {tag}: missing {w}")
            continue
        print(f"eval {tag} on x40 test ...", flush=True)
        try:
            rows.append(val(tag, Path(w), args.device, out))
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {str(exc)[:200]}")
            if args.device != "cpu":
                try:
                    rows.append(val(tag, Path(w), "cpu", out))
                except Exception as e2:
                    print(f"  CPU also failed: {e2}")

    rows.sort(key=lambda r: -r["map50"])
    with (out / "x40_testsplit.csv").open("w", newline="") as fh:
        wc = csv.DictWriter(fh, fieldnames=["model", "precision", "recall", "map50", "map50_95"])
        wc.writeheader()
        wc.writerows(rows)
    (out / "x40_testsplit.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'model':28s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for r in rows:
        print(f"{r['model']:28s} {r['precision']:7.3f} {r['recall']:7.3f} "
              f"{r['map50']:7.3f} {r['map50_95']:9.3f}")
    print(f"\nwrote {out/'x40_testsplit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
