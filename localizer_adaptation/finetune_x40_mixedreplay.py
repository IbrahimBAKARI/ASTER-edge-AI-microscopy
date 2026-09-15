#!/usr/bin/env python3
"""Mixed-replay fine-tune of the WBC localiser for the x40 prototype domain.

Goal: keep LLD (source) performance, adapt to the x40 target domain, and stay
robust to staining variation (the current x40 batch is over-stained).

**This mirrors the `m.train(...)` call that actually ran** in
`05_mixedreplay_x40_colab.ipynb` (the deployed x40 localiser was produced there,
on Colab, where LLD lives). That notebook is the executable source of truth; this
CLI is the readable distillation of its training step and takes the mixed
`data.yaml` ready-made.

How the notebook builds that `data.yaml` (cell 5, not reproduced here):
  - LLD side: labels switched to the `nonone` variant (real WBC only).
  - replay subset: `rng(42).sample` of *that base's own* original train list
    (exp2 replays multi-sharpness, exp4 replays multi-domain), size
    `REPLAY_RATIO * len(x40_train)` with `REPLAY_RATIO = 4` -> ~20% x40 per batch.
  - train/val = sorted(replay + x40_split); the yaml `test:` points at x40 val.

Recipe facts (for reproducibility): **no `freeze`** (full fine-tune),
**no `cos_lr`** (linear decay), `patience=10`, `imgsz=960`, `degrees=15`,
`hsv_h=0.5`, everything else at Ultralytics defaults.
Success criteria the notebook checks (cell 9): x40-test mAP50 > 0.396 and
recall >= 0.6, with LLD-test mAP50 >= 0.92 (no regression).

  python training/finetune_x40_mixedreplay.py \
      --data /content/mix/exp4_domain_aug__nonone_mix.yaml \
      --base YOLO11/state/runs/exp4_domain_aug__nonone/weights/best.pt \
      --project runs --name mixedreplay__exp4_domain_aug__nonone
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="mixed-replay data.yaml (LLD replay + x40)")
    ap.add_argument("--base", required=True, help="starting checkpoint (winner = exp4_domain_aug/nonone)")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="mixedreplay__exp4_domain_aug__nonone")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    # exact augmentation block from the notebook (cell 2, dict AUG)
    AUG = dict(hsv_h=0.5, hsv_s=0.7, hsv_v=0.4,
               scale=0.5, degrees=15.0, mosaic=1.0, fliplr=0.5, flipud=0.5)

    model = YOLO(str(args.base), task="detect")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,                 # low: starting from already-good weights
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        workers=8,
        plots=False,
        verbose=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        **AUG,
    )
    print("\nfine-tune done. Evaluate the held-out x40 test split with:")
    print(f"  python training/eval_on_x40_test.py --weights {args.project}/{args.name}/weights/best.pt "
          f"--tag x40_finetuned --device {args.device}")
    print("and check LLD-test did not regress (evaluate on the LLD test split).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
