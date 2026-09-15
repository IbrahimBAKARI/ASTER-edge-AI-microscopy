#!/usr/bin/env python3
"""Build a slide-adjacency-safe train/val/test split of the annotated x40
prototype fields, for mixed-replay fine-tuning of the WBC localiser.

Split unit = an acquisition *cluster* (fields captured < GAP seconds apart, i.e.
spatially adjacent on the slide). Whole clusters go to one split, so no field is
ever a few seconds away from a field in another split. Clusters are shuffled with
a fixed seed and packed into test (~20%), val (~15%), train (rest).

  train / val : GARDER + A_REVOIR only  (clean material to learn from)
  test        : ALL verdicts, incl. REJET  (representative of deployment)

Output (real file copies, ready to zip for Colab):
  <out>/images/{train,val,test}/*.jpg
  <out>/labels/{train,val,test}/*.txt
  <out>/data.yaml
  <out>/data_mixedreplay_TEMPLATE.yaml
  <out>/SPLIT.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import shutil
from collections import Counter
from pathlib import Path

# Raw annotated prototype fields and the split dataset: not distributed (available from
# the authors on reasonable request, see docs/DATA.md). Point the variables at your copy.
_ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("ASTER_X40_RAW", _ROOT / "external_data/prototype_fields_raw")) / "triage"
OUT_DEFAULT = Path(os.environ.get("ASTER_X40_DATASET", _ROOT / "external_data/prototype_fields_dataset"))


def collect() -> list[dict]:
    tri = {r["file"]: r for r in csv.DictReader((SRC / "_triage.csv").open())}
    rows = []
    for jpg in sorted(SRC.rglob("slide_*.jpg")):
        if jpg.name.startswith("._"):
            continue
        txt = jpg.with_suffix(".txt")
        if not txt.is_file():
            continue
        ts = dt.datetime.strptime(jpg.stem, "slide_%Y%m%d_%H%M%S_%f")
        n_box = sum(1 for ln in txt.read_text().splitlines() if ln.strip())
        rows.append({"jpg": jpg, "txt": txt, "name": jpg.name, "ts": ts,
                     "verdict": tri.get(jpg.name, {}).get("verdict", "?"), "n_box": n_box})
    rows.sort(key=lambda r: r["ts"])
    return rows


def cluster(rows: list[dict], gap_s: float, max_size: int = 30) -> list[list[dict]]:
    clusters, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if (r["ts"] - prev["ts"]).total_seconds() > gap_s:
            clusters.append(cur)
            cur = []
        cur.append(r)
    clusters.append(cur)
    # recursively split any oversized cluster at its widest internal gap
    out = []
    stack = list(clusters)
    while stack:
        c = stack.pop()
        if len(c) <= max_size:
            out.append(c)
            continue
        gaps = [((c[i + 1]["ts"] - c[i]["ts"]).total_seconds(), i) for i in range(len(c) - 1)]
        _, k = max(gaps)
        stack.append(c[:k + 1])
        stack.append(c[k + 1:])
    out.sort(key=lambda c: c[0]["ts"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--gap", type=float, default=90.0, help="seconds; cluster boundary")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copy")
    args = ap.parse_args()

    rows = collect()
    clusters = cluster(rows, args.gap)
    total = len(rows)
    rng = random.Random(args.seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)

    want_test = args.test_frac * total
    want_val = args.val_frac * total
    assign: dict[int, str] = {}
    n_test = n_val = 0
    for ci in order:
        size = len(clusters[ci])
        if n_test < want_test:
            assign[ci] = "test"; n_test += size
        elif n_val < want_val:
            assign[ci] = "val"; n_val += size
        else:
            assign[ci] = "train"

    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = Counter()
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ci, cl in enumerate(clusters):
        split = assign[ci]
        for r in cl:
            if split in ("train", "val") and r["verdict"] == "REJET":
                dropped[split] += 1
                continue
            dst_i = out / "images" / split / r["name"]
            dst_l = out / "labels" / split / r["txt"].name
            for src, dst in ((r["jpg"], dst_i), (r["txt"], dst_l)):
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.symlink:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            kept[split] += 1
            split_rows[split].append(r | {"cluster": ci})

    (out / "data.yaml").write_text(
        f"# x40 prototype WBC localisation - slide-adjacency-safe split "
        f"(gap {args.gap}s, seed {args.seed})\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: wbc\n", encoding="utf-8")

    (out / "data_mixedreplay_TEMPLATE.yaml").write_text(
        "# MIXED-REPLAY training set: LLD source domain + this x40 target domain.\n"
        "# Fill the LLD paths on Colab (they live under /content/lld_cloud/...).\n"
        "# Ultralytics accepts a list of dirs for train/val.\n"
        "path: /content\n"
        "train:\n"
        "  - lld_cloud/<lld_train_images_dir>        # LLD train (keeps source perf + good staining)\n"
        f"  - {out}/images/train                      # x40 target (this split)\n"
        "val:\n"
        "  - lld_cloud/<lld_val_images_dir>          # watch LLD does not regress\n"
        f"  - {out}/images/val\n"
        f"test:\n"
        f"  - {out}/images/test                       # held-out x40 - the deployment number\n"
        "names:\n  0: wbc\n", encoding="utf-8")

    lines = ["# x40 prototype split for mixed-replay fine-tuning\n",
             f"- source: {SRC}",
             f"- {total} annotated fields, {sum(r['n_box'] for r in rows)} WBC boxes",
             f"- split unit: acquisition cluster (gap > {args.gap}s = spatially separate)",
             f"- {len(clusters)} clusters, shuffled seed {args.seed}, packed test~{args.test_frac:.0%} / val~{args.val_frac:.0%} / train rest",
             f"- train/val keep GARDER+A_REVOIR only; test keeps ALL verdicts\n",
             "| split | fields | boxes | GARDER | A_REVOIR | REJET | REJET dropped |",
             "|---|---|---|---|---|---|---|"]
    for split in ("train", "val", "test"):
        sr = split_rows[split]
        vc = Counter(r["verdict"] for r in sr)
        lines.append(f"| {split} | {len(sr)} | {sum(r['n_box'] for r in sr)} | "
                     f"{vc['GARDER']} | {vc['A_REVOIR']} | {vc['REJET']} | {dropped[split]} |")
    lines.append("\n## clusters per split")
    for split in ("train", "val", "test"):
        cs = sorted({r["cluster"] for r in split_rows[split]})
        spans = []
        for ci in cs:
            cl = clusters[ci]
            spans.append(f"C{ci}[{cl[0]['ts']:%H:%M}-{cl[-1]['ts']:%H:%M},{len(cl)}]")
        lines.append(f"- **{split}**: {' '.join(spans)}")
    (out / "SPLIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
