"""Objective sharpness, and the reference interval that decides what is kept.

A sharpness variant serves robustness; past a point it serves confusion. LeukemiaAttr
ships the same physical cells re-acquired at several sharpness levels, and the blurriest
ones carry labels that neither a model nor a human can verify. They are removed - but on
a measured threshold, not on a folder name.

Metric: variance of the Laplacian on the grayscale crop resized to 224 - the block-2
input size. Measuring at the input resolution is the honest comparison: upscaling a 60 px
crop to 224 yields no new detail and scores low, which is exactly the property we want to
detect. Measuring at native size would flatter small crops.

Threshold: NOT tuned on any downstream metric. It is the 1st percentile of the sharpness
distribution of the sources already in the corpus (cAItomorph, AML-LMU, PBC) - the same
reference-interval logic as the [REF] thresholds in PREREGISTRATION.md section 5.3. The
statement it supports is "this crop is blurrier than 99% of the images we already train
on", which is checkable and does not depend on what the classifier prefers.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 224  # the block-2 input size: measure the detail the model actually sees


def sharpness(path: Path) -> float:
    with Image.open(path) as handle:
        grey = handle.convert("L").resize((SIDE, SIDE), Image.BILINEAR)
    a = np.asarray(grey, dtype=np.float64)
    laplacian = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(laplacian.var())


def sample(paths: list[Path], n: int, rng: random.Random) -> list[float]:
    return [sharpness(p) for p in rng.sample(paths, min(n, len(paths)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="_work/corpus",
                        help="reference corpus (cAItomorph / LMU / PBC)")
    parser.add_argument("--crops", default="_work/leukemiaattr_crops")
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--percentile", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reference: dict[str, list[float]] = {}
    corpus = Path(args.corpus)
    for key in ("caitomorph", "aml_lmu", "pbc", "all_idb2"):
        paths = list((corpus / key).rglob("*.webp")) if (corpus / key).is_dir() else []
        if paths:
            reference[key] = sample(paths, args.sample, rng)

    if not reference:
        print("no reference corpus yet - run datasets/prepare_corpus.py first.\n"
              "Falling back to reporting LeukemiaAttr subdomains only.\n")
        threshold = None
    else:
        pooled = [v for values in reference.values() for v in values]
        threshold = float(np.percentile(pooled, args.percentile))
        print(f"{'reference source':<16}{'n':>6}{'p1':>10}{'median':>10}{'p99':>10}")
        for key, values in reference.items():
            print(f"{key:<16}{len(values):>6}{np.percentile(values,1):>10.1f}"
                  f"{np.median(values):>10.1f}{np.percentile(values,99):>10.1f}")
        print(f"\nthreshold = p{args.percentile:g} of the pooled reference = {threshold:.1f}\n")

    crops = Path(args.crops)
    by_subdomain: dict[str, list[Path]] = {}
    manifest = crops / "manifest.csv"
    if manifest.exists():
        for row in csv.DictReader(manifest.open()):
            by_subdomain.setdefault(row["subdomain"], []).append(crops / row["path"])
    else:
        for path in crops.rglob("*.webp"):
            by_subdomain.setdefault(path.name.split("__")[1], []).append(path)

    print(f"{'subdomain':<14}{'n':>7}{'p10':>9}{'median':>9}{'p90':>9}{'kept %':>9}")
    for key in sorted(by_subdomain):
        values = sample(by_subdomain[key], args.sample, rng)
        if not values:
            continue
        kept = (100.0 * sum(v >= threshold for v in values) / len(values)) if threshold else float("nan")
        print(f"{key:<14}{len(by_subdomain[key]):>7}{np.percentile(values,10):>9.1f}"
              f"{np.median(values):>9.1f}{np.percentile(values,90):>9.1f}{kept:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
