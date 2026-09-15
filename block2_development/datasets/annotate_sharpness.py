"""Add a measured `sharpness` column to the LeukemiaAttr crop manifest.

Cleaning is then a filter on the manifest, not a re-extraction: the threshold can be moved
without recutting a single crop. The metric is defined in datasets/sharpness.py (variance
of the Laplacian at the 224 block-2 input size).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharpness import sharpness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="_work/leukemiaattr_crops")
    args = parser.parse_args()

    root = Path(args.crops)
    source = root / "manifest.csv"
    rows = list(csv.DictReader(source.open()))
    fields = list(rows[0].keys())
    if "sharpness" not in fields:
        fields.append("sharpness")

    target = root / "manifest_sharpness.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            row["sharpness"] = f"{sharpness(root / row['path']):.3f}"
            writer.writerow(row)
            if index % 10000 == 0:
                print(f"  {index}/{len(rows)}", flush=True)
    source.rename(root / "manifest_raw.csv")
    target.rename(source)
    print(f"done: {len(rows)} rows, sharpness column added to {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
