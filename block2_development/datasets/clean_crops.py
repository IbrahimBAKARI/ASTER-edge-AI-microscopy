"""Materialise the clean LeukemiaAttr crop set: a filter on the manifest, not a re-cut.

Threshold: `--min-sharpness`, measured by datasets/sharpness.py (variance of the Laplacian
at the 224 block-2 input size) and stored per crop by datasets/annotate_sharpness.py.

4.0 was chosen by looking at crops binned by sharpness (`_work/bandes_nettete.png`): below
4 the cell is still recognisable as a cell, but the fine chromatin texture is gone - and
that texture is exactly what separates a lymphoblast from a lymphocyte, or an abnormal
promyelocyte from a normal one. A label that neither a model nor a human can verify is
noise, not supervision.

The cell_uid groups the acquisitions of one physical cell. The field filename encodes the
magnification (100/400/1000), so it is stripped: without that, the same cell imaged at two
magnifications counts as two cells.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

MAG_TOKENS = {"100", "400", "1000"}


def cell_uid(row: dict) -> str:
    stem = row["field_image"].rsplit(".", 1)[0]
    parts = [p for p in stem.split("_") if p not in MAG_TOKENS]
    return f'{"_".join(parts)}/{row["box_index"]}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", default="_work/leukemiaattr_crops")
    parser.add_argument("--out", default="_work/leukemiaattr_clean")
    parser.add_argument("--min-sharpness", type=float, default=4.0)
    args = parser.parse_args()

    source = Path(args.crops)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    rows = list(csv.DictReader((source / "manifest.csv").open()))
    if "sharpness" not in rows[0]:
        raise SystemExit("run datasets/annotate_sharpness.py first")
    kept = [r for r in rows if float(r["sharpness"]) >= args.min_sharpness]

    fields = list(rows[0].keys()) + ["cell_uid_nomag"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in kept:
            target = out / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row["path"], target)
            row["cell_uid_nomag"] = cell_uid(row)
            writer.writerow(row)

    crops = Counter((r["split"], r["cell_class"]) for r in kept)
    cells = defaultdict(set)
    for r in kept:
        cells[(r["split"], r["cell_class"])].add(cell_uid(r))
    classes = sorted({c for _, c in crops})
    patients = defaultdict(set)
    for r in kept:
        patients[(r["diagnosis"], r["split"])].add(r["patient"])
    subdomains = Counter(r["subdomain"] for r in kept)

    lines = [
        "# LeukemiaAttr — clean crop set",
        "",
        f"Sharpness threshold: **>= {args.min_sharpness}** "
        f"({len(kept)} of {len(rows)} crops, {len(kept)/len(rows)*100:.1f} %).",
        "",
        "Cut with the deployed block-1 convention (`aster_block2.crops`: box +10 %/side,",
        "clipped, native pixels, degenerate boxes dropped). Splits are patient-disjoint and",
        "enforced globally: any patient seen in test in any subdomain is in test everywhere.",
        "",
        "| class | crops train | crops test | cells train | cells test |",
        "|---|---|---|---|---|",
    ]
    for c in classes:
        lines.append(f"| `{c}` | {crops[('train', c)]} | {crops[('test', c)]} | "
                     f"{len(cells[('train', c)])} | {len(cells[('test', c)])} |")
    lines += [
        f"| **total** | **{sum(v for (s, _), v in crops.items() if s == 'train')}** | "
        f"**{sum(v for (s, _), v in crops.items() if s == 'test')}** | "
        f"**{len(set().union(*[cells[('train', c)] for c in classes]))}** | "
        f"**{len(set().union(*[cells[('test', c)] for c in classes]))}** |",
        "",
        "## Patients",
        "",
        "| diagnosis | train | test |",
        "|---|---|---|",
    ]
    for diagnosis in sorted({d for d, _ in patients}):
        lines.append(f"| {diagnosis} | {len(patients[(diagnosis, 'train')])} | "
                     f"{len(patients[(diagnosis, 'test')])} |")
    lines += [
        "",
        "## Surviving subdomains",
        "",
        "| subdomain | crops |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in subdomains.most_common()] + [
        "",
        "## Two limits to carry into the paper",
        "",
        "1. **APML and CLL have one patient on each side.** `promyelocyte_abnormal` and",
        "   `lymphocyte_atypical` therefore have as many test cells as training cells, all",
        "   from a single patient per split. These cells are for TRAINING. No test metric",
        "   for the APL flag may be reported from this source - that test stays cAItomorph",
        "   and the AML-MLL PML_RARA patients.",
        "2. **Annotation is incomplete**: ~4.3 cells are annotated per 640x640 field while",
        "   more leukocytes are visible. Usable for labelled crops, not for measuring",
        "   localizer recall.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:3] + lines[7:]))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
