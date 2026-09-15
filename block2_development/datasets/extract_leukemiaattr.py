"""LeukemiaAttr fields -> labelled single-cell crops, patient-disjoint train/test.

Why this script exists
----------------------
LeukemiaAttr ships 640x640 microscopy FIELDS with COCO boxes, not crops. Block 2 consumes
single-cell crops only, so the boxes must be cut with the deployed block-1 convention
before anything else happens (`aster_block2.crops`, byte-identical to
`aster_pipeline/crop_extraction.py`).

Three things this script is careful about
-----------------------------------------
1. **Patient-disjoint splits, enforced globally.** The shipped train/test split is
   patient-disjoint *within* a subdomain but NOT across them: one patient appears in train
   in one subdomain and in test in another. Any patient seen in test anywhere is assigned
   to test everywhere, so no acquisition of a test patient's cell can reach train.
2. **The 10 subdomains are re-acquisitions of the SAME physical cells** (verified in
   LLD_work: median box-size ratio 0.98 across 10X/40X/100X - they differ in sharpness and
   camera, not scale). They are legitimate augmentation, never independent samples: the
   manifest carries `cell_uid = patient/field/box` so a caller can group by physical cell.
3. **Quality filter**, recorded not silent: degenerate boxes, crops under `--min-side`, and
   near-blank crops (a few boxes land on smear streaks) are dropped and counted.

Usage
-----
    python datasets/extract_leukemiaattr.py --out _work/leukemiaattr_crops
    python datasets/extract_leukemiaattr.py --out /tmp/probe --subdomains H_100X_C1 --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageStat  # noqa: E402

from aster_block2.crops import coco_to_xyxy, expand_and_clip_box, extract_crop  # noqa: E402

# The dataset ships with an extra nesting level: <base>/LeukemiaAttr/<subdomain>.
# Pass either path; resolve_root() descends when needed.
DEFAULT_ROOT = Path(os.environ.get("ASTER_DATA_ROOT", "external_data")) / "LeukemiaAttr"


def resolve_root(path: Path) -> Path:
    path = Path(path)
    nested = path / "LeukemiaAttr"
    return nested if nested.is_dir() else path

# COCO category -> the frozen 16-class vocabulary (PREREGISTRATION.md section 2).
# `monoblast` is a blast of monocytic (myeloid) lineage -> myeloblast, as for AML-LMU MOB.
# `promonocyte` is an immature monocyte; the vocabulary has no such class and the CMML
# criterion counts promonocytes with monocytes, so it maps to `monocyte`. Declared here,
# not hidden.
CLASS_MAP = {
    "myeloblast": "myeloblast",
    "monoblast": "myeloblast",
    "lymphoblast": "lymphoblast",
    "abnormal promyelocyte": "promyelocyte_abnormal",
    "myelocyte": "myelocyte",
    "metamyelocyte": "metamyelocyte",
    "neutrophil": "segmented_neutrophil",
    "eosinophil": "eosinophil",
    "basophil": "basophil",
    "monocyte": "monocyte",
    "promonocyte": "monocyte",
    "lymphocyte": "lymphocyte",
    "atypical lymphocyte": "lymphocyte_atypical",
    "none": "other_artifact",
}

FIELDS = ["path", "split", "cell_class", "coco_class", "patient", "diagnosis", "subdomain",
          "field_image", "box_index", "cell_uid", "box_w", "box_h", "crop_w", "crop_h"]


DIR_NAMES = ("json_labels", "WBC_detection", "WBC_Detection")


def load_coco(subdomain: Path, split: str) -> dict | None:
    """COCO annotations for one split, from a directory or from the shipped zip.

    Subdomains are not packaged consistently: most expose `json_labels/`, while
    L_100X_C2 ships `labels.zip` holding the same `json_labels/` inside. Reading the zip
    in place keeps this tool self-contained on the LeukemiaAttr folder - no dependency on
    anything already unpacked elsewhere.
    """
    for name in DIR_NAMES:
        candidate = subdomain / name / f"{split}.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
    for archive_path in sorted(subdomain.glob("*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    parts = member.split("/")
                    if len(parts) >= 2 and parts[-1] == f"{split}.json" and parts[-2] in DIR_NAMES:
                        with archive.open(member) as handle:
                            return json.loads(handle.read().decode("utf-8"))
        except zipfile.BadZipFile:
            continue
    return None


def has_annotations(subdomain: Path) -> bool:
    return any(load_coco(subdomain, split) is not None for split in ("train", "test"))


def image_dir(subdomain: Path, split: str) -> Path | None:
    for name in ("Images", "images"):
        candidate = subdomain / name / split
        if candidate.is_dir():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="_work/leukemiaattr_crops")
    parser.add_argument("--subdomains", nargs="*", default=None)
    parser.add_argument("--min-side", type=int, default=16)
    parser.add_argument("--blank-std", type=float, default=6.0,
                        help="drop crops whose mean channel std is below this (blank/streak)")
    parser.add_argument("--limit", type=int, default=0, help="per subdomain, for a probe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="add to an existing manifest instead of truncating it")
    args = parser.parse_args()

    root = resolve_root(Path(args.root))
    out = Path(args.out)
    subdomains = sorted(p for p in root.iterdir()
                        if p.is_dir() and (args.subdomains is None or p.name in args.subdomains))

    # ---- pass 1: global patient -> split, enforced across every subdomain -----
    seen_in: dict[str, set[str]] = defaultdict(set)
    diagnosis: dict[str, str] = {}
    for subdomain in subdomains:
        for split in ("train", "test"):
            data = load_coco(subdomain, split)
            if data is None:
                continue
            for image in data["images"]:
                patient = image["file_name"].split("_")[0]
                seen_in[patient].add(split)
                diagnosis[patient] = image["file_name"].rsplit("_", 1)[1].rsplit(".", 1)[0]
    split_of = {p: ("test" if "test" in s else "train") for p, s in seen_in.items()}
    conflicted = [p for p, s in seen_in.items() if len(s) > 1]
    print(f"patients: {len(split_of)}  ->  train {sum(v=='train' for v in split_of.values())}"
          f" / test {sum(v=='test' for v in split_of.values())}")
    if conflicted:
        print(f"  {len(conflicted)} patient(s) appeared in BOTH splits across subdomains "
              f"{sorted(conflicted)} -> forced to test (no acquisition of a test patient "
              f"can reach train)")
    by_diagnosis = Counter(f"{diagnosis[p]}/{split_of[p]}" for p in split_of)
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(by_diagnosis.items())))

    if args.dry_run:
        return 0

    # ---- pass 2: cut ---------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    append = args.append and manifest_path.exists()
    manifest = manifest_path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest, fieldnames=FIELDS)
    if not append:
        writer.writeheader()

    kept: Counter = Counter()
    dropped = Counter()
    cells: set[str] = set()
    for subdomain in subdomains:
        if not has_annotations(subdomain):
            print(f"  [skip] {subdomain.name}: no annotations "
                  f"({'no images either' if image_dir(subdomain, 'train') is None else 'images present'})")
            continue
        written = 0
        for split in ("train", "test"):
            data = load_coco(subdomain, split)
            idir = image_dir(subdomain, split)
            if data is None or idir is None:
                continue
            categories = {c["id"]: c["name"] for c in data["categories"]}
            images = {i["id"]: i["file_name"] for i in data["images"]}
            per_image: dict[str, int] = defaultdict(int)
            for annotation in data["annotations"]:
                if args.limit and written >= args.limit:
                    break
                file_name = images[annotation["image_id"]]
                patient = file_name.split("_")[0]
                target_split = split_of[patient]          # global assignment wins
                coco_class = categories[annotation["category_id"]]
                cell_class = CLASS_MAP.get(coco_class)
                if cell_class is None:
                    dropped["unmapped_class"] += 1
                    continue
                box_index = per_image[file_name]
                per_image[file_name] += 1
                source = idir / file_name
                if not source.exists():
                    dropped["missing_image"] += 1
                    continue
                with Image.open(source) as handle:
                    image = handle.convert("RGB")
                    width, height = image.size
                    crop = extract_crop(
                        image, expand_and_clip_box(coco_to_xyxy(annotation["bbox"]), width, height))
                if crop is None:
                    dropped["degenerate_box"] += 1
                    continue
                if min(crop.size) < args.min_side:
                    dropped["too_small"] += 1
                    continue
                if sum(ImageStat.Stat(crop).stddev) / 3.0 < args.blank_std:
                    dropped["blank_or_streak"] += 1
                    continue

                stem = Path(file_name).stem
                name = f"{patient}__{subdomain.name}__{stem}__box{box_index:02d}.webp"
                target = out / target_split / cell_class
                target.mkdir(parents=True, exist_ok=True)
                crop.save(target / name, format="WEBP", lossless=True, quality=100, method=4)
                cell_uid = f"{patient}/{stem}/{box_index:02d}"
                cells.add(cell_uid)
                writer.writerow({
                    "path": f"{target_split}/{cell_class}/{name}", "split": target_split,
                    "cell_class": cell_class, "coco_class": coco_class, "patient": patient,
                    "diagnosis": diagnosis[patient], "subdomain": subdomain.name,
                    "field_image": file_name, "box_index": box_index, "cell_uid": cell_uid,
                    "box_w": round(annotation["bbox"][2], 1), "box_h": round(annotation["bbox"][3], 1),
                    "crop_w": crop.size[0], "crop_h": crop.size[1],
                })
                kept[(target_split, cell_class)] += 1
                written += 1
        print(f"  {subdomain.name:<12} {written:>6} crops")
        manifest.flush()
    manifest.close()

    print(f"\n{'class':<24}{'train':>8}{'test':>8}{'total':>8}")
    classes = sorted({c for _, c in kept})
    for cell_class in classes:
        train = kept[("train", cell_class)]
        test = kept[("test", cell_class)]
        print(f"{cell_class:<24}{train:>8}{test:>8}{train + test:>8}")
    total = sum(kept.values())
    print(f"{'TOTAL':<24}{sum(v for (s, _), v in kept.items() if s == 'train'):>8}"
          f"{sum(v for (s, _), v in kept.items() if s == 'test'):>8}{total:>8}")
    print(f"\nunique physical cells: {len(cells)}   crops: {total}   "
          f"({total / max(len(cells), 1):.1f} acquisitions per cell)")
    if dropped:
        print("dropped: " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    print(f"manifest: {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
