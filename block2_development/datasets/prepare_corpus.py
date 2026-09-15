"""Mac side: build one compact, self-contained training corpus.

For every image of every local dataset:
    SquarePad(median of the 4 corners)      <- the deployed block-2 contract
    downscale so that the side is <= --max-side   (never upscale)
    save lossless WebP
and write a single manifest.csv describing everything.

Why: the raw datasets are ~28 GB of uncompressed TIFF. The corpus is roughly 8 GB,
lossless, already square-padded, and small enough to move to Colab in one go. Images
already at or below --max-side (cAItomorph and AML-MLL are 144x144) are re-encoded
without any resampling, so the test set stays pixel-exact.

Only Pillow is required. Run:  python datasets/prepare_corpus.py --out _work/corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from adapters import ADAPTERS  # noqa: E402
from aster_block2.crops import (  # noqa: E402
    coco_to_xyxy, expand_and_clip_box, extract_crop, guard_is_single_cell, yolo_to_xyxy,
)
from aster_block2.preprocess import square_pad  # noqa: E402

MANIFEST_FIELDS = [
    "corpus_path", "source", "patient_id", "role", "cell_labels", "weight",
    "bag_label", "orig_width", "orig_height", "stored_side", "resampled",
    "cropped_from_field", "extra",
]


def load_sources(config_path: Path) -> dict:
    """Minimal YAML reader for the flat structure of sources.yaml (no PyYAML needed)."""
    try:
        import yaml
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except ImportError:
        pass
    data, stack = {}, [(-1, data)]
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, value = raw.strip().partition(":")
        value = value.strip().strip('"').strip("'")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value:
            parent[key] = value
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return data


def convert_one(job: tuple) -> dict | None:
    """One image: crop if the record carries a box, square-pad, cap the side, encode.

    Runs in a worker process, so it must not touch the manifest or any shared state.
    """
    (source_path, corpus_rel, out_root, key, granularity, box_norm, bbox_xywh,
     max_side, method) = job
    target = Path(out_root) / corpus_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _convert(source_path, corpus_rel, target, key, granularity,
                        box_norm, bbox_xywh, max_side, method)
    except (OSError, ValueError) as exc:
        # A corrupt or truncated source file is EXCLUDED and NAMED, never repaired.
        # Pillow can pad a truncated JPEG with LOAD_TRUNCATED_IMAGES, but that would put a
        # half-decoded cell into the training set under a valid label, and nothing
        # downstream would ever notice.
        if isinstance(exc, ValueError) and "declared `field`" in str(exc):
            raise
        return {"corrupt": f"{source_path}: {exc}"}


def _convert(source_path, corpus_rel, target, key, granularity, box_norm, bbox_xywh,
             max_side, method) -> dict | None:
    with Image.open(source_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        if box_norm is not None:
            box = yolo_to_xyxy(box_norm, width, height)
        elif bbox_xywh is not None:
            box = coco_to_xyxy(bbox_xywh)
        else:
            box = None
        if box is None:
            if granularity == "field":
                raise ValueError(f"{key}: {Path(source_path).name} declared `field` but "
                                 f"carries no box")
            guard_is_single_cell((width, height), key, Path(source_path))
            cell = image
        else:
            cell = extract_crop(image, expand_and_clip_box(box, width, height))
            if cell is None:
                return None                      # degenerate box: detection dropped
        padded = square_pad(cell)
    side = padded.size[0]
    resampled = 0
    if side > max_side:
        padded = padded.resize((max_side, max_side), Image.BILINEAR)
        side, resampled = max_side, 1
    padded.save(target, format="WEBP", lossless=True, quality=100, method=method)
    return {"corpus_path": corpus_rel, "orig_width": width, "orig_height": height,
            "stored_side": side, "resampled": resampled, "bytes": target.stat().st_size,
            "cropped_from_field": int(box is not None)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="datasets/sources.yaml")
    parser.add_argument("--out", default="_work/corpus")
    parser.add_argument("--max-side", type=int, default=256,
                        help="stored side cap; 224 is applied at load time")
    parser.add_argument("--only", nargs="*", help="restrict to these source keys")
    parser.add_argument("--limit", type=int, default=0, help="per-source cap, for a dry run")
    parser.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    parser.add_argument("--webp-method", type=int, default=2,
                        help="WebP effort 0-6. Measured on this corpus: method 0 = 1151 "
                             "img/s at 33.2 KB, method 2 = 145 img/s at 29.1 KB, method 4 "
                             "= 20 img/s at 27.7 KB. 4 costs 57x the time of 0 for 17 %% "
                             "less space - 2 is the useful trade.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="conversion is embarrassingly parallel; the manifest is still "
                             "written by the parent process, one row at a time")
    args = parser.parse_args()

    config = load_sources(Path(args.config))
    data_root = Path(config["root"])
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    done = set()
    if manifest_path.exists():  # idempotent: skip what is already converted
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            done = {row["corpus_path"] for row in csv.DictReader(handle)}
        print(f"resuming: {len(done)} images already in the corpus")

    handle = None if args.dry_run else manifest_path.open("a", newline="", encoding="utf-8")
    writer = None
    if handle is not None:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not done:
            writer.writeheader()

    totals: dict[str, dict[str, int]] = {}
    corrupt_files: list[str] = []
    for key, spec in config["sources"].items():
        if args.only and key not in args.only:
            continue
        source_root = data_root / spec["path"] if not str(spec["path"]).startswith("/") else Path(spec["path"])
        if str(spec.get("enabled", "true")).lower() == "false":
            print(f"  [skip] {key}: disabled in sources.yaml")
            continue
        granularity = spec.get("granularity")
        if granularity not in ("cell", "field"):
            raise SystemExit(
                f"{key}: sources.yaml must declare `granularity: cell` or `granularity: field`.\n"
                f"  Image dimensions cannot decide this - Taleqani's 224x224 images are\n"
                f"  downscaled FIELDS, not cells. Look at the images, then declare it.")
        if not source_root.exists():
            print(f"  [skip] {key}: {source_root} not found")
            continue
        stats = totals.setdefault(key, {"n": 0, "bytes": 0, "resampled": 0, "skipped": 0,
                                        "dropped_degenerate": 0, "dropped_corrupt": 0})
        print(f"  [{key}] {source_root}")
        jobs, meta = [], []
        for index, record in enumerate(ADAPTERS[key](source_root)):
            if args.limit and index >= args.limit:
                break
            relative = record["path"].relative_to(source_root)
            box_index = record["extra"].get("box_index")
            if box_index is None:
                corpus_rel = f"{key}/{relative.with_suffix('.webp')}"
            else:
                corpus_rel = f"{key}/{relative.parent}/{relative.stem}__box{box_index:03d}.webp"
            if corpus_rel in done:
                stats["skipped"] += 1
                continue
            stats["n"] += 1
            if args.dry_run:
                continue
            jobs.append((str(record["path"]), corpus_rel, str(out_root), key, granularity,
                         record["extra"].get("box_norm"), record["extra"].get("bbox_xywh"),
                         args.max_side, args.webp_method))
            meta.append(record)

        if args.dry_run or not jobs:
            continue

        executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        results = (executor.map(convert_one, jobs, chunksize=64) if executor
                   else map(convert_one, jobs))
        written = 0
        for record, outcome in zip(meta, results):
            if outcome is None:
                stats["dropped_degenerate"] += 1
                stats["n"] -= 1
                continue
            if "corrupt" in outcome:
                stats["dropped_corrupt"] += 1
                stats["n"] -= 1
                corrupt_files.append(outcome["corrupt"])
                continue
            stats["bytes"] += outcome.pop("bytes")
            stats["resampled"] += outcome["resampled"]
            writer.writerow({
                **outcome,
                "source": record["source"],
                "patient_id": record["patient_id"] or "",
                "role": record["role"],
                "cell_labels": "|".join(record["cell_labels"]) if record["cell_labels"] else "",
                "weight": record["weight"],
                "bag_label": record["bag_label"] or "",
                "extra": json.dumps(record["extra"], separators=(",", ":")),
            })
            written += 1
            if written % 5000 == 0:
                handle.flush()
                print(f"      {written}/{len(jobs)}  {stats['bytes'] / 1e9:.2f} GB", flush=True)
        if executor is not None:
            executor.shutdown()
        handle.flush()

    if handle is not None:
        handle.close()

    print("\n  source            images   resampled       size")
    for key, stats in totals.items():
        print(f"  {key:<16} {stats['n']:>8} {stats['resampled']:>11} {stats['bytes'] / 1e9:>8.2f} GB"
              + (f"   (+{stats['skipped']} already done)" if stats["skipped"] else "")
              + (f"   [{stats['dropped_degenerate']} degenerate boxes dropped]"
                 if stats.get("dropped_degenerate") else "")
              + (f"   [{stats['dropped_corrupt']} CORRUPT files dropped]"
                 if stats.get("dropped_corrupt") else ""))
    grand = sum(s["bytes"] for s in totals.values())
    print(f"  {'TOTAL':<16} {sum(s['n'] for s in totals.values()):>8} {'':>11} {grand / 1e9:>8.2f} GB")
    if corrupt_files:
        listing = out_root / "corrupt_files.txt"
        listing.write_text("\n".join(corrupt_files) + "\n")
        print(f"\n  {len(corrupt_files)} corrupt source file(s) excluded, listed in {listing}")
        for line in corrupt_files[:5]:
            print(f"    {line}")
    if not args.dry_run:
        print(f"\n  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
