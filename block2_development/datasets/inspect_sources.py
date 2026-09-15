"""Contact sheet of random samples from every ENABLED source, for eye verification.

Granularity is a declared property in sources.yaml (`granularity: cell | field`). This
script is how that declaration gets checked: it samples at random - not the first files,
which are the easiest to be misled by - and writes one labelled row per source.

    python datasets/inspect_sources.py --out _work/inspect.png --per-source 8 --seed 0

Change --seed and run again for a different draw. Pillow only.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw  # noqa: E402

from adapters import ADAPTERS  # noqa: E402
from prepare_corpus import load_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="datasets/sources.yaml")
    parser.add_argument("--out", default="_work/inspect.png")
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=int, default=220)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--scan-cap", type=int, default=6000,
                        help="stop enumerating a source after this many records")
    args = parser.parse_args()

    config = load_sources(Path(args.config))
    data_root = Path(config["root"])
    rng = random.Random(args.seed)

    rows: list[tuple[str, str, list[Path]]] = []
    for key, spec in config["sources"].items():
        enabled = str(spec.get("enabled", "true")).lower() != "false"
        if not enabled and not args.include_disabled:
            continue
        root = Path(spec["path"]) if str(spec["path"]).startswith("/") else data_root / spec["path"]
        if not root.exists():
            print(f"  [skip] {key}: {root} not found")
            continue
        paths = []
        for index, record in enumerate(ADAPTERS[key](root)):
            if index >= args.scan_cap:
                break
            paths.append(record["path"])
        if not paths:
            continue
        chosen = rng.sample(paths, min(args.per_source, len(paths)))
        label = f"{key}  [{spec.get('granularity', 'UNDECLARED')}]"
        rows.append((label, str(root), chosen))
        print(f"  {label:<34} {len(paths):>6} records scanned, {len(chosen)} sampled")

    tile, margin = args.tile, 26
    width = tile * args.per_source
    height = (tile + margin) * len(rows)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row, (label, root, paths) in enumerate(rows):
        top = row * (tile + margin)
        draw.rectangle([0, top, width, top + margin], fill=(24, 24, 28))
        draw.text((6, top + 7), f"{label}   {root}", fill=(255, 255, 255))
        for column, path in enumerate(paths):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                original = image.size
                image = image.resize((tile, tile), Image.LANCZOS)
            sheet.paste(image, (column * tile, top + margin))
            draw.text((column * tile + 4, top + margin + 4),
                      f"{original[0]}x{original[1]}", fill=(255, 255, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\n  written: {out}   ({len(rows)} sources, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
