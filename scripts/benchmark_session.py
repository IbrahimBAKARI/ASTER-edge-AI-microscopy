"""Target-device benchmark: 5 warm-ups, then 100 active/flush measurements."""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import ASTERBackend, AnalysisRequest


def summary(values: list[float]) -> str:
    ordered = sorted(values)
    p95 = ordered[round(.95 * (len(ordered) - 1))]
    return (f"mean ± SD {statistics.mean(values):.3f} ± {statistics.stdev(values):.3f} ms; "
            f"median {statistics.median(values):.3f} ms; P95 {p95:.3f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    root, source = Path(args.output_root).resolve(), Path(args.input).resolve()
    root.mkdir(parents=True, exist_ok=True)
    backend = ASTERBackend(Path(__file__).parents[1] / "config" / "inference.yaml")
    active, persistence = [], []
    for index in range(args.warmups + args.runs):
        destination = root / f"run_{index:03d}"
        started = perf_counter()
        backend.analyze(AnalysisRequest(source, f"benchmark-{index:03d}", destination))
        active_ms = (perf_counter() - started) * 1000
        flush_started = perf_counter(); backend.flush_artifacts()
        flush_ms = (perf_counter() - flush_started) * 1000
        missing = [name for name in ("crop_manifest.csv", "result.json", "annotated") if not (destination / name).exists()]
        if missing:
            raise RuntimeError(f"{destination}: missing {missing}")
        if index >= args.warmups:
            active.append(active_ms); persistence.append(flush_ms)
    print("Active processing:", summary(active))
    print("Persistence flush:", summary(persistence))
    print("Perceived async return:", summary(active))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
