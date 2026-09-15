#!/usr/bin/env python3
"""End-to-end benchmark on the Jetson: microscope fields -> block 1 (deployed YOLO + crop
contract) -> block 2 v2 (this kit). Latency per stage, power, energy, temperatures, RAM.

Block 1 is imported from the deployed repository, not re-implemented: the crops are the
ones the product produces. Block 2 is `load_kit.load_scorer`. The tegrastats capture and
its parsing are copied verbatim from scripts/benchmark_energy.py (repository root)
(VDD_IN rail, host timestamps, trapezoidal energy), so the numbers are comparable with the
deployed pipeline's published campaign.

Modes
  session  one analysis per session, result.json per session. With --crops-per-session N,
           all fields are localised first and the crops are cut into sessions of N - the
           ×40 stress-test protocol of the Colab run (§10), replayed on the device.
  timing   --warmup W then --runs R full analyses of the whole field set; per-stage
           latencies (tegrastats recorded alongside when available).
  energy   --idle-seconds of idle, then full analyses back to back for --duration seconds
           under tegrastats: mean/max power, energy per analysis, max GPU/CPU temperature.

Example (on the Jetson, MAXN_SUPER, jetson_clocks):
  python3 benchmark_block2_e2e.py --mode timing --fields fields/benchmark_allidb_L2 --output out/timing_allidb
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent

# --- tegrastats: verbatim from scripts/benchmark_energy.py (repository root) ----------
POWER = re.compile(r"\b([A-Za-z0-9_]+)\s+(\d+)mW(?:/(\d+)mW)?")
TEMP = re.compile(r"\b([A-Za-z0-9_]+)@([0-9.]+)C")
RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
STAMP = re.compile(r"^BENCHMARK_TS_NS=(\d+)\s+")


def start_tegrastats(raw_path: Path, interval_ms: int):
    shell = (f"tegrastats --interval {interval_ms} | while IFS= read -r line; do "
             "printf 'BENCHMARK_TS_NS=%s %s\\n' \"$(date +%s%N)\" \"$line\"; done")
    handle = raw_path.open("w", encoding="utf-8")
    process = subprocess.Popen(["bash", "-c", shell], stdout=handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return process, handle


def stop_tegrastats(process, handle) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    handle.close()


def parse_samples(path: Path):
    samples, rail_counts = [], {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stamp = STAMP.search(line)
        if not stamp:
            continue
        powers = {name: int(current) / 1000 for name, current, _avg in POWER.findall(line)}
        for name in powers:
            rail_counts[name] = rail_counts.get(name, 0) + 1
        temperatures = {name.lower(): float(v) for name, v in TEMP.findall(line)}
        ram = RAM.search(line)
        samples.append({"timestamp_ns": int(stamp.group(1)),
                        **{f"power_{name}_w": value for name, value in powers.items()},
                        **{f"temperature_{name}_c": value for name, value in temperatures.items()},
                        "ram_used_mb": int(ram.group(1)) if ram else ""})
    if len(samples) < 2 or not rail_counts:
        raise RuntimeError(f"{path}: tegrastats produced fewer than two usable samples")
    rail = "VDD_IN" if "VDD_IN" in rail_counts else max(rail_counts, key=rail_counts.get)
    return [row for row in samples if f"power_{rail}_w" in row], rail


def summarize_power(samples, rail, completed_units, unit_name):
    t = np.asarray([row["timestamp_ns"] for row in samples], dtype=np.float64) / 1e9
    p = np.asarray([row[f"power_{rail}_w"] for row in samples], dtype=np.float64)
    energy = float(np.trapz(p, t)) if hasattr(np, "trapz") else float(np.trapezoid(p, t))
    temps = lambda prefix: [v for row in samples for k, v in row.items()
                            if k.startswith(f"temperature_{prefix}") and v != ""]
    ram = [row["ram_used_mb"] for row in samples if row["ram_used_mb"] != ""]
    return {"duration_from_samples_s": float(t[-1] - t[0]), "power_rail": rail,
            "power_mean_w": float(p.mean()), "power_std_w": float(p.std(ddof=1)),
            "power_median_w": float(np.median(p)), "power_max_w": float(p.max()),
            "total_energy_j": energy, "completed_units": completed_units, "unit_name": unit_name,
            "gross_energy_per_unit_j": energy / completed_units if completed_units else None,
            "gpu_temperature_max_c": max(temps("gpu"), default=None),
            "cpu_temperature_max_c": max(temps("cpu"), default=None),
            "ram_max_mb": max(ram, default=None), "sample_count": len(samples)}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
# ----------------------------------------------------------------------------------------------


def command(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def read_fields(source: Path) -> tuple[list[Path], list[str]]:
    """Fields from a directory (recursive) or a list file; unreadable fields are excluded and
    named (one ×40 field has a broken JPEG stream: OpenCV tolerates it, Pillow does not)."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("._"))
    else:
        lines = [l.strip() for l in source.read_text().splitlines() if l.strip() and not l.startswith("#")]
        paths = [(source.parent / l).resolve() if not Path(l).is_absolute() else Path(l) for l in lines]
    good, bad = [], []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.convert("RGB")
            good.append(path)
        except Exception as exc:
            bad.append(f"{path}: {exc}")
    return good, bad


class EndToEnd:
    def __init__(self, deployed: Path, kit: Path, backend: str, engine: Path | None, device: str) -> None:
        sys.path.insert(0, str(deployed))
        sys.path.insert(0, str(kit))
        from aster_pipeline.config import load_config
        from aster_pipeline.timing import Timings
        from aster_pipeline.yolo_detector import YoloWBCDetector
        from load_kit import load_config as load_kit_config, load_scorer

        self.Timings = Timings
        config = load_config(deployed / "config/inference.yaml")
        yolo = config.yolo
        # same rule as the deployed pipeline: the TensorRT YOLO engine when it exists on CUDA
        if device == "cuda" and yolo.tensorrt_weights is not None and yolo.tensorrt_weights.is_file():
            yolo = replace(yolo, weights=yolo.tensorrt_weights)
        self.yolo_weights = yolo.weights
        self.detector = YoloWBCDetector(yolo, device, config.output)
        self.scorer = load_scorer(kit, backend=backend, engine=engine, device=device)
        self.kit_config = load_kit_config(kit)
        self.device = device
        self.sync = torch.cuda.synchronize if device == "cuda" else None
        self.scratch = Path(tempfile.mkdtemp(prefix="aster_e2e_"))

    def localise(self, fields: list[Path], timings) -> list:
        # Crops stay in memory (crops_dir=None) and the overlay write is discarded: persistence
        # belongs to the product's I/O, not to the model; its encode cost is still timed.
        return self.detector.detect(fields, self.scratch, None, write_artifact=lambda _p, _b: None,
                                    timings=timings)

    def score(self, detections: list, session_id: str, n_fields: int, timings):
        with timings.measure("block2_crop_transform_ms"):
            tensors = torch.stack([self.scorer.transform(d.image_rgb) for d in detections])
        with timings.measure("block2_total_ms", gpu=self.device == "cuda"):
            result, detail = self.scorer.score(tensors, session_id=session_id, number_of_fields=n_fields)
        for key, value in result.timing_ms.items():
            timings.values[f"block2_{key}"] = value
        return result

    def analyse(self, fields: list[Path], session_id: str):
        timings = self.Timings(self.sync)
        start = time.perf_counter_ns()
        detections = self.localise(fields, timings)
        result = self.score(detections, session_id, len(fields), timings) if detections else None
        timings.values["total_ms"] = (time.perf_counter_ns() - start) / 1e6
        timings.values["n_crops"] = len(detections)
        return result, timings.values


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "p95": ordered[round(0.95 * (len(ordered) - 1))],
            "min": ordered[0], "max": ordered[-1], "n": len(values)}


def manifest(args, e2e: EndToEnd, fields: list[Path], excluded: list[str]) -> dict:
    kit = Path(args.kit)
    engine_meta = None
    engine = Path(args.engine) if args.engine else kit / "models/encoder_fp16.engine"
    if engine.with_suffix(".engine.json").exists():
        engine_meta = json.loads(engine.with_suffix(".engine.json").read_text())
    release = Path("/etc/nv_tegra_release")
    return {"started": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
            "fields": [str(p) for p in fields], "fields_excluded": excluded,
            "fields_sha256": hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in fields).encode()).hexdigest(),
            "block1_weights": str(e2e.yolo_weights), "block2_backend": args.backend,
            "block2_engine": engine_meta, "kit_run_id": e2e.kit_config.get("run_id"),
            "device": args.device, "python": platform.python_version(), "torch": torch.__version__,
            "l4t": release.read_text().splitlines()[0] if release.exists() else None,
            "nvpmodel": command(["nvpmodel", "-q"]),
            "jetson_clocks": command(["jetson_clocks", "--show"]),
            "declaration": "technical benchmark on a fixed field set, not a clinical session"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("session", "timing", "energy"), required=True)
    ap.add_argument("--fields", required=True, help="directory of fields, or a list file")
    ap.add_argument("--deployed", default=str(HERE.parents[2]))   # repository root (deployed pipeline)
    ap.add_argument("--kit", default=str(HERE.parent / "kit"))
    ap.add_argument("--backend", choices=("tensorrt", "torch"), default="tensorrt")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--crops-per-session", type=int, default=0)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fields, excluded = read_fields(Path(args.fields))
    if not fields:
        raise SystemExit("no readable field")
    print(f"{len(fields)} fields" + (f", {len(excluded)} unreadable excluded" if excluded else ""))
    e2e = EndToEnd(Path(args.deployed).expanduser(), Path(args.kit), args.backend,
                   Path(args.engine) if args.engine else None, args.device)
    (out / "manifest.json").write_text(json.dumps(manifest(args, e2e, fields, excluded), indent=2))
    has_tegrastats = shutil.which("tegrastats") is not None

    if args.mode == "session":
        sessions = []
        if args.crops_per_session:
            timings = e2e.Timings(e2e.sync)
            detections = e2e.localise(fields, timings)
            chunks = [detections[i:i + args.crops_per_session]
                      for i in range(0, len(detections), args.crops_per_session)]
            chunks = [c for c in chunks if len(c) >= e2e.scorer.thresholds.tier_screening]
            print(f"{len(detections)} crops -> {len(chunks)} sessions of <= {args.crops_per_session}")
            for index, chunk in enumerate(chunks):
                t = e2e.Timings(e2e.sync)
                result = e2e.score(chunk, f"session_{index:03d}", 0, t)
                sessions.append((result, {**t.values, "n_crops": len(chunk)}))
        else:
            result, values = e2e.analyse(fields, "session_000")
            sessions.append((result, values))
        rows = []
        for index, (result, values) in enumerate(sessions):
            if result is None:
                print("no WBC localised"); continue
            (out / f"result_{index:03d}.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
            rows.append({"session": result.session_id, "label": result.label, "tier": result.tier,
                         "n_crops": values["n_crops"], "n_classified": result.number_of_classified_leukocytes,
                         "p_abn": round(result.uncertainty["p_abn"], 4),
                         "ood": round(result.uncertainty["ood_score"], 2),
                         "reasons": " ; ".join(result.reasons)})
            print(f"{result.session_id}: {result.label:<34} crops {values['n_crops']:>4}  "
                  f"P_abn {result.uncertainty['p_abn']:.3f}  OOD {result.uncertainty['ood_score']:.1f}")
        write_csv(out / "sessions.csv", rows)
        return 0

    if args.mode == "timing":
        for _ in range(args.warmup):
            e2e.analyse(fields, "warmup")
        process = handle = None
        if has_tegrastats:
            process, handle = start_tegrastats(out / "tegrastats_raw.log", args.interval_ms)
        runs = []
        try:
            for index in range(args.runs):
                result, values = e2e.analyse(fields, f"timing_{index:03d}")
                runs.append({"run": index, "label": result.label if result else "no_wbc", **values})
        finally:
            if process:
                stop_tegrastats(process, handle)
        write_csv(out / "timing_runs.csv", runs)
        stages = sorted({k for r in runs for k, v in r.items() if k.endswith("_ms")})
        summary = {"runs": len(runs), "fields": len(fields),
                   "n_crops": describe([r["n_crops"] for r in runs]),
                   "labels": {l: sum(r["label"] == l for r in runs) for l in {r["label"] for r in runs}},
                   "stages_ms": {k: describe([r[k] for r in runs if k in r]) for k in stages}}
        if process:
            samples, rail = parse_samples(out / "tegrastats_raw.log")
            write_csv(out / "tegrastats_samples.csv", samples)
            summary["power_during_timing"] = summarize_power(samples, rail, len(runs), "analysis")
        (out / "timing_summary.json").write_text(json.dumps(summary, indent=2))
        total = summary["stages_ms"]["total_ms"]
        print(f"total {total['mean']:.1f} ± {total['sd']:.1f} ms (median {total['median']:.1f}, P95 {total['p95']:.1f})")
        for k in stages:
            print(f"  {k:<36} median {summary['stages_ms'][k]['median']:9.2f} ms")
        return 0

    # energy
    if not has_tegrastats:
        raise SystemExit("tegrastats not found: run the energy mode on the Jetson")
    for _ in range(args.warmup):
        e2e.analyse(fields, "warmup")
    report = {}
    if args.idle_seconds > 0:
        process, handle = start_tegrastats(out / "tegrastats_idle.log", args.interval_ms)
        time.sleep(args.idle_seconds)
        stop_tegrastats(process, handle)
        samples, rail = parse_samples(out / "tegrastats_idle.log")
        report["idle"] = summarize_power(samples, rail, 0, "second")
    process, handle = start_tegrastats(out / "tegrastats_full.log", args.interval_ms)
    completed, walls = 0, []
    deadline = time.perf_counter() + args.duration
    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            e2e.analyse(fields, f"energy_{completed:05d}")
            walls.append(time.perf_counter() - t0); completed += 1
    finally:
        stop_tegrastats(process, handle)
    samples, rail = parse_samples(out / "tegrastats_full.log")
    write_csv(out / "energy_samples.csv", samples)
    report["full_analysis"] = {**summarize_power(samples, rail, completed, "analysis"),
                               "analysis_wall_mean_s": float(np.mean(walls))}
    if "idle" in report:
        report["incremental_energy_per_analysis_j"] = (
            (report["full_analysis"]["power_mean_w"] - report["idle"]["power_mean_w"])
            * report["full_analysis"]["analysis_wall_mean_s"])
    (out / "energy_summary.json").write_text(json.dumps(report, indent=2))
    full = report["full_analysis"]
    print(f"{completed} analyses in {full['duration_from_samples_s']:.0f} s: mean {full['power_mean_w']:.2f} W "
          f"(max {full['power_max_w']:.2f}), {full['gross_energy_per_unit_j']:.2f} J/analysis gross, "
          f"GPU max {full['gpu_temperature_max_c']} °C, CPU max {full['cpu_temperature_max_c']} °C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
