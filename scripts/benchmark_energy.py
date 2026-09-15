#!/usr/bin/env python3
"""Measure Idle, Acquisition, and Full Analysis under one comparable protocol.

Run this script on the target NVIDIA Jetson from the ASTER project environment.
Every mode is measured with the same tegrastats rail, sampling interval, and
campaign duration. Idle and Acquisition are therefore not reused from an older
campaign.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark_common import ROOT, build_manifest, clock_state, read_images, write_json
from aster_pipeline.config import load_config
from aster_pipeline.pipeline import LeukemiaPipeline


POWER = re.compile(r"\b([A-Za-z0-9_]+)\s+(\d+)mW(?:/(\d+)mW)?")
TEMP = re.compile(r"\b([A-Za-z0-9_]+)@([0-9.]+)C")
RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
STAMP = re.compile(r"^BENCHMARK_TS_NS=(\d+)\s+")
MODES = ("idle", "acquisition", "full_analysis")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Comparable Jetson energy benchmark for Idle, camera Acquisition, "
            "and the complete ASTER analysis pipeline."
        )
    )
    value.add_argument("--images", default=str(ROOT / "benchmark_session.txt"),
                       help="Text file with one benchmark image path per line")
    value.add_argument("--duration", type=float, default=300.0, help="Measured duration per mode in seconds")
    value.add_argument("--interval-ms", type=int, default=1000, help="tegrastats sampling interval")
    value.add_argument("--stabilization-s", type=float, default=30.0, help="Pause between measured modes")
    value.add_argument(
        "--initial-stabilization-s",
        type=float,
        default=30.0,
        help="Cooldown after model loading/validation and before Idle measurement",
    )
    value.add_argument("--warmup-runs", type=int, default=5, help="Full-pipeline warm-up runs")
    value.add_argument("--acquisition-warmup-s", type=float, default=5.0)
    value.add_argument("--skip-acquisition", action="store_true",
                       help="skip the camera Acquisition mode (measure only Idle + Full analysis; "
                            "the duty-cycle reconciliation needs only those two)")
    value.add_argument("--camera-index", type=int, default=0)
    value.add_argument("--camera-width", type=int, default=1280)
    value.add_argument("--camera-height", type=int, default=720)
    value.add_argument("--camera-fps", type=float, default=30.0)
    value.add_argument("--min-wbc", type=int, default=50)
    value.add_argument("--output", default="benchmarks/current_pipeline/energy_three_modes")
    value.add_argument("--config", default=str(ROOT / "config/inference.yaml"))
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--mil-backend", choices=("auto", "pytorch", "tensorrt"), default="auto")
    return value


def start_tegrastats(raw_path: Path, interval_ms: int) -> tuple[subprocess.Popen, object]:
    """Start tegrastats and prefix every sample with a monotonic host timestamp."""
    shell = (
        f"tegrastats --interval {interval_ms} | "
        "while IFS= read -r line; do "
        "printf 'BENCHMARK_TS_NS=%s %s\\n' \"$(date +%s%N)\" \"$line\"; "
        "done"
    )
    handle = raw_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-c", shell],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def stop_tegrastats(process: subprocess.Popen, handle: object) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    handle.close()


def parse_samples(path: Path) -> tuple[list[dict], str]:
    samples: list[dict] = []
    rail_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stamp = STAMP.search(line)
        if not stamp:
            continue
        powers = {name: int(current) / 1000 for name, current, _avg in POWER.findall(line)}
        for name in powers:
            rail_counts[name] = rail_counts.get(name, 0) + 1
        temperatures = {name.lower(): float(v) for name, v in TEMP.findall(line)}
        ram = RAM.search(line)
        samples.append(
            {
                "timestamp_ns": int(stamp.group(1)),
                **{f"power_{name}_w": value for name, value in powers.items()},
                **{f"temperature_{name}_c": value for name, value in temperatures.items()},
                "ram_used_mb": int(ram.group(1)) if ram else "",
                "raw": line[stamp.end() :],
            }
        )
    if len(samples) < 2:
        raise RuntimeError(f"{path}: tegrastats produced fewer than two usable samples")
    rail = "VDD_IN" if "VDD_IN" in rail_counts else max(rail_counts, key=rail_counts.get)
    samples = [row for row in samples if f"power_{rail}_w" in row]
    if len(samples) < 2:
        raise RuntimeError(f"{path}: fewer than two samples contain the selected rail {rail}")
    return samples, rail


def write_samples(path: Path, samples: list[dict]) -> None:
    fields = sorted({key for row in samples for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def summarize(samples: list[dict], rail: str, completed_units: int, unit_name: str) -> dict:
    timestamps = np.asarray([row["timestamp_ns"] for row in samples], dtype=np.float64) / 1e9
    powers = np.asarray([row[f"power_{rail}_w"] for row in samples], dtype=np.float64)
    energy_j = float(np.trapz(powers, timestamps))
    duration_s = float(timestamps[-1] - timestamps[0])
    gpu_temp = [
        value
        for row in samples
        for key, value in row.items()
        if key.startswith("temperature_gpu") and value != ""
    ]
    cpu_temp = [
        value
        for row in samples
        for key, value in row.items()
        if key.startswith("temperature_cpu") and value != ""
    ]
    ram = [row["ram_used_mb"] for row in samples if row["ram_used_mb"] != ""]
    return {
        "duration_from_samples_s": duration_s,
        "power_rail": rail,
        "power_mean_w": float(np.mean(powers)),
        "power_std_w": float(np.std(powers, ddof=1)),
        "power_median_w": float(np.median(powers)),
        "power_max_w": float(np.max(powers)),
        "total_energy_j": energy_j,
        "completed_units": completed_units,
        "unit_name": unit_name,
        "gross_energy_per_unit_j": energy_j / completed_units if completed_units else None,
        "gpu_temperature_max_c": max(gpu_temp, default=None),
        "cpu_temperature_max_c": max(cpu_temp, default=None),
        "ram_max_mb": max(ram, default=None),
        "sample_count": len(samples),
    }


def _reconcile(idle: dict, full: dict) -> dict:
    """Resolve the historical "full pipeline draws less power than MIL alone".

    The old pipeline spent ~90% of each run in PNG-zlib persistence I/O, so its
    300 s mean power was pulled toward idle. Report the active fraction and the
    energy that is actually attributable to an analysis, so a low *mean* power is
    not mistaken for a low *cost*.

      net power    = P(full mean) - P(idle mean)
      active wall  = sum of per-analysis wall-clock time
      net energy / analysis = net power x (active wall / analyses)
      gross energy / analysis = integral(P) over the whole mode / analyses
    """
    analyses = full.get("analyses")
    active_wall = full.get("analysis_wall_total_s")
    mode_wall = full.get("wall_duration_s")
    net_power = full["power_mean_w"] - idle["power_mean_w"]
    active_fraction = (active_wall / mode_wall) if (active_wall and mode_wall) else None
    net_energy = (net_power * active_wall / analyses) if (active_wall and analyses) else None
    return {
        "analyses": analyses,
        "analysis_wall_mean_s": full.get("analysis_wall_mean_s"),
        "active_wall_total_s": active_wall,
        "mode_wall_s": mode_wall,
        "active_fraction": active_fraction,
        "idle_gap_total_s": (mode_wall - active_wall) if (active_wall and mode_wall) else None,
        "artifact_flush_s": full.get("artifact_flush_s"),
        "idle_power_mean_w": idle["power_mean_w"],
        "full_analysis_power_mean_w": full["power_mean_w"],
        "net_power_w": net_power,
        "gross_energy_per_analysis_j": full.get("gross_energy_per_unit_j"),
        "net_energy_per_analysis_j": net_energy,
        "note": (
            "A near-1.0 active_fraction means the corrected pipeline is compute-bound, "
            "not I/O-bound; the mean power then reflects the real analysis cost."
        ),
    }


def measure_idle(duration_s: float) -> int:
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        # A quiet wait represents the application/system idle state.
        time.sleep(min(0.25, max(0.0, deadline - time.perf_counter())))
    return 0


def open_camera(args: argparse.Namespace):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Acquisition mode requires OpenCV (cv2) on the target Jetson") from error
    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera index {args.camera_index}")
    return capture


def acquisition_warmup(capture, duration_s: float) -> None:
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        ok, _frame = capture.read()
        if not ok:
            raise RuntimeError("Camera acquisition failed during warm-up")


def measure_acquisition(capture, duration_s: float) -> int:
    frames = 0
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        ok, _frame = capture.read()
        if not ok:
            raise RuntimeError("Camera acquisition failed during the measured campaign")
        frames += 1
    return frames


def measure_full_analysis(pipeline, inputs: Path, workspace: Path, duration_s: float) -> tuple[int, dict]:
    completed = 0
    analysis_wall: list[float] = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        result = pipeline.run(
            inputs,
            f"energy_full_{completed:05d}",
            workspace,
            benchmark=True,
        ).to_dict()
        analysis_wall.append(time.perf_counter() - started)
        # Block 2 v2 can legitimately withhold a verdict on a fixture outside its
        # validated acquisition domain (``out_of_domain``) while still running the
        # full pass; only a skipped-block-2 status is a real measurement failure.
        if result["status"] not in ("completed", "out_of_domain"):
            raise RuntimeError(f"Full analysis did not complete: {result['status']}")
        completed += 1
    # Drain the async artefact queue while tegrastats is still sampling, so the
    # power/energy cost of actually persisting this mode's output is included
    # in its measurement window rather than silently bleeding into whatever
    # runs next.
    flush_started = time.perf_counter()
    pipeline.flush_artifacts()
    flush_s = time.perf_counter() - flush_started
    active_wall = float(sum(analysis_wall))
    extra = {
        "analyses": completed,
        "analysis_wall_total_s": active_wall,
        "analysis_wall_mean_s": active_wall / completed if completed else None,
        "artifact_flush_s": flush_s,
    }
    return completed, extra


def measure_mode(
    mode: str,
    output: Path,
    interval_ms: int,
    workload,
) -> tuple[dict, list[dict]]:
    mode_dir = output / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    raw_path = mode_dir / "tegrastats_raw.log"
    process, handle = start_tegrastats(raw_path, interval_ms)
    started = time.perf_counter()
    try:
        outcome = workload()
    finally:
        stop_tegrastats(process, handle)
    completed, extra = outcome if isinstance(outcome, tuple) else (outcome, {})
    wall_duration = time.perf_counter() - started
    samples, rail = parse_samples(raw_path)
    write_samples(mode_dir / "energy_samples.csv", samples)
    unit_name = {"idle": "none", "acquisition": "frame", "full_analysis": "analysis"}[mode]
    summary = summarize(samples, rail, completed, unit_name)
    summary.update(extra)
    summary["mode"] = mode
    summary["wall_duration_s"] = wall_duration
    summary["created_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    write_json(mode_dir / "energy_summary.json", summary)
    return summary, samples


def plot_summary(summaries: dict[str, dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_by_mode = {"idle": "Idle", "acquisition": "Acquisition",
                     "full_analysis": "Full analysis\npipeline"}
    color_by_mode = {"idle": "#4C78A8", "acquisition": "#72A074", "full_analysis": "#E68A35"}
    modes = [m for m in MODES if m in summaries]
    labels = [label_by_mode[m] for m in modes]
    means = [summaries[mode]["power_mean_w"] for mode in modes]
    stds = [summaries[mode]["power_std_w"] for mode in modes]
    colors = [color_by_mode[m] for m in modes]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set(
        title="Power consumption over 300 s per mode",
        ylabel="Mean power (W)",
        xticks=x,
        xticklabels=labels,
    )
    upper = max(mean + std for mean, std in zip(means, stds))
    ax.set_ylim(0, max(12, upper * 1.22))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value, spread in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + spread + upper * 0.02,
            f"{value:.2f} W",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output / "power_consumption_three_modes.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "power_consumption_three_modes.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parser().parse_args()
    if args.duration <= 0 or args.interval_ms < 1:
        raise SystemExit("duration and interval-ms must be positive")
    if (
        args.stabilization_s < 0
        or args.initial_stabilization_s < 0
        or args.warmup_runs < 0
        or args.acquisition_warmup_s < 0
    ):
        raise SystemExit("stabilization and warm-up values must be non-negative")
    if shutil.which("tegrastats") is None:
        raise SystemExit("tegrastats was not found. Run this benchmark on the target NVIDIA Jetson.")

    images = read_images(args.images)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "_fixed_inputs"
    inputs.mkdir(exist_ok=True)
    for index, image in enumerate(images):
        shutil.copy2(image, inputs / f"{index:04d}_{image.name}")

    # Initialize and validate before any measured campaign.
    pipeline = LeukemiaPipeline(
        load_config(args.config),
        device=args.device,
        mil_backend=args.mil_backend,
    )
    validation = pipeline.run(
        inputs,
        "energy_validation",
        output / "_validation",
        benchmark=True,
    ).to_dict()
    pipeline.flush_artifacts()
    found = int(validation["number_of_detected_wbc"])
    if found < args.min_wbc or validation["status"] not in ("completed", "out_of_domain"):
        raise SystemExit(
            f"Validation failed: {found} accepted WBC(s); minimum is {args.min_wbc}; "
            f"status={validation['status']}"
        )
    for index in range(args.warmup_runs):
        pipeline.run(inputs, f"energy_warmup_{index}", output / "_run_workspace", benchmark=True)
    pipeline.flush_artifacts()

    summaries: dict[str, dict] = {}

    if args.initial_stabilization_s:
        print(f"Initial stabilization for {args.initial_stabilization_s:.0f} s")
        time.sleep(args.initial_stabilization_s)

    print(f"[1/3] Measuring Idle for {args.duration:.0f} s")
    summaries["idle"], _ = measure_mode(
        "idle",
        output,
        args.interval_ms,
        lambda: measure_idle(args.duration),
    )

    if args.stabilization_s:
        print(f"Stabilizing for {args.stabilization_s:.0f} s")
        time.sleep(args.stabilization_s)

    if args.skip_acquisition:
        print("Skipping Acquisition mode (--skip-acquisition)")
    else:
        capture = open_camera(args)
        try:
            acquisition_warmup(capture, args.acquisition_warmup_s)
            print(f"[2/3] Measuring Acquisition for {args.duration:.0f} s")
            summaries["acquisition"], _ = measure_mode(
                "acquisition",
                output,
                args.interval_ms,
                lambda: measure_acquisition(capture, args.duration),
            )
        finally:
            capture.release()

        if args.stabilization_s:
            print(f"Stabilizing for {args.stabilization_s:.0f} s")
            time.sleep(args.stabilization_s)

    print(f"[3/3] Measuring Full analysis for {args.duration:.0f} s")
    summaries["full_analysis"], _ = measure_mode(
        "full_analysis",
        output,
        args.interval_ms,
        lambda: measure_full_analysis(
            pipeline,
            inputs,
            output / "_run_workspace",
            args.duration,
        ),
    )
    pipeline.close()

    rails = {summary["power_rail"] for summary in summaries.values()}
    if len(rails) != 1:
        raise RuntimeError(f"Modes used different power rails: {sorted(rails)}")

    modes_run = [m for m in MODES if m in summaries]
    combined = {
        "protocol": {
            "mode_order": modes_run,
            "requested_duration_per_mode_s": args.duration,
            "tegrastats_interval_ms": args.interval_ms,
            "stabilization_between_modes_s": args.stabilization_s,
            "initial_stabilization_s": args.initial_stabilization_s,
            "power_rail": next(iter(rails)),
            "camera_index": args.camera_index,
            "camera_resolution": [args.camera_width, args.camera_height],
            "camera_requested_fps": args.camera_fps,
            "full_pipeline_warmup_runs": args.warmup_runs,
        },
        "modes": summaries,
        "incremental_power_w": {
            **(
                {"acquisition_minus_idle": (
                    summaries["acquisition"]["power_mean_w"] - summaries["idle"]["power_mean_w"]
                )}
                if "acquisition" in summaries else {}
            ),
            "full_analysis_minus_idle": (
                summaries["full_analysis"]["power_mean_w"] - summaries["idle"]["power_mean_w"]
            ),
        },
        "duty_cycle_reconciliation": _reconcile(summaries["idle"], summaries["full_analysis"]),
        "clock_state": clock_state(),
        "note": (
            "All three modes were remeasured in one execution using the same target, "
            "power rail, sampling interval, and requested duration."
        ),
    }
    write_json(output / "energy_three_modes_summary.json", combined)

    manifest = build_manifest(images, validation, args.config)
    manifest["energy_three_modes_protocol"] = combined["protocol"]
    write_json(output / "benchmark_manifest.json", manifest)
    plot_summary(summaries, output)
    print(f"Completed comparable three-mode campaign: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
