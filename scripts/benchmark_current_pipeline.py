#!/usr/bin/env python3
"""Latency benchmark of the exact LeukemiaPipeline.run path used by Analyze."""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark_common import ROOT, build_manifest, read_images, write_json
from aster_pipeline.config import load_config
from aster_pipeline.pipeline import LeukemiaPipeline

STAGES = [
    "session_image_loading_ms", "yolo_inference_ms", "wbc_crop_extraction_ms",
    "crop_preprocessing_ms", "block2_encode_ms", "block2_heads_ms",
    "block2_grid_ms", "result_rendering_ms",
    "output_persistence_ms", "total_analysis_ms",
]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Benchmark current full ASTER Analyze pipeline.")
    value.add_argument("--images", default=str(ROOT / "benchmark_session.txt"),
                       help="Text file containing one image path per line")
    value.add_argument("--warmup", type=int, default=5)
    value.add_argument("--runs", type=int, default=100)
    value.add_argument("--min-wbc", type=int, default=50)
    value.add_argument("--output", default="benchmarks/current_pipeline/timing")
    value.add_argument("--config", default=str(ROOT / "config/inference.yaml"))
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--mil-backend", choices=("auto", "pytorch", "tensorrt"), default="auto")
    return value


def prepare_inputs(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        shutil.copy2(image, destination / f"{index:04d}_{image.name}")


def run_once(pipeline, inputs: Path, output: Path, session_id: str) -> dict:
    """Run one session and report both halves of the async story.

    ``total_analysis_ms`` (measured here, wrapping ``pipeline.run``) is the
    time until a decision is available -- writes are only queued at that
    point. ``artifact_flush_ms`` additionally measures the time until every
    queued artefact for this session is fsync'd and atomically renamed, by
    draining the queue before starting the next iteration.
    """
    started = time.perf_counter_ns()
    result = pipeline.run(inputs, session_id, output, benchmark=True)
    finished = time.perf_counter_ns()
    value = result.to_dict()
    value["timing_ms"]["total_analysis_ms"] = (finished - started) / 1_000_000
    value["artifact_flush_ms"] = pipeline.flush_artifacts()
    return value


# Block 2 v2 can legitimately withhold a verdict on a fixture outside its validated
# acquisition domain (``out_of_domain``, e.g. the ALL-IDB benchmark fixture) while
# still running the full encode/heads/grid pass -- the timing is still meaningful.
# ``insufficient_evidence`` means block 2 never ran at all and is excluded.
_MEASURABLE_STATUSES = {"completed", "out_of_domain"}


def validate(result: dict, minimum: int) -> None:
    found = int(result["number_of_detected_wbc"])
    if found < minimum:
        raise RuntimeError(
            f"Technical benchmark aborted: {found} accepted WBC(s), minimum required is {minimum}."
        )
    if result["status"] not in _MEASURABLE_STATUSES:
        raise RuntimeError(f"Technical benchmark validation did not complete: {result['status']}")


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for stage in STAGES + ["artifact_flush_ms"]:
        values = [float(row[stage]) for row in rows]
        summary[stage] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "p95": float(np.percentile(values, 95)),
            "min": min(values), "max": max(values), "n": len(values),
        }
    differences = [
        row["total_analysis_ms"] - sum(row[key] for key in STAGES if key != "total_analysis_ms")
        for row in rows
    ]
    summary["orchestration_difference_ms"] = {
        "mean": statistics.fmean(differences), "note": "Total minus named stages; includes orchestration and logging."
    }
    return summary


def figures(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stage_names = STAGES[:-1]
    means = [statistics.fmean(row[key] for row in rows) for key in stage_names]
    stds = [statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0 for key in stage_names]
    labels = [name.removesuffix("_ms").replace("_", " ") for name in stage_names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, means, xerr=stds, color="#239CCB", capsize=3)
    ax.set(xlabel="Latency (ms)", title="Current session-level analysis latency breakdown")
    fig.tight_layout(); fig.savefig(output / "latency_breakdown.png", dpi=300); plt.close(fig)

    totals = [row["total_analysis_ms"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(totals, bins="auto", color="#3DDBAC", edgecolor="white")
    ax.axvline(np.median(totals), color="#F5B85E", label="Median")
    ax.axvline(np.percentile(totals, 95), color="#FF6577", label="P95")
    ax.set(xlabel="Total analysis latency (ms)", ylabel="Runs", title="Total analysis latency distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "total_latency_distribution.png", dpi=300); plt.close(fig)

    flush = [row["artifact_flush_ms"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.boxplot([totals, flush], tick_labels=["Decision\n(run)", "Persistence\n(flush)"])
    ax.set(ylabel="Latency (ms)", title="Time-to-decision vs. time-to-persisted artefacts")
    fig.tight_layout(); fig.savefig(output / "decision_vs_persistence.png", dpi=300); plt.close(fig)


def main() -> int:
    args = parser().parse_args()
    if args.warmup < 0 or args.runs < 1 or args.min_wbc < 1:
        raise SystemExit("warmup must be >= 0; runs and min-wbc must be >= 1")
    images = read_images(args.images)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "_fixed_inputs"
    prepare_inputs(images, inputs)
    pipeline = LeukemiaPipeline(load_config(args.config), device=args.device, mil_backend=args.mil_backend)
    try:
        validation = run_once(pipeline, inputs, output / "_validation", "benchmark_validation")
        validate(validation, args.min_wbc)
        for index in range(args.warmup):
            run_once(pipeline, inputs, output / "_run_workspace", f"warmup_{index:03d}")
        rows = []
        for index in range(args.runs):
            result = run_once(pipeline, inputs, output / "_run_workspace", f"measured_{index:04d}")
            validate(result, args.min_wbc)
            timing = {key: float(result["timing_ms"].get(key, 0.0)) for key in STAGES}
            rows.append({
                "run": index, **timing, "artifact_flush_ms": float(result["artifact_flush_ms"]),
                "number_of_fields": result["number_of_fields"],
                "number_of_wbc": result["number_of_detected_wbc"],
                "backend": result["inference_backend"], "final_result": result["final_label"],
                "label": result["label"], "tier": result["tier"],
                "number_of_classified_leukocytes": result["number_of_classified_leukocytes"],
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            })
    finally:
        pipeline.close()
    with (output / "timing_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "timing_summary.json", summarize(rows))
    manifest = build_manifest(images, validation, args.config)
    manifest["protocol"] = {"warmup_runs": args.warmup, "measured_runs": args.runs, "minimum_wbc": args.min_wbc}
    write_json(output / "benchmark_manifest.json", manifest)
    figures(rows, output)
    print(f"Completed {len(rows)} measured runs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
