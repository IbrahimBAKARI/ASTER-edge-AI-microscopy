"""Shared, non-clinical helpers for the opt-in ASTER benchmark commands."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aster_pipeline.config import load_config


def read_images(list_file: str | Path) -> list[Path]:
    source = Path(list_file).expanduser().resolve()
    paths = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Benchmark image does not exist: {path}")
            paths.append(path)
    if not paths:
        raise ValueError(f"No images listed in {source}")
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        value = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return (value.stdout or value.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _clocks_pinned(show: str | None) -> bool | None:
    """True/False from a `jetson_clocks --show` dump, or None if undetermined.

    Pinned = per-core CPU MinFreq == MaxFreq *and* the GPU MinFreq == MaxFreq
    (`jetson_clocks` clamps the frequency range; it leaves the governor label as
    `schedutil`, so the governor name is not a reliable signal).
    """
    if not show or "Online" not in show:
        return None
    import re
    cpu_ok = bool(re.search(r"cpu\d+:.*MinFreq=(\d+) MaxFreq=\1 ", show))
    gpu = re.search(r"GPU MinFreq=(\d+) MaxFreq=(\d+)", show)
    gpu_ok = bool(gpu) and gpu.group(1) == gpu.group(2)
    return cpu_ok and gpu_ok


def clock_state() -> dict:
    """Record the Jetson power mode and whether the clocks are pinned.

    ``jetson_clocks --show`` needs root, so in an unprivileged run the pinned
    state cannot be read back from the OS. ``run_campaign.sh`` saves the real
    (sudo) dump and points ``ASTER_CLOCKS_STATE_FILE`` at it; otherwise
    ``clocks_pinned`` is ``null`` (undetermined) rather than asserted either way.
    """
    show = command_output(["jetson_clocks", "--show"])
    saved = os.environ.get("ASTER_CLOCKS_STATE_FILE")
    if (not show or "Online" not in show) and saved and Path(saved).is_file():
        show = Path(saved).read_text(errors="replace")
    governors = command_output(
        ["bash", "-c", "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | paste -sd,"]
    )
    pinned = _clocks_pinned(show)
    return {
        "nvpmodel": command_output(["nvpmodel", "-q"]),
        "jetson_clocks_show": show,
        "cpu_governors": governors,
        "clocks_pinned": pinned,
        "declaration": {
            True: "jetson_clocks engaged; CPU+GPU clocks pinned",
            False: "jetson_clocks NOT engaged; DVFS active (declare when reporting)",
            None: "clock state undetermined (jetson_clocks --show needs root; no saved dump)",
        }[pinned],
    }


def build_manifest(images: list[Path], result: dict, config_path: str | Path) -> dict:
    config = load_config(config_path)
    model_paths = [config.yolo.weights]
    if config.yolo.tensorrt_weights:
        model_paths.append(config.yolo.tensorrt_weights)
    block2 = config.block2
    if block2 is not None:
        model_paths.append(block2.encoder_onnx)
        if block2.encoder_engine.is_file():
            model_paths.append(block2.encoder_engine)
        model_paths.append(block2.heads)
    return {
        "session_description": "technical benchmark session assembled from previously evaluated microscopy fields",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "images": [{"path": str(path), "sha256": sha256(path)} for path in images],
        "number_of_fields": len(images),
        "number_of_detected_wbc": result["number_of_detected_wbc"],
        "models": [{"path": str(path), "sha256": sha256(path)} for path in model_paths if path.is_file()],
        "block2_bundle": (
            {
                "grid": {"path": str(block2.grid), "sha256": sha256(block2.grid)},
                "thresholds": {"path": str(block2.thresholds), "sha256": sha256(block2.thresholds)},
                "ood_stats": {"path": str(block2.ood_stats), "sha256": sha256(block2.ood_stats)},
                "quantifier": {"path": str(block2.quantifier), "sha256": sha256(block2.quantifier)},
            }
            if block2 is not None
            else None
        ),
        "yolo_parameters": {
            "confidence": config.yolo.confidence, "iou": config.yolo.iou,
            "image_size": config.yolo.image_size, "padding": config.yolo.padding,
            "class_id": config.yolo.class_id,
        },
        "block2_parameters": (
            {
                "bag_size": block2.bag_size, "seed": block2.seed, "gamma": block2.gamma,
                "min_classified_cells_screening": block2.min_classified_cells_screening,
                "min_classified_cells_pattern": block2.min_classified_cells_pattern,
                "min_classified_cells_reference": block2.min_classified_cells_reference,
                "thresholds_and_grid": "frozen decision grid + resolved thresholds, hashed above",
            }
            if block2 is not None
            else None
        ),
        "backend": result.get("inference_backend"),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "jetpack": command_output(["dpkg-query", "-W", "nvidia-jetpack"]),
            "cuda": command_output(["nvcc", "--version"]),
            "tensorrt": command_output(["dpkg-query", "-W", "libnvinfer8"]),
            "pytorch": _torch_version(),
            "power_mode": command_output(["nvpmodel", "-q"]),
            "clock_state": clock_state(),
        },
    }


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
