# Frozen environment — Jetson target

Captured 2026-09-08 on the target device. This file is the authoritative
environment record for every measured number in `docs/EVALUATION.md`. Do not
change the board, driver stack or power state during a measurement campaign.

## Host

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Orin Nano 8 GB — Engineering Reference Developer Kit *Super* |
| SoC | tegra234 (Orin, Ampere GPU, 1024 CUDA / 32 Tensor cores) |
| RAM | 7620 MiB unified LPDDR5 (shared CPU/GPU) + 3810 MiB zram swap |
| Storage | NVMe `/dev/nvme0n1`, ext4, `rw,nosuid,nodev,relatime` — workspace on this mount, 202 GB free |
| Kernel | `5.15.148-tegra #1 SMP PREEMPT Fri Jul 4 06:53:27 UTC 2025 aarch64` |

## JetPack / L4T / CUDA stack

| Component | Version |
|---|---|
| L4T / `nv_tegra_release` | **R36.4.7**, GCID 42132812, DATE Thu Sep 18 2025 (JetPack 6.2.x) |
| CUDA toolkit | 12.6 (`cuda-toolkit-12-6` 12.6.11-1) |
| cuDNN | 9.3.0.75-1 (CUDA 12.6) — `torch.backends.cudnn.version()` → 90300 |
| TensorRT | **10.3.0.30-1+cuda12.5** (`libnvinfer*`, `python3-libnvinfer`), `tensorrt` wheel 10.3.0 |
| `trtexec` | `/usr/src/tensorrt/bin/trtexec` (on PATH) |

## Python environment (`.venv`, Python 3.10.12)

| Package | Version |
|---|---|
| torch | 2.8.0 (built against CUDA 12.6, `torch.cuda.is_available()` → True) |
| torchvision | 0.23.0 |
| tensorrt | 10.3.0 (+ tensorrt_dispatch / tensorrt_lean 10.3.0) |
| ultralytics | 8.4.53 (+ ultralytics-thop 2.1.1) |
| onnx | 1.22.0 |
| opencv-python | 4.10.0.84 |
| numpy | 1.26.4 |
| scikit-learn | 1.6.1 |
| pillow | 12.3.0 |
| matplotlib | 3.10.8 |

Full list: `docs/requirements-frozen.txt` (`pip freeze`, 143 packages).

## Power / clock state

| Item | Value | Notes |
|---|---|---|
| `nvpmodel` | **`MAXN_SUPER`** (mode id 2) | max power budget preset for the Orin Nano Super |
| CPU cpufreq | governor `schedutil`, `scaling_max_freq` 1728000 kHz | under DVFS; pinned = 1728 MHz Min==Max |
| GPU devfreq | governor `nvhost_podgov` (dynamic) | under DVFS; pinned = 1020 MHz Min==Max |
| Fan | `pwm-fan` (auto) | |
| Idle draw (tegrastats, no load) | VDD_IN ≈ 5.4–5.5 W | |
| Idle temps | ~50 °C all zones | |

> **Clock state.** The isolated localizer benchmark was run twice — under MAXN_SUPER with
> **dynamic frequency scaling** (schedutil + podgov) and under
> `sudo jetson_clocks` (**clocks pinned** to max: CPU 6×1728 MHz, GPU 1020 MHz,
> EMC 3199 MHz) — and the two agree to **< 1 %**
> (`results/embedded_campaign/`, `docs/EVALUATION.md` §1.6). Each benchmark
> manifest records its own `clock_state`. Without `sudo` only the DVFS run is
> possible.

## tegrastats sampling

`tegrastats` runs without root and exposes VDD_IN (total board), VDD_CPU_GPU_CV,
VDD_SOC in mW, RAM, SWAP, per-core CPU load and freq, GR3D (GPU) load, and
thermal zones. Sampling interval: 1000 ms.

## Free-memory constraint

The full pipeline (YOLO, block-2 encoder, torch/cuDNN context) needs **≈ 4 GB
free** of unified memory. Close the IDE and browser before running the
full-pipeline benchmarks (`scripts/HEADLESS_CAMPAIGN.md`).
