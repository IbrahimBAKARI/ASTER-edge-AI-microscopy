# Latency / energy campaign — headless procedure (Jetson)

Localizer YOLO11n (prototype-field mixed replay), `imgsz 960`, `conf 0.18`.
Run once, with stable clocks, with the IDE and the browser closed. Every command
starts from the repository root and calls `.venv/bin/python` directly.

`scripts/run_campaign.sh` runs steps 0–4 in one go (about 25 min, asks for
`sudo` twice). The block-2 device campaign (409-patient re-scoring, stress test,
block-2 latency and energy) is `scripts/run_block2_v2_benchmarks.sh`.

---

## 0. Free the memory and pin the clocks

The full pipeline needs about 4 GB of free unified memory; with a desktop, an
IDE and a browser open, far less remains. Use a text console:

```bash
# from a TTY (Ctrl+Alt+F3) or an SSH session
sudo systemctl isolate multi-user.target      # stops the desktop
# or, without sudo, close the heavy applications by hand
```

Pin the clocks:

```bash
sudo nvpmodel -q                 # must print MAXN_SUPER / 2  (else: sudo nvpmodel -m 2)
sudo jetson_clocks               # CPU/GPU/EMC at maximum
sudo jetson_clocks --show | tee jetson_clocks_$(date +%Y%m%d).txt
```

Without `sudo`, skip this step: every manifest records
`clock_state.declaration = "jetson_clocks NOT engaged; DVFS active (declare when reporting)"`. In the
published localizer benchmark DVFS and pinned clocks agreed within 1 %
(`results/embedded_campaign/`).

Check memory and CUDA:

```bash
free -m | awk 'NR==2 {print "available:", $7, "MB (target >= 4000)"}'
.venv/bin/python -c "import torch; x=torch.zeros(int(1e9//4), device='cuda'); torch.cuda.synchronize(); print('CUDA OK', round(torch.cuda.mem_get_info()[0]/1e9,2), 'GB free')"
```

Restore at the end: `sudo jetson_clocks --restore` (or reboot) and
`sudo systemctl isolate graphical.target`.

## 1. TensorRT FP16 engines (on the device only)

```bash
scripts/export_yolo_trt.sh        # models/yolo/wbc_detector.{onnx,engine}, imgsz 960
scripts/build_block2_trt.sh       # models/mil/encoder_fp16.engine, static batch 8
```

`export_yolo_trt.sh` pins `numpy<2`, `onnx==1.16.2` and `ml-dtypes<0.5` before
and after the export (an unpinned `yolo export` pulls NumPy 2, which breaks the
Jetson torch 2.8 wheel). Without an engine the pipeline runs the `.pt` on CUDA
and records it in `warnings`.

## 2. Full-session latency, 100 runs

```bash
.venv/bin/python scripts/benchmark_current_pipeline.py --runs 100 --warmup 5 \
    --mil-backend tensorrt --output benchmarks/current_pipeline/timing
```

## 3. Isolated localizer and energy

```bash
.venv/bin/python scripts/benchmark_yolo_isolated.py --runs 100 --warmup 5
.venv/bin/python scripts/benchmark_energy.py --duration 300 --camera-index 0 \
    --output benchmarks/current_pipeline/energy_three_modes      # --skip-acquisition without a camera
```

Do not re-run `benchmark_yolo_isolated.py` with `--runs 1` afterwards: it
overwrites the 100-run JSON with a single cold-start point.

`benchmark_energy.py` reports, besides gross energy, `active_fraction`,
`net_power_w = P_full − P_idle` and
`net_energy_per_analysis_j = net_power_w × active_wall / analyses`.

## 4. Outputs

| File | Content |
|---|---|
| `timing/timing_summary.json` | stage latency, deployed configuration |
| `yolo_isolated/yolo_isolated.json` | PyTorch vs TensorRT FP16 at imgsz 960, box parity |
| `energy_three_modes/energy_three_modes_summary.json` | power in three modes, net energy |

Copy them into `results/` (layout in `results/README.md`).
