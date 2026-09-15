# Camera acquisition (Aster interface)

The interface captures every frame at the **camera's full native resolution** —
the same setting the prototype capture tool used to build the ×40 slide dataset.
For the Arducam IMX477 HQ that is **4032 × 3040** (12.3 MP).

## How it works

- The camera is opened MJPEG-first (`CAP_PROP_FOURCC` = `MJPG` set **before** the
  resolution — the IMX477 is MJPEG-only over USB and will not negotiate the
  high-resolution modes otherwise).
- The live view is a **downscaled preview** (`preview_max_width`, default 1280 px)
  so the UI and the live WBC detector stay responsive at ~10 fps.
- Each **Capture Frame** stores the most recent *full-resolution* frame to a
  per-session scratch folder (`Interface/_capture_cache/session_*/capture_NN.jpg`,
  JPEG q97). Only downscaled copies are kept in memory.
- Analysis reads the full-resolution files. "Save Result" copies them verbatim as
  `source_NN.jpg` (no re-encode); the annotated previews are saved downscaled as
  `sample_NN.jpg`.
- The scratch folder is deleted on Clear, on close, and swept at startup.
- The preview card is drawn at exactly the feed's aspect ratio (4:3 for the
  IMX477); any surrounding area matches the panel colour, so there is no
  letterbox fill and the visible feed equals the field that will be captured.

## Configuration — `Interface/camera_config.json`

| Key | Default | Meaning |
|---|---|---|
| `device` | `/dev/video0` | V4L2 device (override: `LEUKEMIA_CAMERA_DEVICE`) |
| `fourcc` | `MJPG` | pixel format four-character code |
| `capture_width` / `capture_height` | `4032` / `3040` | requested sensor resolution |
| `fps` | `10` | stream frame rate (12 MP MJPEG tops out near here) |
| `buffersize` | `1` | V4L2 buffer depth — keep at 1 for a fresh frame per capture |
| `preview_max_width` | `1280` | live-preview downscale width |
| `still_jpeg_quality` | `97` | quality of the saved full-resolution captures |

Point `LEUKEMIA_CAMERA_CONFIG` at another JSON file to override the whole set.
Valid IMX477 modes if you need a smaller one: 3840×2160, 2592×1944, 2560×1440,
1920×1080, 1280×960, 1280×720.

## CSI / Jetson Argus

If no USB V4L2 device is found, the interface falls back to
`nvarguscamerasrc` at the same configured resolution (needs an OpenCV build with
GStreamer support). Absent both, it runs a demo feed.

## Session size

Block 2 decides only from **100 classified leukocytes** (tier S, screening); it
applies the full pattern grid from 200 (tier P) and marks a reference-size count
at 400 (tier R) (`config/inference.yaml`). At the deployed x40 yield (~1.76 WBC
per field) the tiers need about **57, 114 and 227 fields**; the acquisition screen
shows the running estimate against these tiers. Below 100 classified leukocytes
the pipeline returns `insufficient_evidence`. Capture stops at 200 fields: a
larger session needs crop streaming to fit the device memory (see the memory
note below), so tier R is not reachable in one session yet.

## Memory note

A full-resolution frame is ~37 MB in memory. Only downscaled copies (longest side 1200 px, ~3 MB each) stay resident and the
full-resolution captures live on disk, so even a 200-field session is ~0.6 GB of
UI memory — but the analysis pipeline still needs ≈ 4 GB free
(`docs/ENVIRONMENT.md`). Close other applications before a large session.
