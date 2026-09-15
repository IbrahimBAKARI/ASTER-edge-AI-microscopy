import json
import os
import platform
import shutil
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter

# This file is launched directly (``python3 Interface/leukemia_ui.py``).  The
# self-contained ASTER backend, the ``aster_pipeline``/``YOLO`` packages and the
# ``Interface`` package itself all live at the project root, which is not on
# ``sys.path`` when a script is run by path.  Add it (and this directory) so the
# local model integration and live-guidance modules resolve to this project.
_INTERFACE_DIR = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
for _extra_path in (_PROJECT_ROOT, _INTERFACE_DIR):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

# ``opencv-python`` bundles its own Qt plugins.  Mixing that copy with PyQt5
# makes Qt discover an incompatible xcb plugin on Ubuntu/Jetson.  Prefer the
# platform plugins installed with the system PyQt package, and set this before
# ``cv2`` is imported since importing it can otherwise win the discovery race.
SYSTEM_QT_PLATFORM_PLUGINS = Path("/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms")


def _configure_qt_platform_plugins() -> None:
    if SYSTEM_QT_PLATFORM_PLUGINS.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(SYSTEM_QT_PLATFORM_PLUGINS)


_configure_qt_platform_plugins()

import cv2
import numpy as np
from PyQt5.QtCore import QRectF, QThread, QTimer, Qt, pyqtProperty, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QStyle,
)

try:
    from model_integration import build_workspace_model_runner
    MODEL_INTEGRATION_IMPORT_ERROR = None
except Exception as exc:
    build_workspace_model_runner = None
    MODEL_INTEGRATION_IMPORT_ERROR = exc


INTERFACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = INTERFACE_ROOT.parent
SAVED_RESULTS_DIR = INTERFACE_ROOT / "saved_results"
CAPTURE_CACHE_DIR = INTERFACE_ROOT / "_capture_cache"


def _load_camera_config():
    """Camera acquisition settings. Default: the Arducam IMX477 HQ at its full
    native resolution (4032x3040), the same setting the prototype capture tool
    used to build the slide dataset. Override the whole file with
    ``LEUKEMIA_CAMERA_CONFIG`` or just the device with ``LEUKEMIA_CAMERA_DEVICE``.
    """
    cfg = {
        "device": "/dev/video0",
        "fourcc": "MJPG",
        "capture_width": 4032,
        "capture_height": 3040,
        "fps": 10,
        "buffersize": 1,
        "preview_max_width": 1280,
        "still_jpeg_quality": 97,
    }
    path = os.environ.get("LEUKEMIA_CAMERA_CONFIG", str(INTERFACE_ROOT / "camera_config.json"))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cfg.update({k: v for k, v in json.load(handle).items() if not k.startswith("_")})
    except (OSError, ValueError):
        pass
    device_override = os.environ.get("LEUKEMIA_CAMERA_DEVICE")
    if device_override:
        cfg["device"] = device_override
    return cfg


CAMERA_CONFIG = _load_camera_config()


COLORS = {
    "bg_start": "#08111A",
    "bg_end": "#0D1723",
    "panel": "rgba(17, 28, 42, 0.96)",
    "panel_soft": "rgba(22, 37, 54, 0.96)",
    "border": "rgba(138, 156, 176, 0.18)",
    "border_strong": "rgba(118, 176, 206, 0.30)",
    "text": "#F3F7FA",
    "muted": "#9DAFC2",
    "accent": "#78B8D6",
    "accent_alt": "#4D93B8",
    "success": "#69A68E",
    "warning": "#D59A59",
    "danger": "#D66969",
    "danger_alt": "#DE7A6A",
}

BADGE_COLORS = {
    "neutral": {"bg": "rgba(120, 184, 214, 0.08)", "border": "rgba(120, 184, 214, 0.22)", "text": COLORS["accent"]},
    "success": {"bg": "rgba(105, 166, 142, 0.10)", "border": "rgba(105, 166, 142, 0.24)", "text": COLORS["success"]},
    "warning": {"bg": "rgba(213, 154, 89, 0.10)", "border": "rgba(213, 154, 89, 0.24)", "text": COLORS["warning"]},
    "danger": {"bg": "rgba(214, 105, 105, 0.12)", "border": "rgba(214, 105, 105, 0.26)", "text": COLORS["danger"]},
}


def _scaled_px(value, scale, minimum=1):
    return max(minimum, int(round(float(value) * float(scale))))


def _widget_scale(widget):
    if widget is None:
        return 1.0
    top_level = widget.window()
    return float(getattr(top_level, "_style_scale", 1.0))


def load_claim_tiers():
    """Claim tiers of block 2 (classified leukocytes), read from config/inference.yaml.

    Tier S (screening) is the minimum for any decision; below it block 2 returns
    insufficient_evidence. Falls back to the frozen values if the file cannot be read.
    """
    tiers = {"screening": 100, "pattern": 200, "reference": 400}
    try:
        import yaml

        cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "inference.yaml").read_text())
        block2 = cfg.get("block2", cfg)
        for name in tiers:
            value = block2.get(f"min_classified_cells_{name}")
            if value is not None:
                tiers[name] = int(value)
    except Exception:
        pass
    return ((tiers["screening"], "S", "screening"), (tiers["pattern"], "P", "pattern"),
            (tiers["reference"], "R", "reference"))


def build_default_model_runner():
    try:
        if build_workspace_model_runner is None:
            raise RuntimeError(str(MODEL_INTEGRATION_IMPORT_ERROR or "Model integration is unavailable."))
        return build_workspace_model_runner()
    except Exception as exc:
        message = str(exc)

        def unavailable_runner(images):
            return {
                "label": "Integration Unavailable",
                "confidence": 0,
                "risk_level": "warning",
                "images_used": len(images),
                "summary": f"Integrated model loading failed: {message}",
                "mode": "full_image_detection",
                "processed_images": [],
                "total_detected_cells": 0,
                "positive_cells": 0,
                "max_positive_probability": None,
            }

        return unavailable_runner


def numpy_to_qimage(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    return QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def rounded_pixmap_from_frame(frame, width, height, radius=20):
    pixmap = QPixmap.fromImage(numpy_to_qimage(frame))
    scaled = pixmap.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    target = QPixmap(width, height)
    target.fill(Qt.transparent)

    painter = QPainter(target)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, width, height), radius, radius)
    painter.setClipPath(path)
    x_pos = (width - scaled.width()) // 2
    y_pos = (height - scaled.height()) // 2
    painter.drawPixmap(x_pos, y_pos, scaled)
    painter.end()
    return target


def set_badge(label, text, tone="neutral"):
    palette = BADGE_COLORS[tone]
    scale = _widget_scale(label)
    radius = _scaled_px(14, scale, 8)
    padding_v = _scaled_px(7, scale, 4)
    padding_h = _scaled_px(14, scale, 8)
    font_size = _scaled_px(12, scale, 10)
    label.setText(text)
    label.setProperty("badge_text", text)
    label.setProperty("badge_tone", tone)
    label.setStyleSheet(
        f"""
        QLabel {{
            color: {palette['text']};
            background-color: {palette['bg']};
            border: 1px solid {palette['border']};
            border-radius: {radius}px;
            padding: {padding_v}px {padding_h}px;
            font-size: {font_size}px;
            font-weight: 600;
        }}
        """
    )


def default_model_inference(images):
    """
    Replace this mock with the production TensorRT/PyTorch inference function.
    It returns a UI-ready payload so the front end stays stable during model swaps.
    """
    if not images:
        return {
            "label": "No Samples",
            "confidence": 0,
            "risk_level": "low",
            "images_used": 0,
            "summary": "Capture or import microscopy images before running analysis.",
        }

    focus_scores = []
    color_scores = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        color_scores.append(float(np.mean(image[:, :, 2]) / max(np.mean(image[:, :, 1]), 1.0)))

    focus_norm = min(float(np.mean(focus_scores)) / 650.0, 1.0)
    color_norm = min(float(np.mean(color_scores)) / 1.45, 1.0)
    sample_norm = min(len(images) / 20.0, 1.0)
    risk_score = 0.42 * focus_norm + 0.33 * color_norm + 0.25 * sample_norm

    aml_suspected = risk_score >= 0.67
    confidence = int(np.clip(64 + risk_score * 34, 64, 98))

    if aml_suspected:
        return {
            "label": "AML Suspected",
            "confidence": confidence,
            "risk_level": "high",
            "images_used": len(images),
            "summary": "Pattern exceeds the alert threshold. Recommend expert review and confirmatory testing.",
        }

    return {
        "label": "Low AML Likelihood",
        "confidence": confidence,
        "risk_level": "low",
        "images_used": len(images),
        "summary": "Observed morphology remains below the current AML alert threshold for this session.",
    }

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=None,
    capture_height=None,
    display_width=1280,
    display_height=960,
    framerate=None,
    flip_method=0,
):
    capture_width = capture_width or CAMERA_CONFIG["capture_width"]
    capture_height = capture_height or CAMERA_CONFIG["capture_height"]
    framerate = framerate or CAMERA_CONFIG["fps"]
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


class CameraThread(QThread):
    """Streams a downscaled preview and keeps the latest full-resolution frame.

    ``frame_ready`` carries a small preview frame (``preview_max_width`` wide) so
    the UI and live detector stay responsive. ``grab_full()`` returns the most
    recent full-resolution frame for capture, so every saved image is at the
    camera's native resolution (4032x3040 for the Arducam IMX477 HQ).
    """

    frame_ready = pyqtSignal(object)
    camera_status = pyqtSignal(bool, str)

    def __init__(self, sensor_id=0, fps=None, parent=None):
        super().__init__(parent)
        self.sensor_id = sensor_id
        stream_fps = fps or CAMERA_CONFIG["fps"]
        self.frame_interval_ms = max(int(1000 / max(stream_fps, 1)), 25)
        self.running = False
        self._mock_tick = 0
        self._last_status = None
        self._full_lock = threading.Lock()
        self._full_frame = None

    def _open_camera(self):
        """Open USB/V4L2 microscopes before falling back to Jetson CSI cameras.

        USB microscopes (including the attached Arducam IMX477 HQ) are exposed as
        ``/dev/video*`` V4L2 devices; genuine Jetson CSI sensors need Argus. The
        Arducam is MJPEG-only over USB, so the FOURCC must be set before the
        resolution or the high-resolution modes are never negotiated.
        """
        cfg = CAMERA_CONFIG
        device = cfg["device"]
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["capture_width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["capture_height"])
            capture.set(cv2.CAP_PROP_FPS, cfg["fps"])
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, cfg["buffersize"])
            except Exception:
                pass
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg["capture_width"]
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg["capture_height"]
            return capture, f"Microscope feed online ({device}, V4L2, {width}x{height})"
        capture.release()

        pipeline = gstreamer_pipeline(sensor_id=self.sensor_id)
        print("Trying CSI pipeline:")
        print(pipeline)
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if capture.isOpened():
            return capture, (
                f"Microscope feed online (CSI/Argus, "
                f"{cfg['capture_width']}x{cfg['capture_height']})"
            )
        capture.release()
        return None, (
            f"Camera unavailable: could not open {device} via V4L2 or a CSI/Argus sensor"
        )

    def _publish(self, frame):
        """Store the full frame and emit a downscaled preview."""
        with self._full_lock:
            self._full_frame = frame
        max_width = CAMERA_CONFIG["preview_max_width"]
        if frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            preview = cv2.resize(
                frame,
                (max_width, max(1, int(round(frame.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = frame.copy()
        self.frame_ready.emit(preview)

    def grab_full(self):
        """The most recent full-resolution frame (BGR), or ``None``."""
        with self._full_lock:
            return None if self._full_frame is None else self._full_frame.copy()

    def run(self):
        self.running = True
        capture, online_message = self._open_camera()

        if capture is None:
            self._emit_status(False, online_message + " - using demo feed")
            while self.running:
                self._publish(self._generate_mock_frame())
                self.msleep(self.frame_interval_ms)
            return

        self._emit_status(True, online_message)
        while self.running:
            success, frame = capture.read()
            if success and frame is not None:
                self._emit_status(True, online_message)
                self._publish(frame)
            else:
                self._emit_status(False, "Camera frame unavailable - using demo feed")
                self._publish(self._generate_mock_frame())
            self.msleep(self.frame_interval_ms)
        capture.release()

    def stop(self):
        self.running = False
        self.wait(1500)

    def _emit_status(self, online, message):
        payload = (online, message)
        if payload != self._last_status:
            self._last_status = payload
            self.camera_status.emit(online, message)

    def _generate_mock_frame(self):
        self._mock_tick += 1
        height, width = 960, 1280  # 4:3, matching the IMX477 native aspect ratio
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        x_gradient = np.tile(np.linspace(18, 48, width, dtype=np.uint8), (height, 1))
        y_gradient = np.tile(np.linspace(10, 42, height, dtype=np.uint8).reshape(height, 1), (1, width))
        frame[:, :, 0] = x_gradient
        frame[:, :, 1] = y_gradient
        frame[:, :, 2] = 22

        glow = np.zeros_like(frame)
        center_x = int(width * 0.52 + np.sin(self._mock_tick / 10.0) * 70)
        center_y = int(height * 0.48 + np.cos(self._mock_tick / 12.0) * 35)
        cv2.circle(glow, (center_x, center_y), 180, (58, 198, 255), -1)
        cv2.circle(glow, (int(width * 0.35), int(height * 0.63)), 110, (0, 182, 147), -1)
        cv2.addWeighted(glow, 0.18, frame, 0.82, 0, frame)

        for x_pos in range(120, width, 180):
            cv2.line(frame, (x_pos, 0), (x_pos, height), (42, 78, 110), 1)
        for y_pos in range(80, height, 120):
            cv2.line(frame, (0, y_pos), (width, y_pos), (42, 78, 110), 1)

        cv2.rectangle(frame, (290, 160), (990, 560), (94, 230, 255), 2)
        cv2.putText(frame, "DEMO FEED", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (94, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Microscope camera not detected",
            (90, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (198, 226, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            datetime.now().strftime("%H:%M:%S"),
            (1040, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (198, 226, 245),
            2,
            cv2.LINE_AA,
        )
        return frame

class InferenceWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def __init__(self, images, model_runner=None, parent=None):
        super().__init__(parent)
        # images are either file paths (full-resolution captures on disk) or
        # in-memory frames; copy the frames, pass the paths through untouched.
        self.images = [
            img.copy() if isinstance(img, np.ndarray) else os.fspath(img)
            for img in images
        ]
        self.model_runner = model_runner or default_model_inference
        self.running = True

    def run(self):
        if not self.running:
            return

        self._emit_progress(4, f"Preparing {len(self.images)} image(s) for analysis...")
        try:
            if hasattr(self.model_runner, "analyze_full_images"):
                result = self.model_runner.analyze_full_images(
                    self.images,
                    progress_callback=self._emit_progress,
                )
            else:
                result = self.model_runner(self.images)
        except Exception as exc:
            result = {
                "label": "Analysis Failed",
                "confidence": 0,
                "risk_level": "warning",
                "images_used": len(self.images),
                "summary": str(exc),
                "mode": "full_image_detection",
                "processed_images": [],
                "total_detected_cells": 0,
                "positive_cells": 0,
                "max_positive_probability": None,
            }
        self._emit_progress(100, "Analysis complete")
        self.finished.emit(result)

    def _emit_progress(self, progress, detail):
        if not self.running:
            return
        safe_progress = max(0, min(100, int(progress)))
        self.progress_changed.emit(safe_progress, str(detail))

    def stop(self):
        self.running = False
        self.wait(1500)


class MetricTile(QFrame):
    def __init__(self, title, compact=False, parent=None):
        super().__init__(parent)
        self.setObjectName("metricTile")
        self.compact = compact
        self._note_text = ""
        self.setMinimumHeight(68 if compact else 92)
        layout = QVBoxLayout(self)
        if compact:
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(2)
        else:
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        self.value_label.setWordWrap(True)
        self.note_label = QLabel("")
        self.note_label.setObjectName("metricNote")
        self.note_label.setWordWrap(not compact)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.note_label)
        self.note_label.setVisible(not compact)

    def set_data(self, value, note):
        self.value_label.setText(value)
        self._note_text = note
        self.setToolTip(note)
        self.title_label.setToolTip(note)
        self.value_label.setToolTip(note)
        self.note_label.setToolTip(note)
        if self.compact:
            self.note_label.clear()
        else:
            self._refresh_note_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.compact:
            self._refresh_note_label()

    def _refresh_note_label(self):
        if self.compact:
            return
        available_width = max(self.note_label.width(), 0)
        if not self._note_text or available_width <= 0:
            self.note_label.setText(self._note_text)
            return
        self.note_label.setText(self._note_text)

    def set_compact_mode(self, compact):
        compact = bool(compact)
        if self.compact == compact:
            return
        self.compact = compact
        self.note_label.setVisible(not compact)
        if compact:
            self.note_label.clear()
        else:
            self._refresh_note_label()


class NeonButton(QPushButton):
    def __init__(self, text, variant="primary", glow_color=None, parent=None):
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setProperty("pulse", "false")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(52 if variant == "primary" else 46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def pulse(self):
        self.setProperty("pulse", "true")
        self.style().unpolish(self)
        self.style().polish(self)
        QTimer.singleShot(240, self._clear_pulse)

    def _clear_pulse(self):
        self.setProperty("pulse", "false")
        self.style().unpolish(self)
        self.style().polish(self)


class PreviewSurface(QWidget):
    # Default to the Arducam IMX477 HQ native aspect ratio; updated from the
    # first real frame. The preview card is drawn at exactly this ratio so the
    # visible feed matches the field that will be captured - no letterbox fill.
    DEFAULT_ASPECT = 4032 / 3040

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(360)
        self.current_image = None
        self.camera_online = True
        self.badge_override_text = None
        self.badge_override_tone = "neutral"
        self.flash_value = 0.0
        self._aspect = self.DEFAULT_ASPECT
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(28)
        self.flash_timer.timeout.connect(self._decay_flash)

    def set_frame(self, frame):
        image = numpy_to_qimage(frame)
        if image.width() > 0 and image.height() > 0:
            self._aspect = image.width() / image.height()
        self.current_image = image
        self.update()

    def _image_rect(self):
        """Largest rectangle of the feed's aspect ratio, centred in the widget."""
        area = QRectF(self.rect().adjusted(1, 1, -1, -1))
        if area.height() <= 0 or self._aspect <= 0:
            return area
        if area.width() / area.height() > self._aspect:
            height = area.height()
            width = height * self._aspect
        else:
            width = area.width()
            height = width / self._aspect
        left = area.left() + (area.width() - width) / 2
        top = area.top() + (area.height() - height) / 2
        return QRectF(left, top, width, height)

    def set_camera_online(self, online):
        self.camera_online = online
        self.update()

    def set_status_badge(self, text=None, tone="neutral"):
        self.badge_override_text = text
        self.badge_override_tone = tone
        self.update()

    def trigger_flash(self):
        self.flash_value = 0.45
        self.flash_timer.start()
        self.update()

    def get_flash_value(self):
        return self.flash_value

    def set_flash_value(self, value):
        self.flash_value = value
        self.update()

    flashValue = pyqtProperty(float, fget=get_flash_value, fset=set_flash_value)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Everything outside the feed rectangle is the panel-card colour, so the
        # preview reads as a card sized exactly to the camera field.
        painter.fillRect(self.rect(), QColor(17, 28, 42))

        rect = self._image_rect()
        radius = max(10.0, min(22.0, rect.height() * 0.05))
        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, radius, radius)
        painter.fillPath(clip_path, QColor(COLORS["bg_start"]))

        painter.save()
        painter.setClipPath(clip_path)
        if self.current_image is not None:
            pixmap = QPixmap.fromImage(self.current_image)
            scaled = pixmap.scaled(
                rect.size().toSize(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            painter.drawPixmap(
                int(rect.left() + (rect.width() - scaled.width()) / 2),
                int(rect.top() + (rect.height() - scaled.height()) / 2),
                scaled,
            )

        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QColor(6, 12, 25, 150))
        gradient.setColorAt(0.5, QColor(6, 12, 25, 10))
        gradient.setColorAt(1.0, QColor(6, 12, 25, 185))
        painter.fillRect(rect, gradient)
        painter.restore()

        painter.setPen(QPen(QColor(COLORS["border_strong"]), 1.6))
        painter.drawPath(clip_path)
        self._draw_focus_brackets(painter, rect)
        self._draw_status_badge(painter, rect)

        if self.flash_value > 0:
            flash_color = QColor(255, 255, 255, int(self.flash_value * 120))
            painter.fillPath(clip_path, flash_color)
        painter.end()

    def _draw_focus_brackets(self, painter, rect):
        box_width = min(rect.width() * 0.22, 210)
        box_height = min(rect.height() * 0.28, 190)
        center_x = rect.center().x()
        center_y = rect.center().y()
        left = center_x - box_width / 2
        top = center_y - box_height / 2
        right = center_x + box_width / 2
        bottom = center_y + box_height / 2
        render_scale = min(rect.width() / 640.0, rect.height() / 360.0, 1.0)
        corner = max(14, int(round(26 * render_scale)))

        pen = QPen(QColor(COLORS["accent"]), max(1.6, 3.0 * render_scale))
        painter.setPen(pen)
        painter.drawLine(int(left), int(top + corner), int(left), int(top))
        painter.drawLine(int(left), int(top), int(left + corner), int(top))
        painter.drawLine(int(right - corner), int(top), int(right), int(top))
        painter.drawLine(int(right), int(top), int(right), int(top + corner))
        painter.drawLine(int(left), int(bottom - corner), int(left), int(bottom))
        painter.drawLine(int(left), int(bottom), int(left + corner), int(bottom))
        painter.drawLine(int(right - corner), int(bottom), int(right), int(bottom))
        painter.drawLine(int(right), int(bottom - corner), int(right), int(bottom))

    def _draw_status_badge(self, painter, rect):
        render_scale = min(rect.width() / 640.0, rect.height() / 360.0, 1.0)
        badge_x = int(rect.left()) + max(12, int(round(26 * render_scale)))
        badge_y = int(rect.top()) + max(10, int(round(24 * render_scale)))
        badge_width = max(84, int(round(118 * render_scale)))
        badge_height = max(24, int(round(34 * render_scale)))
        badge_radius = max(10, int(round(17 * render_scale)))
        badge_rect = QRectF(badge_x, badge_y, badge_width, badge_height)
        if self.badge_override_text:
            palette = BADGE_COLORS.get(self.badge_override_tone, BADGE_COLORS["neutral"])
            line_color = QColor(palette["text"])
            fill_color = QColor(palette["bg"])
            text = self.badge_override_text
        else:
            line_color = QColor(COLORS["success"]) if self.camera_online else QColor(COLORS["danger"])
            fill_color = QColor(line_color)
            fill_color.setAlpha(26 if self.camera_online else 34)
            text = "LIVE FEED" if self.camera_online else "DEMO FEED"

        painter.setPen(QPen(line_color, 1.0))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(badge_rect, badge_radius, badge_radius)
        painter.setPen(line_color)
        font = painter.font()
        font.setPointSizeF(max(9.0, 10.5 * render_scale))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge_rect, Qt.AlignCenter, text)

    def _decay_flash(self):
        self.flash_value = max(0.0, self.flash_value - 0.08)
        if self.flash_value <= 0.0:
            self.flash_timer.stop()
        self.update()


class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_data = None
        self.preview_title = ""
        self.setCursor(Qt.ArrowCursor)
        self.setToolTip("Click to enlarge")

    def set_preview_payload(self, frame, title):
        self.frame_data = None if frame is None else frame.copy()
        self.preview_title = title or "Image Preview"
        self.setCursor(Qt.PointingHandCursor if self.frame_data is not None else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.frame_data is not None:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class ImageLightboxDialog(QDialog):
    def __init__(self, gallery_items, start_index=0, parent=None):
        super().__init__(parent)
        self.setObjectName("imageLightboxDialog")
        self.setModal(True)
        self.resize(1080, 760)
        self.setMinimumSize(320, 240)
        self._gallery_items = [
            (frame.copy(), title or f"Image {index + 1:02d}")
            for index, (frame, title) in enumerate(gallery_items)
            if frame is not None
        ]
        self._current_index = 0
        self._pixmap_cache = {}
        if self._gallery_items:
            self._current_index = max(0, min(start_index, len(self._gallery_items) - 1))
        self.setWindowTitle(
            self._gallery_items[self._current_index][1]
            if self._gallery_items
            else "Image Preview"
        )

        self.setStyleSheet(
            f"""
            QDialog#imageLightboxDialog {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {COLORS['bg_start']},
                    stop: 1 {COLORS['bg_end']}
                );
            }}

            QLabel#lightboxTitle {{
                color: {COLORS['text']};
                font-size: 20px;
                font-weight: 700;
            }}

            QLabel#lightboxHint {{
                color: {COLORS['muted']};
                font-size: 12px;
            }}

            QLabel#lightboxCounter {{
                color: {COLORS['accent']};
                font-size: 12px;
                font-weight: 700;
                padding: 6px 10px;
                border-radius: 12px;
                background: rgba(120, 184, 214, 0.08);
                border: 1px solid rgba(120, 184, 214, 0.18);
            }}

            QLabel#lightboxImage {{
                background: {COLORS['panel']};
                border: 1px solid {COLORS['border_strong']};
                border-radius: 24px;
            }}

            QPushButton#lightboxNavButton {{
                min-width: 52px;
                max-width: 52px;
                min-height: 56px;
                max-height: 56px;
                font-size: 28px;
                font-weight: 800;
                padding: 0px;
                border-radius: 18px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        self.title_label = QLabel("Image Preview")
        self.title_label.setObjectName("lightboxTitle")
        self.counter_label = QLabel("")
        self.counter_label.setObjectName("lightboxCounter")
        self.prev_button = QPushButton("‹")
        self.prev_button.setObjectName("lightboxNavButton")
        self.prev_button.setCursor(Qt.PointingHandCursor)
        self.prev_button.setToolTip("Previous image")
        self.prev_button.clicked.connect(self._show_previous_image)
        self.next_button = QPushButton("›")
        self.next_button.setObjectName("lightboxNavButton")
        self.next_button.setCursor(Qt.PointingHandCursor)
        self.next_button.setToolTip("Next image")
        self.next_button.clicked.connect(self._show_next_image)
        close_button = QPushButton("Close")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.clicked.connect(self.accept)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.counter_label)
        header_layout.addWidget(close_button)

        hint_label = QLabel("Use the side arrows or the left/right keys to browse without closing the viewer.")
        hint_label.setObjectName("lightboxHint")
        self.image_label = QLabel()
        self.image_label.setObjectName("lightboxImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 160)

        image_row = QHBoxLayout()
        image_row.setSpacing(14)
        image_row.addWidget(self.prev_button, 0, Qt.AlignVCenter)
        image_row.addWidget(self.image_label, 1)
        image_row.addWidget(self.next_button, 0, Qt.AlignVCenter)

        layout.addLayout(header_layout)
        layout.addWidget(hint_label)
        layout.addLayout(image_row, 1)
        QTimer.singleShot(0, self._fit_to_available_screen)
        self._show_current_image()

    def _fit_to_available_screen(self):
        app = QApplication.instance()
        screen = self.parentWidget().window().screen() if self.parentWidget() is not None else None
        if screen is None and app is not None:
            screen = app.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        target_width = min(1080, max(320, available.width() - 24))
        target_height = min(760, max(240, available.height() - 24))
        self.resize(target_width, target_height)
        self.move(
            available.x() + max((available.width() - target_width) // 2, 0),
            available.y() + max((available.height() - target_height) // 2, 0),
        )

    def _pixmap_for_index(self, index):
        if index not in self._pixmap_cache:
            frame, _ = self._gallery_items[index]
            self._pixmap_cache[index] = QPixmap.fromImage(numpy_to_qimage(frame))
        return self._pixmap_cache[index]

    def _show_current_image(self):
        if not self._gallery_items:
            self.image_label.clear()
            self.title_label.setText("Image Preview")
            self.counter_label.setText("0 / 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        frame_title = self._gallery_items[self._current_index][1]
        self.setWindowTitle(frame_title)
        self.title_label.setText(frame_title)
        self.counter_label.setText(
            f"{self._current_index + 1} / {len(self._gallery_items)}"
        )

        pixmap = self._pixmap_for_index(self._current_index)
        available_size = self.image_label.contentsRect().size()
        if available_size.width() > 0 and available_size.height() > 0:
            scaled = pixmap.scaled(
                available_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setPixmap(pixmap)
        self.prev_button.setEnabled(self._current_index > 0)
        self.next_button.setEnabled(self._current_index < len(self._gallery_items) - 1)

    def _show_previous_image(self):
        if self._current_index <= 0:
            return
        self._current_index -= 1
        self._show_current_image()

    def _show_next_image(self):
        if self._current_index >= len(self._gallery_items) - 1:
            return
        self._current_index += 1
        self._show_current_image()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_PageUp):
            self._show_previous_image()
            event.accept()
            return
        if event.key() in (Qt.Key_Right, Qt.Key_PageDown, Qt.Key_Space):
            self._show_next_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_current_image()


class LeukemiaDetectionUI(QMainWindow):
    def __init__(self, model_runner=None, camera_source=0):
        super().__init__()
        self.page_acquisition = 0
        self.page_result = 1
        self.page_saved_results = 2
        self.model_runner = model_runner or build_default_model_runner()
        self.camera_source = camera_source
        # Block 2 decides only from tier S (>= 100 classified leukocytes); tier P
        # (>= 200) applies the full pattern grid and tier R (>= 400) marks a
        # reference-size count (config/inference.yaml). At the deployed x40 yield
        # (~WBC_PER_FIELD localized WBC per field) the tiers need about 57 / 114 / 227
        # fields. MAX_SESSION_IMAGES bounds one session by device memory: larger
        # sessions need crop streaming (not implemented), so tier R is not reachable.
        self.minimum_images = 1
        self.wbc_per_field = 1.76
        self.claim_tiers = load_claim_tiers()
        self.tier_fields = [int(round(n / self.wbc_per_field)) for n, _, _ in self.claim_tiers]  # 57 / 114 / 227
        self.max_session_images = 200
        self.session_started_at = datetime.now()
        self.camera_online = False
        # image_buffer holds downscaled frames for display only; the matching
        # full-resolution captures live on disk and their paths are in
        # capture_paths (that is what analysis and "source_*" archival use).
        self.image_buffer = []
        self.capture_paths = []
        self._capture_dir = None
        # sweep away capture scratch left behind by a previous crash
        if CAPTURE_CACHE_DIR.is_dir():
            for stale in CAPTURE_CACHE_DIR.glob("session_*"):
                shutil.rmtree(stale, ignore_errors=True)
        self.result_gallery_frames = []
        self.latest_frame = None
        self.last_result = None
        self.current_result_saved_dir = None
        self.analysis_worker = None
        self.camera_thread = None
        self.thumbnail_tiles = []
        self.saved_result_records = []
        self.saved_result_buttons = []
        self.selected_saved_result_dir = None
        self.saved_result_gallery_items = []
        self.saved_result_gallery_cache_dir = None
        self._screen_fit_applied = False
        self._style_scale = 1.0

        self.setWindowTitle("Leukemia Detection System")
        self.resize(1280, 800)
        self.setMinimumSize(320, 240)
        self._build_ui()
        self._apply_styles()
        self._update_save_button_state()
        self.saved_result_records = self._load_saved_result_records()
        self._update_header_chips()
        self._reset_saved_result_details()
        self._update_workflow_state()
        self._sync_camera_for_current_page()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        self.root_layout = QVBoxLayout(root)
        self.root_layout.setContentsMargins(24, 24, 24, 24)
        self.root_layout.setSpacing(20)

        self.header_card = QFrame()
        self.header_card.setObjectName("headerCard")
        self.header_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.header_layout.setContentsMargins(24, 20, 24, 20)
        self.header_layout.setSpacing(18)
        self.header_card.setLayout(self.header_layout)

        self.title_block = QVBoxLayout()
        self.title_block.setSpacing(4)

        self.eyebrow_label = QLabel("AI-ASSISTED LEUKEMIA SCREENING")
        self.eyebrow_label.setObjectName("eyebrow")
        self.header_title_label = QLabel("Leukemia Detection System")
        self.header_title_label.setObjectName("titleLabel")
        self.header_subtitle_label = QLabel("Live microscopy capture, full-frame detection, and import-ready review workflow for Jetson Orin Nano.")
        self.header_subtitle_label.setObjectName("subtitleLabel")
        self.header_subtitle_label.setWordWrap(True)

        self.title_block.addWidget(self.eyebrow_label)
        self.title_block.addWidget(self.header_title_label)
        self.title_block.addWidget(self.header_subtitle_label)

        self.header_chip_container = QWidget()
        self.header_chip_container.setObjectName("transparentSurface")
        self.header_chip_row = QBoxLayout(QBoxLayout.LeftToRight)
        self.header_chip_row.setContentsMargins(0, 0, 0, 0)
        self.header_chip_row.setSpacing(10)
        self.date_chip = QLabel()
        self.saved_results_button = QPushButton("Saved Results")
        self.saved_results_button.setObjectName("headerActionButton")
        self.saved_results_button.setCursor(Qt.PointingHandCursor)
        self.saved_results_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.saved_results_button.clicked.connect(self.show_saved_results_page)
        set_badge(self.date_chip, datetime.now().strftime("%d %b %Y"), "neutral")
        self.header_chip_row.addWidget(self.saved_results_button)
        self.header_chip_row.addWidget(self.date_chip)
        self.header_chip_container.setLayout(self.header_chip_row)

        self.header_right = QVBoxLayout()
        self.header_right.setSpacing(10)
        self.header_right.addWidget(self.header_chip_container, 0, Qt.AlignRight | Qt.AlignTop)
        self.header_right.addStretch(1)

        self.header_layout.addLayout(self.title_block, 3)
        self.header_layout.addLayout(self.header_right, 2)

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self._build_acquisition_page())
        self.view_stack.addWidget(self._build_result_page())
        self.view_stack.addWidget(self._build_saved_results_page())
        self.view_stack.currentChanged.connect(self._handle_page_changed)

        self.root_layout.addWidget(self.header_card)
        self.root_layout.addWidget(self.view_stack, 1)

    def _available_screen_geometry(self):
        app = QApplication.instance()
        screen = self.screen()
        if screen is None and app is not None:
            screen = app.primaryScreen()
        if screen is None and app is not None:
            screens = app.screens()
            screen = screens[0] if screens else None
        return screen.availableGeometry() if screen is not None else None

    def _fit_window_to_screen(self):
        available = self._available_screen_geometry()
        if available is None:
            return

        if available.width() <= 1280 or available.height() <= 800:
            target_width = max(self.minimumWidth(), available.width())
            target_height = max(self.minimumHeight(), available.height())
            self.resize(target_width, target_height)
            self.move(available.topLeft())
        else:
            target_width = min(1280, available.width() - 48)
            target_height = min(800, available.height() - 48)
            self.resize(target_width, target_height)
            self.move(
                available.x() + max((available.width() - target_width) // 2, 0),
                available.y() + max((available.height() - target_height) // 2, 0),
            )
        self._update_responsive_layouts()

    def _build_acquisition_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        self.acquisition_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.acquisition_layout.setSpacing(14)
        self.acquisition_layout.addWidget(self._build_preview_panel(), 7)
        self.acquisition_layout.addWidget(self._build_control_panel(), 4)

        self.gallery_card = QFrame()
        self.gallery_card.setObjectName("panelCard")
        gallery_layout = QVBoxLayout(self.gallery_card)
        gallery_layout.setContentsMargins(20, 18, 20, 18)
        gallery_layout.setSpacing(10)

        self.gallery_header_layout = QBoxLayout(QBoxLayout.LeftToRight)
        gallery_title = QLabel("Buffered Images")
        gallery_title.setObjectName("sectionTitle")
        self.gallery_hint_label = QLabel("Captured and imported images appear here before analysis. Click any image to enlarge it.")
        self.gallery_hint_label.setObjectName("sectionHint")
        self.gallery_hint_label.setWordWrap(True)
        self.gallery_header_layout.addWidget(gallery_title)
        self.gallery_header_layout.addStretch(1)
        self.gallery_header_layout.addWidget(self.gallery_hint_label)

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(False)
        self.gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gallery_scroll.setFrameShape(QFrame.NoFrame)
        self.gallery_scroll.viewport().setObjectName("transparentSurface")

        self.gallery_track = QWidget()
        self.gallery_track.setObjectName("savedResultsListSurface")
        self.gallery_layout = QHBoxLayout(self.gallery_track)
        self.gallery_layout.setContentsMargins(4, 4, 4, 4)
        self.gallery_layout.setSpacing(10)
        self.gallery_track.setLayout(self.gallery_layout)
        self.gallery_scroll.setWidget(self.gallery_track)

        gallery_layout.addLayout(self.gallery_header_layout)
        gallery_layout.addWidget(self.gallery_scroll)

        page_layout.addLayout(self.acquisition_layout, 1)
        page_layout.addWidget(self.gallery_card, 0)
        return page

    def _build_preview_panel(self):
        panel = QFrame()
        panel.setObjectName("panelCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Live Microscope Preview")
        title.setObjectName("sectionTitle")
        self.camera_badge = QLabel()
        set_badge(self.camera_badge, "Waiting for camera", "warning")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.camera_badge)

        self.preview_surface = PreviewSurface()
        self.preview_surface.setMinimumHeight(300)

        self.preview_caption = QLabel("Frame the whole field in the preview. Every capture is saved at full sensor resolution.")
        self.preview_caption.setObjectName("sectionHint")
        self.preview_caption.setWordWrap(True)

        layout.addLayout(header)
        layout.addWidget(self.preview_surface, 1)
        layout.addWidget(self.preview_caption)

        # Live, optional WBC guidance overlaid on the acquisition preview.
        controls = QHBoxLayout()
        self.live_count_label = QLabel("WBC live: starting…")
        self.live_count_label.setObjectName("sectionHint")
        self.auto_capture_toggle = QCheckBox("Auto-capture")
        self.auto_threshold = QSpinBox()
        self.auto_threshold.setRange(1, 99)
        self.auto_threshold.setValue(3)
        self.auto_threshold.setPrefix("Threshold: ")
        controls.addWidget(self.live_count_label)
        controls.addStretch(1)
        controls.addWidget(self.auto_capture_toggle)
        controls.addWidget(self.auto_threshold)
        layout.insertLayout(2, controls)
        return panel

    def _build_control_panel(self):
        panel = QFrame()
        panel.setObjectName("panelCard")
        self.control_panel_layout = QVBoxLayout(panel)
        self.control_panel_layout.setContentsMargins(18, 18, 18, 18)
        self.control_panel_layout.setSpacing(10)

        title = QLabel("Acquisition Workflow")
        title.setObjectName("sectionTitle")
        self.workflow_badge = QLabel()
        set_badge(self.workflow_badge, "Ready", "neutral")

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.workflow_badge)

        self.workflow_detail_label = QLabel(
            "Analysis unlocks from one field. Block 2 decides from 100 classified leukocytes "
            "(tier S, ~57 fields) and applies the full pattern grid from 200 (tier P, ~114 fields); "
            "fewer than 100 returns insufficient_evidence."
        )
        self.workflow_detail_label.setObjectName("sectionHint")
        self.workflow_detail_label.setWordWrap(True)

        self.metric_grid = QGridLayout()
        self.metric_grid.setContentsMargins(0, 0, 0, 0)
        self.metric_grid.setHorizontalSpacing(10)
        self.metric_grid.setVerticalSpacing(10)

        self.captured_tile = MetricTile("Fields Captured", compact=True)
        self.remaining_tile = MetricTile("Est. WBC", compact=True)
        self.recommended_tile = MetricTile("Fields per tier", compact=True)
        self.readiness_tile = MetricTile("Run Analysis", compact=True)
        self.metric_tiles = [
            self.captured_tile,
            self.remaining_tile,
            self.recommended_tile,
            self.readiness_tile,
        ]
        self._rebuild_metric_grid(2)

        self.capture_button = NeonButton("Capture Frame", variant="primary", glow_color=COLORS["accent"])
        self.capture_button.clicked.connect(self.capture_image)

        self.import_button = NeonButton("Import Images", variant="secondary", glow_color=COLORS["accent_alt"])
        self.import_button.clicked.connect(self.import_images)

        self.run_button = NeonButton("Run Analysis", variant="secondary", glow_color=COLORS["success"])
        self.run_button.clicked.connect(self.run_inference)

        self.clear_button = NeonButton("Clear Buffer", variant="ghost", glow_color=COLORS["warning"])
        self.clear_button.clicked.connect(self.clear_samples)

        self.button_panel = QWidget()
        self.button_row_layout = QGridLayout(self.button_panel)
        self.button_row_layout.setContentsMargins(0, 0, 0, 0)
        self.button_row_layout.setHorizontalSpacing(10)
        self.button_row_layout.setVerticalSpacing(10)
        self._rebuild_action_button_grid(1)
        self.button_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.progress_card = QFrame()
        self.progress_card.setObjectName("progressCard")
        progress_layout = QVBoxLayout(self.progress_card)
        progress_layout.setContentsMargins(16, 14, 16, 14)
        progress_layout.setSpacing(6)

        progress_header = QHBoxLayout()
        self.progress_title_label = QLabel("Analysis Progress")
        self.progress_title_label.setObjectName("sectionTitle")
        self.progress_value_label = QLabel("0%")
        self.progress_value_label.setObjectName("progressValue")
        progress_header.addWidget(self.progress_title_label)
        progress_header.addStretch(1)
        progress_header.addWidget(self.progress_value_label)

        self.progress_detail_label = QLabel("Idle")
        self.progress_detail_label.setObjectName("sectionHint")
        self.progress_detail_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        progress_layout.addLayout(progress_header)
        progress_layout.addWidget(self.progress_detail_label)
        progress_layout.addWidget(self.progress_bar)

        self.control_panel_layout.addLayout(title_row)
        self.control_panel_layout.addWidget(self.workflow_detail_label)
        self.control_panel_layout.addLayout(self.metric_grid)
        self.control_panel_layout.addWidget(self.button_panel)
        self.control_panel_layout.addWidget(self.progress_card)
        self.control_panel_layout.addStretch(1)
        return panel

    def _build_result_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.result_container = QFrame()
        self.result_container.setObjectName("panelCard")
        self.result_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container_layout = QVBoxLayout(self.result_container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        self.result_page_title_label = QLabel("Analysis Result")
        self.result_page_title_label.setObjectName("titleLabel")
        self.result_risk_badge = QLabel()
        set_badge(self.result_risk_badge, "Awaiting analysis", "neutral")
        header_layout.addWidget(self.result_page_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.result_risk_badge)

        self.result_summary_label = QLabel("Run analysis after capturing or importing microscopy images.")
        self.result_summary_label.setObjectName("subtitleLabel")
        self.result_summary_label.setWordWrap(True)

        self.result_label = QLabel("No active result")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setWordWrap(True)
        self.result_meta_label = QLabel("Confidence: --")
        self.result_meta_label.setObjectName("resultMeta")
        self.result_meta_label.setWordWrap(True)

        result_metrics = QGridLayout()
        result_metrics.setHorizontalSpacing(12)
        result_metrics.setVerticalSpacing(12)
        self.result_confidence_tile = MetricTile("Confidence Score")
        self.result_images_tile = MetricTile("Images Used")
        self.result_timestamp_tile = MetricTile("Session Time")
        self.result_action_tile = MetricTile("Recommendation")

        result_metrics.addWidget(self.result_confidence_tile, 0, 0)
        result_metrics.addWidget(self.result_images_tile, 0, 1)
        result_metrics.addWidget(self.result_timestamp_tile, 1, 0)
        result_metrics.addWidget(self.result_action_tile, 1, 1)

        preview_title = QLabel("Analyzed Images")
        preview_title.setObjectName("sectionTitle")
        self.result_gallery_hint_label = QLabel("Annotated detection previews will appear here after analysis. Click any image to enlarge it.")
        self.result_gallery_hint_label.setObjectName("sectionHint")
        self.result_gallery_hint_label.setWordWrap(True)

        self.result_gallery_scroll = QScrollArea()
        self.result_gallery_scroll.setWidgetResizable(True)
        self.result_gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_gallery_scroll.setFrameShape(QFrame.NoFrame)

        self.result_gallery_track = QWidget()
        self.result_gallery_track.setObjectName("savedResultsListSurface")
        self.result_gallery_layout = QGridLayout(self.result_gallery_track)
        self.result_gallery_layout.setContentsMargins(4, 4, 4, 4)
        self.result_gallery_layout.setHorizontalSpacing(6)
        self.result_gallery_layout.setVerticalSpacing(6)
        self.result_gallery_scroll.setWidget(self.result_gallery_track)

        preview_card = QFrame()
        preview_card.setObjectName("panelCard")
        preview_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(8)
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.result_gallery_hint_label)
        preview_layout.addWidget(self.result_gallery_scroll)
        preview_layout.addStretch(1)

        self.result_body_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.result_body_layout.setSpacing(14)

        self.result_summary_card = QFrame()
        self.result_summary_card.setObjectName("resultCard")
        self.result_summary_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        summary_layout = QVBoxLayout(self.result_summary_card)
        summary_layout.setContentsMargins(18, 18, 18, 18)
        summary_layout.setSpacing(10)
        summary_layout.addLayout(header_layout)
        summary_layout.addWidget(self.result_summary_label)
        summary_layout.addWidget(self.result_label)
        summary_layout.addWidget(self.result_meta_label)
        summary_layout.addLayout(result_metrics)
        summary_layout.addStretch(1)

        self.result_body_layout.addWidget(self.result_summary_card, 6)
        self.result_body_layout.addWidget(preview_card, 5)

        self.result_actions_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.result_actions_layout.setSpacing(10)
        self.back_button = NeonButton("Back to Acquisition", variant="ghost", glow_color=COLORS["accent"])
        self.back_button.clicked.connect(self.show_acquisition_page)
        self.retake_button = NeonButton("Clear Buffer", variant="secondary", glow_color=COLORS["warning"])
        self.retake_button.clicked.connect(self.clear_samples)
        self.save_button = NeonButton("Save Result", variant="primary", glow_color=COLORS["success"])
        self.save_button.clicked.connect(self.save_result)

        self.result_actions_layout.addWidget(self.back_button)
        self.result_actions_layout.addWidget(self.retake_button)
        self.result_actions_layout.addWidget(self.save_button)

        self.result_footer_label = QLabel("No report saved yet.")
        self.result_footer_label.setObjectName("sectionHint")
        self.result_footer_label.setWordWrap(True)

        container_layout.addLayout(self.result_body_layout)
        container_layout.addLayout(self.result_actions_layout)
        container_layout.addWidget(self.result_footer_label)
        layout.addWidget(self.result_container, 1)
        return page

    def _build_saved_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.saved_results_container = QFrame()
        self.saved_results_container.setObjectName("panelCard")
        container_layout = QVBoxLayout(self.saved_results_container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        self.saved_results_page_title_label = QLabel("Saved Results")
        self.saved_results_page_title_label.setObjectName("titleLabel")
        self.saved_results_badge = QLabel()
        set_badge(self.saved_results_badge, "Saved 0", "warning")
        header_layout.addWidget(self.saved_results_page_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.saved_results_badge)

        self.saved_results_summary_label = QLabel("Browse saved inference packages from the local Interface/saved_results folder.")
        self.saved_results_summary_label.setObjectName("subtitleLabel")
        self.saved_results_summary_label.setWordWrap(True)

        self.saved_results_body_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.saved_results_body_layout.setSpacing(14)

        self.saved_results_list_card = QFrame()
        self.saved_results_list_card.setObjectName("panelCard")
        list_layout = QVBoxLayout(self.saved_results_list_card)
        list_layout.setContentsMargins(16, 16, 16, 16)
        list_layout.setSpacing(10)

        list_title = QLabel("Saved Packages")
        list_title.setObjectName("sectionTitle")
        self.saved_result_delete_button = QPushButton("Delete package")
        self.saved_result_delete_button.setObjectName("savedResultDeleteButton")
        self.saved_result_delete_button.setCursor(Qt.PointingHandCursor)
        self.saved_result_delete_button.clicked.connect(self.delete_selected_saved_result)
        list_header = QHBoxLayout()
        list_header.setContentsMargins(0, 0, 0, 0)
        list_header.setSpacing(8)
        list_header.addWidget(list_title)
        list_header.addStretch(1)
        list_header.addWidget(self.saved_result_delete_button, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.saved_results_hint_label = QLabel("Select a saved session to inspect its result and annotated previews.")
        self.saved_results_hint_label.setObjectName("sectionHint")
        self.saved_results_hint_label.setWordWrap(True)

        self.saved_results_scroll = QScrollArea()
        self.saved_results_scroll.setWidgetResizable(True)
        self.saved_results_scroll.setFrameShape(QFrame.NoFrame)

        self.saved_results_list_content = QWidget()
        self.saved_results_list_content.setObjectName("savedResultsListSurface")
        self.saved_results_list_layout = QVBoxLayout(self.saved_results_list_content)
        self.saved_results_list_layout.setContentsMargins(0, 0, 0, 0)
        self.saved_results_list_layout.setSpacing(10)
        self.saved_results_scroll.setWidget(self.saved_results_list_content)

        list_layout.addLayout(list_header)
        list_layout.addWidget(self.saved_results_hint_label)
        list_layout.addWidget(self.saved_results_scroll, 1)

        self.saved_result_detail_card = QFrame()
        self.saved_result_detail_card.setObjectName("resultCard")
        detail_frame_layout = QVBoxLayout(self.saved_result_detail_card)
        detail_frame_layout.setContentsMargins(0, 0, 0, 0)
        detail_frame_layout.setSpacing(0)

        self.saved_result_detail_scroll = QScrollArea()
        self.saved_result_detail_scroll.setWidgetResizable(True)
        self.saved_result_detail_scroll.setFrameShape(QFrame.NoFrame)
        self.saved_result_detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.saved_result_detail_content = QWidget()
        self.saved_result_detail_content.setObjectName("transparentSurface")
        detail_layout = QVBoxLayout(self.saved_result_detail_content)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        detail_layout.setSpacing(10)
        self.saved_result_detail_scroll.setWidget(self.saved_result_detail_content)
        detail_frame_layout.addWidget(self.saved_result_detail_scroll)

        detail_header = QHBoxLayout()
        self.saved_result_title_label = QLabel("No saved result selected")
        self.saved_result_title_label.setObjectName("resultLabel")
        self.saved_result_title_label.setWordWrap(True)
        self.saved_result_risk_badge = QLabel()
        set_badge(self.saved_result_risk_badge, "Empty", "warning")
        detail_header.addWidget(self.saved_result_title_label)
        detail_header.addStretch(1)
        detail_header.addWidget(self.saved_result_risk_badge)

        self.saved_result_summary_label = QLabel("Saved results will appear here once a report package exists.")
        self.saved_result_summary_label.setObjectName("subtitleLabel")
        self.saved_result_summary_label.setWordWrap(True)

        self.saved_result_meta_label = QLabel("Saved at: --")
        self.saved_result_meta_label.setObjectName("resultMeta")
        self.saved_result_meta_label.setWordWrap(True)

        self.saved_metrics_widget = QWidget()
        self.saved_metrics_widget.setObjectName("transparentSurface")
        self.saved_metrics_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.saved_metrics_widget.setFixedHeight(188)

        self.saved_metrics_layout = QVBoxLayout(self.saved_metrics_widget)
        self.saved_metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.saved_metrics_layout.setSpacing(12)

        self.saved_metrics_top_widget = QWidget()
        self.saved_metrics_top_widget.setObjectName("transparentSurface")
        self.saved_metrics_top_widget.setFixedHeight(88)
        self.saved_metrics_top_row = QHBoxLayout(self.saved_metrics_top_widget)
        self.saved_metrics_top_row.setContentsMargins(0, 0, 0, 0)
        self.saved_metrics_top_row.setSpacing(12)

        self.saved_metrics_bottom_widget = QWidget()
        self.saved_metrics_bottom_widget.setObjectName("transparentSurface")
        self.saved_metrics_bottom_widget.setFixedHeight(88)
        self.saved_metrics_bottom_row = QHBoxLayout(self.saved_metrics_bottom_widget)
        self.saved_metrics_bottom_row.setContentsMargins(0, 0, 0, 0)
        self.saved_metrics_bottom_row.setSpacing(12)
        self.saved_result_confidence_tile = MetricTile("Confidence Score", compact=True)
        self.saved_result_images_tile = MetricTile("Images Used", compact=True)
        self.saved_result_saved_at_tile = MetricTile("Saved At", compact=True)
        self.saved_result_folder_tile = MetricTile("Package Folder", compact=True)
        self.saved_result_tiles = [
            self.saved_result_confidence_tile,
            self.saved_result_images_tile,
            self.saved_result_saved_at_tile,
            self.saved_result_folder_tile,
        ]
        for tile in self.saved_result_tiles:
            tile.setFixedHeight(88)
        self.saved_metrics_top_row.addWidget(self.saved_result_confidence_tile, 1)
        self.saved_metrics_top_row.addWidget(self.saved_result_images_tile, 1)
        self.saved_metrics_bottom_row.addWidget(self.saved_result_saved_at_tile, 1)
        self.saved_metrics_bottom_row.addWidget(self.saved_result_folder_tile, 1)
        self.saved_metrics_layout.addWidget(self.saved_metrics_top_widget)
        self.saved_metrics_layout.addWidget(self.saved_metrics_bottom_widget)

        archive_title = QLabel("Archived Samples")
        archive_title.setObjectName("sectionTitle")
        self.saved_result_gallery_hint_label = QLabel("Scroll horizontally to review all images in the selected package. Click any image to enlarge it.")
        self.saved_result_gallery_hint_label.setObjectName("sectionHint")
        self.saved_result_gallery_hint_label.setWordWrap(True)

        self.saved_result_gallery_scroll = QScrollArea()
        self.saved_result_gallery_scroll.setWidgetResizable(False)
        self.saved_result_gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.saved_result_gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.saved_result_gallery_scroll.setFrameShape(QFrame.NoFrame)

        self.saved_result_gallery_track = QWidget()
        self.saved_result_gallery_track.setObjectName("savedResultsListSurface")
        self.saved_result_gallery_layout = QHBoxLayout(self.saved_result_gallery_track)
        self.saved_result_gallery_layout.setContentsMargins(8, 8, 8, 8)
        self.saved_result_gallery_layout.setSpacing(12)
        self.saved_result_gallery_scroll.setWidget(self.saved_result_gallery_track)

        detail_layout.addLayout(detail_header)
        detail_layout.addWidget(self.saved_result_summary_label)
        detail_layout.addWidget(self.saved_result_meta_label)
        detail_layout.addWidget(self.saved_metrics_widget)
        detail_layout.addWidget(archive_title)
        detail_layout.addWidget(self.saved_result_gallery_hint_label)
        detail_layout.addWidget(self.saved_result_gallery_scroll)
        detail_layout.addStretch(1)

        self.saved_results_body_layout.addWidget(self.saved_results_list_card, 4)
        self.saved_results_body_layout.addWidget(self.saved_result_detail_card, 6)

        self.saved_results_actions_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.saved_results_actions_layout.setSpacing(10)
        self.saved_results_back_button = NeonButton("Back to Acquisition", variant="ghost", glow_color=COLORS["accent"])
        self.saved_results_back_button.clicked.connect(self.show_acquisition_page)
        self.saved_results_current_button = NeonButton("Go to Current Result", variant="secondary", glow_color=COLORS["success"])
        self.saved_results_current_button.clicked.connect(self.show_result_page)
        self.saved_results_actions_layout.addWidget(self.saved_results_back_button)
        self.saved_results_actions_layout.addWidget(self.saved_results_current_button)

        container_layout.addLayout(header_layout)
        container_layout.addWidget(self.saved_results_summary_label)
        container_layout.addLayout(self.saved_results_body_layout)
        container_layout.addLayout(self.saved_results_actions_layout)
        layout.addWidget(self.saved_results_container)
        return page

    def _apply_styles(self, scale=1.0):
        frame_radius = _scaled_px(24, scale, 14)
        tile_radius = _scaled_px(20, scale, 12)
        small_radius = _scaled_px(18, scale, 10)
        button_padding_v = _scaled_px(10, scale, 5)
        button_padding_h = _scaled_px(16, scale, 8)
        progress_height = _scaled_px(18, scale, 12)
        scrollbar_size = _scaled_px(10, scale, 6)
        scrollbar_handle = _scaled_px(30, scale, 16)
        base_font = _scaled_px(15, scale, 11)
        title_font = _scaled_px(26, scale, 18)
        eyebrow_font = _scaled_px(12, scale, 10)
        hint_font = _scaled_px(12, scale, 10)
        section_font = _scaled_px(17, scale, 13)
        metric_value_font = _scaled_px(20, scale, 13)
        result_font = _scaled_px(26, scale, 18)
        result_meta_font = _scaled_px(13, scale, 11)
        entry_font = _scaled_px(13, scale, 11)
        header_action_font = _scaled_px(12, scale, 10)
        delete_font = _scaled_px(12, scale, 10)
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {COLORS['bg_start']};
            }}

            QWidget#root {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {COLORS['bg_start']},
                    stop: 0.55 #0A141F,
                    stop: 1 {COLORS['bg_end']}
                );
                color: {COLORS['text']};
            }}

            QWidget#transparentSurface {{
                background: transparent;
            }}

            QWidget#savedResultsListSurface {{
                background: rgba(255, 255, 255, 0.025);
                border-radius: {small_radius}px;
            }}

            QFrame#headerCard,
            QFrame#panelCard,
            QFrame#metricTile,
            QFrame#progressCard,
            QFrame#resultCard {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: {frame_radius}px;
            }}

            QFrame#metricTile {{
                background-color: {COLORS['panel_soft']};
                border: 1px solid rgba(138, 156, 176, 0.14);
                border-radius: {tile_radius}px;
            }}

            QFrame#progressCard {{
                border-radius: {tile_radius}px;
            }}

            QLabel#eyebrow {{
                color: {COLORS['accent']};
                font-size: {eyebrow_font}px;
                font-weight: 700;
                letter-spacing: 0.12em;
            }}

            QLabel#titleLabel {{
                color: {COLORS['text']};
                font-size: {title_font}px;
                font-weight: 700;
            }}

            QLabel#subtitleLabel,
            QLabel#sectionHint,
            QLabel#metricNote {{
                color: {COLORS['muted']};
                font-size: {hint_font}px;
            }}

            QLabel#sectionTitle {{
                color: {COLORS['text']};
                font-size: {section_font}px;
                font-weight: 700;
            }}

            QLabel#metricTitle {{
                color: {COLORS['muted']};
                font-size: {hint_font}px;
                font-weight: 600;
            }}

            QLabel#metricValue {{
                color: {COLORS['text']};
                font-size: {metric_value_font}px;
                font-weight: 700;
            }}

            QLabel#progressValue {{
                color: {COLORS['accent']};
                font-size: {metric_value_font}px;
                font-weight: 700;
            }}

            QLabel#resultLabel {{
                color: {COLORS['text']};
                font-size: {result_font}px;
                font-weight: 800;
            }}

            QLabel#resultMeta {{
                color: {COLORS['muted']};
                font-size: {result_meta_font}px;
            }}

            QPushButton {{
                border: 1px solid transparent;
                border-radius: {small_radius}px;
                padding: {button_padding_v}px {button_padding_h}px;
                font-size: {base_font}px;
                font-weight: 700;
                color: {COLORS['text']};
            }}

            QPushButton#headerActionButton {{
                background: rgba(255, 255, 255, 0.025);
                border: 1px solid rgba(138, 156, 176, 0.18);
                border-radius: {_scaled_px(16, scale, 9)}px;
                padding: {_scaled_px(8, scale, 4)}px {_scaled_px(14, scale, 8)}px;
                font-size: {header_action_font}px;
                font-weight: 600;
                color: {COLORS['text']};
                text-align: center;
            }}

            QPushButton#savedResultEntry {{
                background: rgba(255, 255, 255, 0.025);
                border: 1px solid rgba(138, 156, 176, 0.16);
                border-radius: {small_radius}px;
                padding: {_scaled_px(14, scale, 7)}px {_scaled_px(16, scale, 8)}px;
                font-size: {entry_font}px;
                font-weight: 600;
                color: {COLORS['text']};
                text-align: left;
            }}

            QPushButton#savedResultEntry[selected="true"] {{
                background: rgba(120, 184, 214, 0.10);
                border: 1px solid rgba(120, 184, 214, 0.28);
            }}

            QPushButton#savedResultDeleteButton {{
                background: transparent;
                border: none;
                padding: 0px;
                font-size: {delete_font}px;
                font-weight: 600;
                color: rgba(214, 105, 105, 0.86);
                text-align: right;
            }}

            QPushButton#savedResultDeleteButton:hover {{
                color: {COLORS['danger']};
                text-decoration: underline;
            }}

            QPushButton#savedResultDeleteButton:disabled {{
                color: rgba(157, 175, 194, 0.34);
            }}

            QPushButton[variant="primary"] {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {COLORS['accent']},
                    stop: 1 {COLORS['accent_alt']}
                );
                border: 1px solid rgba(120, 184, 214, 0.34);
                color: #07111A;
            }}

            QPushButton[variant="secondary"] {{
                background: rgba(120, 184, 214, 0.08);
                border: 1px solid rgba(120, 184, 214, 0.24);
                color: {COLORS['accent']};
            }}

            QPushButton[variant="ghost"] {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(138, 156, 176, 0.16);
                color: {COLORS['text']};
            }}

            QPushButton:hover {{
                border-color: {COLORS['border_strong']};
            }}

            QPushButton[pulse="true"] {{
                border-color: rgba(255, 255, 255, 0.28);
                background: rgba(255, 255, 255, 0.08);
            }}

            QPushButton[variant="primary"][pulse="true"] {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #90C5DE,
                    stop: 1 #5BA3C6
                );
            }}

            QPushButton[variant="secondary"][pulse="true"] {{
                background: rgba(120, 184, 214, 0.12);
                border-color: rgba(120, 184, 214, 0.36);
            }}

            QPushButton:disabled {{
                background: rgba(120, 138, 159, 0.10);
                border-color: rgba(120, 138, 159, 0.16);
                color: rgba(190, 205, 220, 0.42);
            }}

            QProgressBar {{
                min-height: {progress_height}px;
                border-radius: {_scaled_px(9, scale, 6)}px;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(138, 156, 176, 0.14);
                text-align: center;
                color: {COLORS['text']};
            }}

            QProgressBar::chunk {{
                border-radius: {_scaled_px(8, scale, 5)}px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {COLORS['accent']},
                    stop: 1 {COLORS['accent_alt']}
                );
            }}

            QScrollArea {{
                background: transparent;
                border: none;
            }}

            QScrollBar:horizontal {{
                background: rgba(255, 255, 255, 0.03);
                height: {scrollbar_size}px;
                border-radius: {_scaled_px(5, scale, 3)}px;
                margin: 2px 0;
            }}

            QScrollBar::handle:horizontal {{
                background: rgba(120, 184, 214, 0.22);
                border-radius: {_scaled_px(5, scale, 3)}px;
                min-width: {scrollbar_handle}px;
            }}

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            QScrollBar:vertical {{
                background: rgba(255, 255, 255, 0.03);
                width: {scrollbar_size}px;
                border-radius: {_scaled_px(5, scale, 3)}px;
                margin: 0;
            }}

            QScrollBar::handle:vertical {{
                background: rgba(120, 184, 214, 0.22);
                border-radius: {_scaled_px(5, scale, 3)}px;
                min-height: {scrollbar_handle}px;
            }}

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

    def _connect_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return

        self.preview_surface.set_status_badge(None)
        set_badge(self.camera_badge, "Starting Camera", "neutral")
        self.preview_caption.setText(
            "Connecting to the microscope feed for live capture preview."
        )
        self.camera_thread = CameraThread(sensor_id=self.camera_source)
        self.camera_thread.frame_ready.connect(self._update_preview)
        self.camera_thread.camera_status.connect(self._update_camera_status)
        self.camera_thread.start()

        if getattr(self, "live_worker", None) is None:
            # The live WBC overlay needs the full inference stack (torch, YOLO).
            # If those are unavailable, keep the camera, capture and analysis
            # working and just disable the live overlay.
            try:
                from Interface.live_guidance import LiveWBCWorker

                self.live_worker = LiveWBCWorker(
                    PROJECT_ROOT / "config" / "inference.yaml", max_fps=4.0, parent=self
                )
                self.live_worker.detection_ready.connect(self._on_live_detection)
                self.live_worker.failed.connect(
                    lambda detail: self.live_count_label.setText(f"Live WBC unavailable: {detail}")
                )
                self.live_worker.start()
            except Exception as exc:  # noqa: BLE001 - any import/init failure
                self.live_worker = None
                self.live_count_label.setText(f"Live WBC overlay disabled: {exc}")
        self._live_boxes, self._live_armed, self._last_auto_capture = [], True, 0.0

    def _disconnect_camera(self):
        worker = getattr(self, "live_worker", None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
            self.live_worker = None

        thread = self.camera_thread
        self.camera_thread = None
        if thread is not None:
            if thread.isRunning():
                thread.stop()
            thread.deleteLater()

        self.preview_surface.set_status_badge("PAUSED", "warning")
        set_badge(self.camera_badge, "Camera Paused", "warning")
        self.preview_caption.setText(
            "Camera paused outside the capture page. Return to acquisition to resume the live preview."
        )

    def _is_acquisition_page_active(self):
        return self.view_stack.currentIndex() == self.page_acquisition

    def _sync_camera_for_current_page(self):
        if self._is_acquisition_page_active():
            self._connect_camera()
        else:
            self._disconnect_camera()

    def _handle_page_changed(self, index):
        del index
        self._sync_camera_for_current_page()

    def _update_header_chips(self):
        set_badge(self.date_chip, datetime.now().strftime("%d %b %Y"), "neutral")
        saved_count = len(self.saved_result_records)
        self.saved_results_button.setToolTip(
            "Open saved result packages" if saved_count else "No saved result package found yet"
        )

    def _refresh_badges(self):
        for badge in (
            self.date_chip,
            self.camera_badge,
            self.workflow_badge,
            self.result_risk_badge,
            self.saved_results_badge,
            self.saved_result_risk_badge,
        ):
            text = badge.property("badge_text")
            tone = badge.property("badge_tone") or "neutral"
            if text:
                set_badge(badge, str(text), str(tone))

    def _compute_style_scale(self, width, height):
        scale = min(width / 1280.0, height / 820.0, 1.0)
        if width <= 720 or height <= 460:
            scale = min(scale, 0.62)
        elif width <= 820 or height <= 520:
            scale = min(scale, 0.70)
        elif width <= 1024 or height <= 640:
            scale = min(scale, 0.82)
        elif height < 820:
            scale = min(scale, 0.92)
        return max(0.60, scale)

    def _apply_progress_panel_mode(self):
        width = self.width()
        height = self.height()
        analysis_running = bool(self.analysis_worker and self.analysis_worker.isRunning())
        compact_progress = analysis_running and (width <= 920 or height <= 620)
        ultra_compact_progress = analysis_running and (width <= 720 or height <= 460)

        show_workflow_detail = not analysis_running or not compact_progress
        show_metric_tiles = not compact_progress

        self.workflow_detail_label.setVisible(show_workflow_detail)
        for tile in self.metric_tiles:
            tile.setVisible(show_metric_tiles)

        progress_layout = self.progress_card.layout()
        if ultra_compact_progress:
            progress_layout.setContentsMargins(10, 8, 10, 8)
            progress_layout.setSpacing(4)
            self.progress_title_label.setText("Analyzing")
            self.progress_detail_label.setVisible(False)
            self.progress_detail_label.setMaximumHeight(0)
            self.progress_card.setMinimumHeight(52)
            self.progress_card.setMaximumHeight(52)
        elif compact_progress:
            progress_layout.setContentsMargins(12, 10, 12, 10)
            progress_layout.setSpacing(5)
            self.progress_title_label.setText("Analysis Progress")
            self.progress_detail_label.setVisible(True)
            self.progress_detail_label.setMaximumHeight(30)
            self.progress_card.setMinimumHeight(84)
            self.progress_card.setMaximumHeight(84)
        else:
            progress_layout.setContentsMargins(16, 14, 16, 14)
            progress_layout.setSpacing(6)
            self.progress_title_label.setText("Analysis Progress")
            self.progress_detail_label.setVisible(analysis_running)
            self.progress_detail_label.setMaximumHeight(16777215)
            self.progress_card.setMinimumHeight(0)
            self.progress_card.setMaximumHeight(16777215)

        self.progress_bar.setTextVisible(not ultra_compact_progress)

    def _format_datetime_text(self, value, fallback="--"):
        if not value:
            return fallback
        try:
            return datetime.fromisoformat(value).strftime("%d %b %Y  %H:%M")
        except ValueError:
            return str(value)

    def _load_saved_result_records(self):
        records = []
        root = SAVED_RESULTS_DIR
        if not root.exists():
            return records

        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            result_path = entry / "result.json"
            if not result_path.exists():
                continue

            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            result = payload.get("result", {})
            records.append(
                {
                    "directory": entry,
                    "payload": payload,
                    "result": result,
                    "saved_at": payload.get("saved_at", ""),
                    "session_started_at": payload.get("session_started_at", ""),
                    "label": result.get("label", "Saved Result"),
                    "confidence": result.get("confidence", 0),
                    "risk_level": result.get("risk_level", "low"),
                    "images_used": payload.get("images_used", result.get("images_used", 0)),
                    "summary": result.get("summary", "No summary available."),
                    "samples": sorted(entry.glob("sample_*.png")),
                }
            )

        records.sort(key=lambda item: item["saved_at"] or item["directory"].name, reverse=True)
        return records

    def _clear_saved_results_list(self):
        while self.saved_results_list_layout.count():
            item = self.saved_results_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.saved_result_buttons.clear()
        self.saved_results_list_content.update()

    def _clear_saved_result_gallery(self):
        while self.saved_result_gallery_layout.count():
            item = self.saved_result_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.saved_result_gallery_track.setFixedSize(0, 0)
        self.saved_result_gallery_track.update()

    def _clear_result_gallery(self):
        while self.result_gallery_layout.count():
            item = self.result_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.result_gallery_track.setFixedSize(0, 0)
        self.result_gallery_track.update()

    def _gallery_items_from_frames(self, frames):
        return [
            (frame, f"Image {index:02d}")
            for index, frame in enumerate(frames, start=1)
            if frame is not None
        ]

    def _gallery_items_from_saved_samples(self, sample_items):
        items = []
        for sample_path, frame in sample_items:
            if frame is None:
                continue
            items.append((frame, sample_path.stem.replace("_", " ").title()))
        return items

    def _create_clickable_thumb(self, frame, title, on_open=None):
        thumb = ClickableImageLabel()
        thumb.setAlignment(Qt.AlignCenter)
        thumb.set_preview_payload(frame, title)
        if on_open is None:
            on_open = lambda label=thumb: self._open_image_lightbox(
                [(label.frame_data, label.preview_title)],
                0,
            )
        thumb.clicked.connect(on_open)
        return thumb

    def _open_image_lightbox(self, gallery_items, start_index=0):
        normalized_items = [
            (frame, title)
            for frame, title in gallery_items
            if frame is not None
        ]
        if not normalized_items:
            return
        start_index = max(0, min(start_index, len(normalized_items) - 1))
        dialog = ImageLightboxDialog(normalized_items, start_index, self)
        dialog.exec_()

    def _update_save_button_state(self):
        has_result = self.last_result is not None
        already_saved = self.current_result_saved_dir is not None
        self.save_button.setEnabled(has_result and not already_saved)
        self.save_button.setText("Result Saved" if already_saved else "Save Result")
        if already_saved:
            self.save_button.setToolTip(f"Current result already saved in {self.current_result_saved_dir.name}")
        elif has_result:
            self.save_button.setToolTip("Save the current result package")
        else:
            self.save_button.setToolTip("Run analysis before saving a result")

    def _update_saved_results_action_state(self):
        has_selected_package = (
            self.selected_saved_result_dir is not None
            and self.selected_saved_result_dir.exists()
        )
        self.saved_result_delete_button.setEnabled(has_selected_package)
        if has_selected_package:
            self.saved_result_delete_button.setToolTip(
                f"Delete saved package {self.selected_saved_result_dir.name}"
            )
        else:
            self.saved_result_delete_button.setToolTip(
                "Select a saved package before deleting it"
            )

    def _populate_result_gallery(self, frames):
        self._clear_result_gallery()

        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            empty_label = QLabel("No analyzed image available for this result yet.")
            empty_label.setObjectName("sectionHint")
            empty_label.setWordWrap(True)
            self.result_gallery_layout.addWidget(empty_label, 0, 0)
            self.result_gallery_track.setFixedSize(max(self.result_gallery_scroll.viewport().width(), 280), 84)
            return

        columns = 5
        rows = 4
        spacing_x = self.result_gallery_layout.horizontalSpacing()
        spacing_y = self.result_gallery_layout.verticalSpacing()
        margins = self.result_gallery_layout.contentsMargins()
        viewport_width = max(self.result_gallery_scroll.viewport().width(), 420)
        viewport_height = max(self.result_gallery_scroll.viewport().height(), 220)
        available_width = max(
            viewport_width - margins.left() - margins.right() - spacing_x * (columns - 1),
            columns * 68,
        )
        available_height = max(
            viewport_height - margins.top() - margins.bottom() - spacing_y * (rows - 1),
            rows * 54,
        )
        tile_width = max(72, available_width // columns)
        tile_height = max(56, available_height // rows)
        thumb_width = max(60, tile_width - 8)
        thumb_height = max(38, tile_height - 18)

        limited_frames = valid_frames[:20]
        gallery_items = self._gallery_items_from_frames(limited_frames)
        for index, frame in enumerate(limited_frames, start=1):
            tile = QFrame()
            tile.setObjectName("metricTile")
            tile.setFixedWidth(tile_width)
            tile.setFixedHeight(tile_height)

            layout = QVBoxLayout(tile)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            title = f"Image {index:02d}"
            thumb = self._create_clickable_thumb(
                frame,
                title,
                on_open=lambda checked=False, items=gallery_items, idx=index - 1: self._open_image_lightbox(items, idx),
            )
            thumb.setFixedSize(thumb_width, thumb_height)
            thumb.setPixmap(rounded_pixmap_from_frame(frame, thumb_width, thumb_height, radius=10))

            label = QLabel(title)
            label.setObjectName("metricTitle")
            label.setAlignment(Qt.AlignCenter)

            layout.addWidget(thumb)
            layout.addWidget(label)
            row = (index - 1) // columns
            column = (index - 1) % columns
            self.result_gallery_layout.addWidget(tile, row, column)

        total_width = viewport_width
        total_height = (
            margins.top()
            + margins.bottom()
            + rows * tile_height
            + spacing_y * max(rows - 1, 0)
        )
        self.result_gallery_track.setFixedSize(total_width, total_height)

    def _schedule_result_gallery_refresh(self):
        QTimer.singleShot(
            0,
            lambda: self._populate_result_gallery(
                self.result_gallery_frames if self.last_result is not None else []
            ),
        )

    def _load_saved_result_gallery_items(self, sample_paths):
        items = []
        for sample_path in sample_paths:
            frame = cv2.imread(str(sample_path))
            if frame is not None:
                items.append((sample_path, frame))
        return items

    def _schedule_saved_result_gallery_refresh(self):
        QTimer.singleShot(
            0,
            lambda: self._populate_saved_result_gallery(self.saved_result_gallery_items),
        )

    def _populate_saved_result_gallery(self, sample_items):
        self._clear_saved_result_gallery()

        if not sample_items:
            empty_label = QLabel("No saved images found in this package.")
            empty_label.setObjectName("sectionHint")
            empty_label.setWordWrap(True)
            self.saved_result_gallery_layout.addWidget(empty_label)
            self.saved_result_gallery_layout.addStretch(1)
            self.saved_result_gallery_track.setFixedSize(max(self.saved_result_gallery_scroll.viewport().width(), 260), 84)
            return

        if self.width() >= 1320:
            tile_width = 150
        elif self.width() >= 1120:
            tile_width = 138
        elif self.width() >= 900:
            tile_width = 126
        else:
            tile_width = 112

        thumb_width = max(108, tile_width - 12)
        thumb_height = max(72, int(thumb_width * 0.62))
        tile_height = thumb_height + 34

        valid_count = 0
        gallery_items = self._gallery_items_from_saved_samples(sample_items)
        for sample_path, frame in sample_items:
            tile = QFrame()
            tile.setObjectName("metricTile")
            tile.setFixedWidth(tile_width)
            tile.setFixedHeight(tile_height)

            layout = QVBoxLayout(tile)
            layout.setContentsMargins(6, 6, 6, 8)
            layout.setSpacing(6)

            title = sample_path.stem.replace("_", " ").title()
            thumb = self._create_clickable_thumb(
                frame,
                title,
                on_open=lambda checked=False, items=gallery_items, idx=valid_count: self._open_image_lightbox(items, idx),
            )
            thumb.setFixedSize(thumb_width, thumb_height)
            thumb.setPixmap(rounded_pixmap_from_frame(frame, thumb_width, thumb_height, radius=12))

            label = QLabel(title)
            label.setObjectName("metricTitle")
            label.setAlignment(Qt.AlignCenter)

            layout.addWidget(thumb)
            layout.addWidget(label)
            self.saved_result_gallery_layout.addWidget(tile)
            valid_count += 1

        if valid_count == 0:
            empty_label = QLabel("No readable images found in this package.")
            empty_label.setObjectName("sectionHint")
            empty_label.setWordWrap(True)
            self.saved_result_gallery_layout.addWidget(empty_label)
            self.saved_result_gallery_layout.addStretch(1)
            self.saved_result_gallery_track.setFixedSize(max(self.saved_result_gallery_scroll.viewport().width(), 280), 84)
            return

        self.saved_result_gallery_layout.addStretch(1)
        total_width = (
            16
            + valid_count * tile_width
            + max(valid_count - 1, 0) * self.saved_result_gallery_layout.spacing()
        )
        self.saved_result_gallery_track.setFixedSize(total_width, tile_height + 16)

    def _set_saved_result_entry_selected(self, button, selected):
        button.setProperty("selected", "true" if selected else "false")
        button.style().unpolish(button)
        button.style().polish(button)

    def _reset_saved_result_details(self):
        self.saved_result_gallery_items = []
        self.saved_result_gallery_cache_dir = None
        self.selected_saved_result_dir = None
        set_badge(self.saved_results_badge, f"Saved {len(self.saved_result_records)}", "warning" if not self.saved_result_records else "success")
        set_badge(self.saved_result_risk_badge, "Empty", "warning")
        has_saved_records = bool(self.saved_result_records)
        self.saved_result_title_label.setText(
            "Select a saved package" if has_saved_records else "Saved results folder is empty"
        )
        self.saved_result_summary_label.setText(
            "Open Saved Results to browse archived packages from the local Interface/saved_results folder."
            if has_saved_records
            else "Run analysis and use Save Result to create report packages in the Interface/saved_results folder."
        )
        self.saved_result_meta_label.setText("Saved at: --")
        self.saved_result_confidence_tile.set_data("--", "Model confidence")
        self.saved_result_images_tile.set_data("--", "Captured images in the saved package")
        self.saved_result_saved_at_tile.set_data("--", "Timestamp of the saved package")
        self.saved_result_folder_tile.set_data("--", "Saved result folder")
        self._clear_saved_result_gallery()
        self.saved_result_gallery_hint_label.setText("No archived images available.")
        self.saved_results_hint_label.setText(
            f"{len(self.saved_result_records)} package(s) available in Interface/saved_results."
            if has_saved_records
            else "No saved result package found yet."
        )
        self.saved_results_summary_label.setText(
            "Browse saved inference packages from the local Interface/saved_results folder."
            if has_saved_records
            else "No saved result package found in the local Interface/saved_results folder."
        )
        self._update_saved_results_action_state()

    def _show_saved_result_details(self, directory):
        selected_record = None
        for record, button in zip(self.saved_result_records, self.saved_result_buttons):
            is_selected = record["directory"] == directory
            self._set_saved_result_entry_selected(button, is_selected)
            if is_selected:
                selected_record = record

        if selected_record is None:
            self._reset_saved_result_details()
            return

        self.selected_saved_result_dir = selected_record["directory"]
        risk_level = selected_record["risk_level"]
        tone = (
            "danger"
            if risk_level == "high"
            else "warning"
            if risk_level in {"warning", "indeterminate"}
            else "success"
        )
        set_badge(self.saved_result_risk_badge, selected_record["label"], tone)
        set_badge(self.saved_results_badge, f"Saved {len(self.saved_result_records)}", "success")
        self.saved_results_hint_label.setText(f"{len(self.saved_result_records)} package(s) available in Interface/saved_results.")
        self.saved_result_title_label.setText(selected_record["label"])
        self.saved_result_summary_label.setText(selected_record["summary"])
        detector_text = selected_record["result"].get("detection_backend", "--")
        self.saved_result_meta_label.setText(
            f"Saved at {self._format_datetime_text(selected_record['saved_at'])}  |  Session start {self._format_datetime_text(selected_record['session_started_at'])}  |  Detector: {detector_text}"
        )
        self.saved_result_confidence_tile.set_data(f"{selected_record['confidence']}%", "Model confidence for the saved run")
        self.saved_result_images_tile.set_data(str(selected_record["images_used"]), "Analyzed images stored in the package")
        self.saved_result_saved_at_tile.set_data(self._format_datetime_text(selected_record["saved_at"]), "Package save timestamp")
        self.saved_result_folder_tile.set_data(selected_record["directory"].name, "Saved result package folder")
        sample_count = len(selected_record["samples"])
        self.saved_result_gallery_hint_label.setText(
            f"{sample_count} archived image(s) in this package. Scroll horizontally to review them and click to enlarge."
            if sample_count
            else "No archived images available for this package."
        )
        if self.saved_result_gallery_cache_dir != selected_record["directory"]:
            self.saved_result_gallery_items = self._load_saved_result_gallery_items(
                selected_record["samples"]
            )
            self.saved_result_gallery_cache_dir = selected_record["directory"]
        self._schedule_saved_result_gallery_refresh()
        self._update_saved_results_action_state()

    def _refresh_saved_results(self, selected_dir=None):
        self.saved_result_records = self._load_saved_result_records()
        self._update_header_chips()
        self._clear_saved_results_list()

        if not self.saved_result_records:
            self.selected_saved_result_dir = None
            empty_label = QLabel("Saved results folder is empty.")
            empty_label.setObjectName("sectionHint")
            empty_label.setWordWrap(True)
            self.saved_results_list_layout.addWidget(empty_label)
            self.saved_results_list_layout.addStretch(1)
            self._reset_saved_result_details()
            return

        self.saved_results_summary_label.setText("Browse saved inference packages from the local Interface/saved_results folder.")
        for record in self.saved_result_records:
            saved_label = self._format_datetime_text(record["saved_at"], fallback=record["directory"].name)
            button = QPushButton(
                f"{saved_label}\n{record['label']}  |  {record['confidence']}% confidence  |  {record['images_used']} images"
            )
            button.setObjectName("savedResultEntry")
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(74)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda checked=False, path=record["directory"]: self._show_saved_result_details(path))
            self.saved_results_list_layout.addWidget(button)
            self.saved_result_buttons.append(button)

        self.saved_results_list_layout.addStretch(1)

        target_dir = selected_dir or self.selected_saved_result_dir or self.saved_result_records[0]["directory"]
        self._show_saved_result_details(target_dir)

    def delete_selected_saved_result(self):
        directory = self.selected_saved_result_dir
        if directory is None:
            return
        if not directory.exists():
            self._refresh_saved_results()
            return

        package_name = directory.name
        confirmation = QMessageBox.question(
            self,
            "Delete Saved Package",
            (
                f"Delete the saved package '{package_name}'?\n\n"
                "This will permanently remove the folder, the archived images, and the saved result.json file."
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if confirmation != QMessageBox.Yes:
            return

        remaining_dirs = [
            record["directory"]
            for record in self.saved_result_records
            if record["directory"] != directory
        ]
        next_selected_dir = remaining_dirs[0] if remaining_dirs else None

        try:
            shutil.rmtree(directory)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Delete Failed",
                f"Could not delete '{package_name}': {exc}",
            )
            return

        if self.current_result_saved_dir == directory:
            self.current_result_saved_dir = None
            self._update_save_button_state()

        self.saved_result_gallery_cache_dir = None
        self.saved_result_gallery_items = []
        self.selected_saved_result_dir = next_selected_dir
        self._refresh_saved_results(selected_dir=next_selected_dir)
        self.saved_results_summary_label.setText(
            f"Deleted package '{package_name}'. Browse saved inference packages from the local Interface/saved_results folder."
        )

    @staticmethod
    def _display_downscale(frame, max_side=1200):
        """A lightweight copy for on-screen use (thumbnails, galleries, lightbox)."""
        if frame is None:
            return None
        longest = max(frame.shape[0], frame.shape[1])
        if longest <= max_side:
            return frame.copy()
        scale = max_side / float(longest)
        return cv2.resize(
            frame,
            (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _ensure_capture_dir(self):
        if self._capture_dir is None or not Path(self._capture_dir).is_dir():
            try:
                CAPTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._capture_dir = Path(tempfile.mkdtemp(prefix="session_", dir=str(CAPTURE_CACHE_DIR)))
            except OSError:
                self._capture_dir = Path(tempfile.mkdtemp(prefix="aster_session_"))
        return self._capture_dir

    def _discard_capture_dir(self):
        if self._capture_dir is not None:
            shutil.rmtree(self._capture_dir, ignore_errors=True)
        self._capture_dir = None

    def _update_preview(self, frame):
        self.latest_frame = frame
        worker = getattr(self, "live_worker", None)
        if worker is not None:
            worker.submit(frame)
        rendered = frame.copy()
        for x1, y1, x2, y2, score in getattr(self, "_live_boxes", []):
            cv2.rectangle(rendered, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 80), 2)
            cv2.putText(
                rendered,
                f"WBC {score:.2f}",
                (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 80),
                2,
            )
        self.preview_surface.set_frame(rendered)

    def _on_live_detection(self, boxes, elapsed):
        self._live_boxes = boxes
        fps = 1.0 / elapsed if elapsed else 0.0
        self.live_count_label.setText(f"WBC live: {len(boxes)} · YOLO {fps:.1f} FPS")
        if len(boxes) < self.auto_threshold.value():
            self._live_armed = True
        elif (
            self.auto_capture_toggle.isChecked()
            and self._live_armed
            and perf_counter() - self._last_auto_capture >= 3.0
        ):
            self._live_armed, self._last_auto_capture = False, perf_counter()
            self.capture_image()

    def _update_camera_status(self, online, message):
        self.camera_online = online
        if not self._is_acquisition_page_active():
            return

        self.preview_surface.set_status_badge(None)
        self.preview_surface.set_camera_online(online)
        short_status = "Camera Online" if online else "Demo Feed"
        set_badge(self.camera_badge, short_status, "success" if online else "warning")
        self.preview_caption.setText(
            f"{message}. The preview is downscaled; every capture is saved at full sensor resolution."
            if online
            else f"{message}. Demo mode remains active so the workflow can still be tested."
        )

    def add_thumbnail(self, frame):
        tile = QFrame()
        tile.setObjectName("metricTile")

        layout = QVBoxLayout(tile)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        label_text = f"Image {len(self.image_buffer):02d}"
        thumb = self._create_clickable_thumb(
            frame,
            label_text,
            on_open=lambda checked=False, idx=len(self.image_buffer) - 1: self._open_image_lightbox(
                self._gallery_items_from_frames(self.image_buffer),
                idx,
            ),
        )
        label = QLabel(label_text)
        label.setObjectName("metricTitle")
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(thumb)
        layout.addWidget(label)
        tile.thumb_label = thumb
        tile.name_label = label
        tile.frame_data = frame.copy()
        self.thumbnail_tiles.append(tile)
        self._refresh_gallery_grid()

    def _rebuild_metric_grid(self, columns):
        for tile in self.metric_tiles:
            self.metric_grid.removeWidget(tile)

        for column in range(len(self.metric_tiles)):
            self.metric_grid.setColumnStretch(column, 1 if column < max(columns, 1) else 0)

        for index, tile in enumerate(self.metric_tiles):
            row = index // columns
            column = index % columns
            self.metric_grid.addWidget(tile, row, column)

    def _rebuild_action_button_grid(self, columns):
        buttons = [
            self.capture_button,
            self.import_button,
            self.run_button,
            self.clear_button,
        ]
        columns = max(1, min(columns, len(buttons)))

        while self.button_row_layout.count():
            item = self.button_row_layout.takeAt(0)
            if item is not None:
                del item

        for column in range(columns):
            self.button_row_layout.setColumnStretch(column, 1)

        for index, button in enumerate(buttons):
            row = index // columns
            column = index % columns
            self.button_row_layout.addWidget(button, row, column)

    def _button_panel_height(self, columns):
        buttons = [
            self.capture_button,
            self.import_button,
            self.run_button,
            self.clear_button,
        ]
        columns = max(1, min(columns, len(buttons)))
        vertical_spacing = self.button_row_layout.verticalSpacing()
        if vertical_spacing < 0:
            vertical_spacing = 0

        row_heights = []
        for start in range(0, len(buttons), columns):
            row_buttons = buttons[start:start + columns]
            row_heights.append(max(button.minimumHeight() for button in row_buttons))

        return sum(row_heights) + max(len(row_heights) - 1, 0) * vertical_spacing

    def _update_acquisition_gallery_geometry(self):
        height = self.height()
        has_images = bool(self.thumbnail_tiles)
        compact_screen = height <= 640 or self.width() <= 1024

        if has_images:
            track_height = max(self.gallery_track.height(), 72 if compact_screen else 84)
            needs_horizontal_scroll = self.gallery_track.width() > max(self.gallery_scroll.viewport().width(), 0)
            scrollbar_allowance = 10 if needs_horizontal_scroll else 0
            scroll_height = track_height + scrollbar_allowance
            card_height = scroll_height + (40 if compact_screen else 60)
        else:
            scroll_height = 0
            if height <= 520:
                card_height = 82
            elif compact_screen:
                card_height = 92
            else:
                card_height = 116 if height >= 760 else 104

        self.gallery_scroll.setVisible(has_images)
        if has_images:
            self.gallery_scroll.setFixedHeight(scroll_height)
        self.gallery_card.setMinimumHeight(card_height)
        self.gallery_card.setMaximumHeight(card_height)

    def _refresh_gallery_grid(self):
        while self.gallery_layout.count():
            self.gallery_layout.takeAt(0)

        if not self.thumbnail_tiles:
            self.gallery_track.setFixedSize(0, 0)
            self._update_acquisition_gallery_geometry()
            return

        if self.width() >= 1320:
            tile_width = 108
        elif self.width() >= 1120:
            tile_width = 96
        elif self.width() >= 900:
            tile_width = 88
        else:
            tile_width = 76

        thumb_width = max(68, tile_width - 12)
        thumb_height = max(40, int(thumb_width * 0.56))
        tile_height = thumb_height + 28

        for tile in self.thumbnail_tiles:
            tile.setFixedWidth(tile_width)
            tile.setFixedHeight(tile_height)
            tile.thumb_label.setFixedSize(thumb_width, thumb_height)
            tile.thumb_label.setPixmap(rounded_pixmap_from_frame(tile.frame_data, thumb_width, thumb_height, radius=12))
            self.gallery_layout.addWidget(tile)

        self.gallery_layout.addStretch(1)

        total_width = (
            8
            + len(self.thumbnail_tiles) * tile_width
            + max(len(self.thumbnail_tiles) - 1, 0) * self.gallery_layout.spacing()
        )
        self.gallery_track.setFixedSize(total_width, tile_height + 8)
        self._update_acquisition_gallery_geometry()

    def capture_image(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            return
        if len(self.image_buffer) >= self.max_session_images:
            self._update_workflow_state(
                f"Reached the {self.max_session_images}-field safety limit for one session. "
                "Run analysis or clear the buffer."
            )
            return

        thread = self.camera_thread
        full = thread.grab_full() if (thread is not None and self.camera_online) else None
        if full is None:
            full = self.latest_frame  # demo feed / no full-res frame yet
        if full is None:
            return

        capture_dir = self._ensure_capture_dir()
        quality = int(CAMERA_CONFIG.get("still_jpeg_quality", 97))
        target = capture_dir / f"capture_{len(self.capture_paths) + 1:02d}.jpg"
        if not cv2.imwrite(str(target), full, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            self._update_workflow_state("Could not write the captured frame to disk.")
            return

        self.capture_paths.append(target)
        self.image_buffer.append(self._display_downscale(full))
        self.add_thumbnail(self.image_buffer[-1])
        self.capture_button.pulse()
        self.preview_surface.trigger_flash()
        res = f"{full.shape[1]}x{full.shape[0]}"
        count = len(self.image_buffer)
        self.gallery_hint_label.setText(
            f"{count} field(s) buffered ({res} full resolution). Click any image to enlarge it."
        )
        self._update_workflow_state(
            f"Captured a {res} field. Session contains {count} field(s) "
            f"(~{round(count * self.wbc_per_field)} WBC est.)."
        )

    def import_images(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            return
        if len(self.image_buffer) >= self.max_session_images:
            self._update_workflow_state(
                f"Reached the {self.max_session_images}-field safety limit for one session. "
                "Run analysis or clear the buffer."
            )
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Microscopy Images",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_paths:
            return

        imported_count = 0
        unreadable_count = 0
        for file_path in file_paths:
            if len(self.image_buffer) >= self.max_session_images:
                break
            frame = cv2.imread(file_path)
            if frame is None:
                unreadable_count += 1
                continue
            self.capture_paths.append(Path(file_path))
            self.image_buffer.append(self._display_downscale(frame))
            self.add_thumbnail(self.image_buffer[-1])
            imported_count += 1

        skipped_for_capacity = max(0, len(file_paths) - imported_count - unreadable_count)

        if imported_count:
            self.import_button.pulse()
            self.gallery_hint_label.setText(
                f"{len(self.image_buffer)} field(s) buffered. Imported files are ready for analysis."
            )

        details = []
        if imported_count:
            details.append(f"Imported {imported_count} image(s).")
        if unreadable_count:
            details.append(f"Skipped {unreadable_count} unreadable file(s).")
        if skipped_for_capacity:
            details.append(f"Ignored {skipped_for_capacity} file(s): {self.max_session_images}-field session limit.")
        if not details:
            details.append("No readable image was imported.")
        self._update_workflow_state(" ".join(details))

    def _update_workflow_state(self, detail_override=None):
        image_count = len(self.image_buffer)
        analysis_running = bool(self.analysis_worker and self.analysis_worker.isRunning())
        est_wbc = round(image_count * self.wbc_per_field)
        reached = [tier for tier in self.claim_tiers if est_wbc >= tier[0]]
        next_tier = next((tier for tier in self.claim_tiers if est_wbc < tier[0]), None)
        next_fields = self.tier_fields[self.claim_tiers.index(next_tier)] if next_tier else None
        to_next = max(next_fields - image_count, 0) if next_tier else 0

        if analysis_running:
            status = "Analyzing"
            tone = "warning"
            detail = detail_override or "Running localization and session inference..."
        elif image_count >= self.max_session_images:
            status = "Session limit"
            tone = "warning"
            detail = detail_override or (
                f"{image_count} fields - the safety limit for one session. Run analysis or clear."
            )
        elif reached:
            count, code, name = reached[-1]
            status = f"Tier {code} reached"
            tone = "success"
            if next_tier is None:
                more = "Run analysis."
            elif next_fields > self.max_session_images:
                more = (f"Tier {next_tier[1]} ({next_tier[0]}) needs ~{next_fields} fields, above the "
                        f"{self.max_session_images}-field session limit.")
            else:
                more = f"About {to_next} more field(s) for tier {next_tier[1]} ({next_tier[0]}), or run analysis."
            detail = detail_override or (
                f"~{est_wbc} WBC estimated across {image_count} fields - {name} tier "
                f"({code}, >= {count} classified leukocytes) reached. {more}"
            )
        elif image_count >= self.minimum_images:
            status = "Analysis unlocked"
            tone = "neutral"
            detail = detail_override or (
                f"~{est_wbc} WBC so far. About {to_next} more field(s) to reach tier S "
                f"({self.claim_tiers[0][0]} classified leukocytes) - fewer returns insufficient_evidence."
            )
        else:
            status = "Ready"
            tone = "neutral"
            detail = detail_override or "Capture or import at least one microscopy field to begin."

        set_badge(self.workflow_badge, status, tone)
        self.workflow_detail_label.setText(detail)

        self.captured_tile.set_data(str(image_count), "Fields in this session")
        self.remaining_tile.title_label.setText("Est. WBC")
        target = next_tier[0] if next_tier else self.claim_tiers[-1][0]
        self.remaining_tile.set_data(
            f"~{est_wbc} / {target}",
            f"next: tier {next_tier[1]} ({next_tier[2]})" if next_tier else "tier R (reference) reached",
        )
        self.recommended_tile.title_label.setText("Fields per tier")
        self.recommended_tile.set_data(
            " / ".join(f"{n}" for n in self.tier_fields),
            "tiers S / P / R (" + " / ".join(str(t[0]) for t in self.claim_tiers) + " leukocytes)",
        )
        self.readiness_tile.set_data("Enabled" if image_count >= self.minimum_images else "Locked", "Unlocks at 1 image")

        self.capture_button.setEnabled(not analysis_running and image_count < self.max_session_images)
        self.import_button.setEnabled(not analysis_running and image_count < self.max_session_images)
        self.run_button.setEnabled(not analysis_running and image_count >= self.minimum_images)
        self.clear_button.setEnabled(not analysis_running and image_count > 0)
        self.button_panel.setVisible(not analysis_running)
        self.progress_card.setVisible(analysis_running)
        self._apply_progress_panel_mode()
        self._update_header_chips()

    def run_inference(self):
        if len(self.image_buffer) < self.minimum_images:
            return
        if self.analysis_worker and self.analysis_worker.isRunning():
            return

        self.show_acquisition_page()
        self.progress_bar.setValue(0)
        self.progress_value_label.setText("0%")
        self.progress_detail_label.setText("Running full-image detection...")
        self.progress_card.setVisible(True)
        self.run_button.pulse()

        analysis_inputs = self.capture_paths or self.image_buffer
        self.analysis_worker = InferenceWorker(analysis_inputs, model_runner=self.model_runner, parent=self)
        self.analysis_worker.progress_changed.connect(self._update_progress)
        self.analysis_worker.finished.connect(self._handle_inference_result)
        self.analysis_worker.start()
        self._update_workflow_state("Running full-image detection...")

    def _update_progress(self, progress, detail):
        progress = max(self.progress_bar.value(), min(100, int(progress)))
        self.progress_bar.setValue(progress)
        self.progress_value_label.setText(f"{progress}%")
        self.progress_detail_label.setText(detail)
        self._update_header_chips()

    def _handle_inference_result(self, result):
        finished_worker = self.analysis_worker
        self.analysis_worker = None
        self.result_gallery_frames = [
            self._display_downscale(frame)
            for frame in result.pop("_display_frames", self.image_buffer)
            if frame is not None
        ]
        self.last_result = result
        self.current_result_saved_dir = None
        self.progress_bar.setValue(100)
        self.progress_value_label.setText("100%")
        self.progress_detail_label.setText("Analysis complete")
        self._update_workflow_state("Analysis complete. Review the result card.")
        self._apply_result(result)
        self.show_result_page()
        if finished_worker is not None:
            finished_worker.deleteLater()

    def _apply_result_label_style(self, color):
        width = self.width()
        height = self.height()
        if width <= 720 or height <= 460:
            font_size = 22
        elif width <= 820 or height <= 520:
            font_size = 24
        elif width <= 1024 or height <= 640:
            font_size = 26
        else:
            font_size = 30
        self.result_label.setStyleSheet(
            f"color: {color}; font-size: {font_size}px; font-weight: 800;"
        )

    def _apply_result(self, result):
        risk = result.get("risk_level", "low")
        if risk == "high":
            tone = "danger"
            accent = COLORS["danger"]
            border = "rgba(214, 105, 105, 0.34)"
            background = "rgba(31, 24, 30, 0.96)"
            recommendation = "Escalate for confirmatory review"
        elif risk == "indeterminate":
            tone = "warning"
            accent = COLORS["warning"]
            border = "rgba(213, 154, 89, 0.30)"
            background = "rgba(39, 31, 18, 0.96)"
            recommendation = "Acquire more fields and request expert review"
        elif risk == "warning":
            tone = "warning"
            accent = COLORS["warning"]
            border = "rgba(213, 154, 89, 0.30)"
            background = "rgba(39, 31, 18, 0.96)"
            recommendation = "Acquire sharper imagery or verify the integration state"
        else:
            tone = "success"
            accent = COLORS["success"]
            border = "rgba(105, 166, 142, 0.28)"
            background = "rgba(21, 34, 48, 0.96)"
            recommendation = "Continue routine expert validation"

        has_session_evidence = any(
            key in result
            for key in ("strong_positive_cells", "moderate_positive_cells", "positive_images")
        )
        if has_session_evidence:
            action_detail = (
                f"Strong ROI: {result.get('strong_positive_cells', 0)}  |  "
                f"Moderate ROI: {result.get('moderate_positive_cells', 0)}  |  "
                f"Positive images: {result.get('positive_images', 0)}"
            )
        else:
            action_detail = (
                f"Positive ROI: {result.get('positive_cells', 0)}  |  "
                f"Max positive probability: {result.get('max_positive_probability', 'n/a')}"
            )

        set_badge(self.result_risk_badge, result["label"], tone)
        self.result_summary_label.setText(result["summary"])
        self.result_label.setText(f"{result['label']}: {result['confidence']}%")
        timing_details = result.get("timing_details", {})
        timing_suffix = ""
        if timing_details:
            total_seconds = timing_details.get("total_seconds")
            backend_seconds = timing_details.get("backend_seconds")
            if total_seconds is not None and backend_seconds is not None:
                timing_suffix = (
                    f"  |  Runtime: {total_seconds:.1f}s"
                    f"  |  Backend: {backend_seconds:.1f}s"
                )
            elif total_seconds is not None:
                timing_suffix = f"  |  Runtime: {total_seconds:.1f}s"
        detector_suffix = ""
        detection_backend = result.get("detection_backend")
        if detection_backend:
            detector_suffix = f"  |  Detector: {detection_backend}"
        self.result_meta_label.setText(
            f"Confidence score: {result['confidence']}%  |  Images used: {result['images_used']}  |  Detected ROI: {result.get('total_detected_cells', 0)}{detector_suffix}{timing_suffix}"
        )

        self.result_confidence_tile.set_data(f"{result['confidence']}%", "Model confidence for this session")
        self.result_images_tile.set_data(
            str(result["images_used"]),
            f"{result.get('total_detected_cells', 0)} ROI detected across buffered full-frame image(s)",
        )
        self.result_timestamp_tile.set_data(self.session_started_at.strftime("%H:%M"), "Session start time")
        self.result_action_tile.set_data(
            recommendation,
            action_detail,
        )

        self.result_container.setStyleSheet(
            f"""
            QFrame#panelCard {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 24px;
            }}
            QFrame#resultCard {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 24px;
            }}
            """
        )

        used_count = len(self.result_gallery_frames)
        self.result_gallery_hint_label.setText(
            f"{used_count} analyzed image(s) are shown below with detection overlays. Click any image to enlarge it."
            if used_count
            else "No analyzed image available for this result yet."
        )
        self._populate_result_gallery(self.result_gallery_frames)
        self._apply_result_label_style(accent)
        self.result_footer_label.setText("Result ready. Save the report or return to acquisition.")
        self._update_save_button_state()
        self._update_header_chips()

    def show_result_page(self):
        self.view_stack.setCurrentIndex(self.page_result)
        self._update_responsive_layouts()
        self._update_header_chips()
        self._schedule_result_gallery_refresh()

    def show_acquisition_page(self):
        self.view_stack.setCurrentIndex(self.page_acquisition)
        self._update_responsive_layouts()
        self._update_header_chips()

    def show_saved_results_page(self):
        self.view_stack.setCurrentIndex(self.page_saved_results)
        self._update_responsive_layouts()
        self._update_header_chips()
        QTimer.singleShot(0, self._refresh_saved_results)

    def clear_samples(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            return

        self.image_buffer.clear()
        self.capture_paths.clear()
        self._discard_capture_dir()
        self.result_gallery_frames.clear()
        self.last_result = None
        self.current_result_saved_dir = None
        self.progress_bar.setValue(0)
        self.progress_value_label.setText("0%")
        self.progress_detail_label.setText("Idle")
        self.progress_card.setVisible(False)
        self.gallery_hint_label.setText("Captured and imported images appear here before analysis. Click any image to enlarge it.")
        self.result_footer_label.setText("No report saved yet.")
        self.result_label.setText("No active result")
        self.result_meta_label.setText("Confidence: --")
        self.result_summary_label.setText("Run analysis after capturing or importing microscopy images.")
        self._apply_result_label_style(COLORS["text"])
        self.result_confidence_tile.set_data("--", "Model confidence for this session")
        self.result_images_tile.set_data("--", "Buffered full-frame images processed")
        self.result_timestamp_tile.set_data("--", "Session start time")
        self.result_action_tile.set_data("--", "Suggested next action")
        set_badge(self.result_risk_badge, "Awaiting analysis", "neutral")
        self.result_container.setStyleSheet("")
        self.result_gallery_hint_label.setText("Annotated detection previews will appear here after analysis. Click any image to enlarge it.")
        self._populate_result_gallery([])
        self._update_save_button_state()
        self.session_started_at = datetime.now()

        for tile in self.thumbnail_tiles:
            tile.deleteLater()
        self.thumbnail_tiles.clear()
        self._refresh_gallery_grid()

        self.show_acquisition_page()
        self._update_workflow_state("Image buffer cleared. Start a new capture/import sequence.")

    def save_result(self):
        if not self.last_result:
            return
        if self.current_result_saved_dir is not None:
            self.result_footer_label.setText(f"Current result already saved to {self.current_result_saved_dir}")
            self._update_save_button_state()
            return

        output_dir = SAVED_RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        frames_to_save = self.result_gallery_frames or self.image_buffer
        for index, frame in enumerate(frames_to_save, start=1):
            cv2.imwrite(str(output_dir / f"sample_{index:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # source_* keeps the full-resolution captures, copied straight from disk
        # (no re-encode). Imported files are copied with their original extension.
        for index, path in enumerate(self.capture_paths, start=1):
            src = Path(path)
            if src.is_file():
                shutil.copy2(src, output_dir / f"source_{index:02d}{src.suffix or '.jpg'}")

        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "session_started_at": self.session_started_at.isoformat(timespec="seconds"),
            "result": self.last_result,
            "images_used": len(self.image_buffer),
            "saved_gallery_images": len(frames_to_save),
        }
        with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.current_result_saved_dir = output_dir
        self.result_footer_label.setText(f"Saved result package to {output_dir}")
        self.save_button.pulse()
        self._update_save_button_state()
        self._refresh_saved_results(selected_dir=output_dir)

    def _update_responsive_layouts(self):
        width = self.width()
        height = self.height()
        landscape = width >= height
        small_screen = width <= 1024 or height <= 640
        tiny_screen = width <= 820 or height <= 520
        ultra_tiny = width <= 720 or height <= 460

        header_row = width >= 760 or landscape
        acquisition_row = width >= 1080 or (landscape and width >= 720)
        result_row = width >= 1040 or (landscape and width >= 820)
        saved_results_row = width >= 980 or (landscape and width >= 820)

        self.header_layout.setDirection(QBoxLayout.LeftToRight if header_row else QBoxLayout.TopToBottom)
        self.header_chip_row.setDirection(QBoxLayout.LeftToRight if width >= 760 else QBoxLayout.TopToBottom)
        self.acquisition_layout.setDirection(QBoxLayout.LeftToRight if acquisition_row else QBoxLayout.TopToBottom)
        self.result_body_layout.setDirection(QBoxLayout.LeftToRight if result_row else QBoxLayout.TopToBottom)
        self.saved_results_body_layout.setDirection(QBoxLayout.LeftToRight if saved_results_row else QBoxLayout.TopToBottom)
        self.result_actions_layout.setDirection(QBoxLayout.LeftToRight if width >= 760 else QBoxLayout.TopToBottom)
        self.saved_results_actions_layout.setDirection(QBoxLayout.LeftToRight if width >= 760 else QBoxLayout.TopToBottom)
        self.gallery_header_layout.setDirection(
            QBoxLayout.LeftToRight if width >= 860 and not tiny_screen else QBoxLayout.TopToBottom
        )

        style_scale = self._compute_style_scale(width, height)
        if abs(style_scale - self._style_scale) > 0.01:
            self._style_scale = style_scale
            self._apply_styles(scale=style_scale)
            self._refresh_badges()

        if ultra_tiny:
            root_margin = 8
            root_spacing = 8
            header_margin_h = 12
            header_margin_v = 10
            header_spacing = 10
            title_spacing = 1
            header_right_spacing = 4
            panel_margin = 10
            panel_spacing = 6
            header_title_style = "font-size: 18px;"
            header_subtitle_style = "font-size: 10px;"
        elif tiny_screen:
            root_margin = 10
            root_spacing = 10
            header_margin_h = 14
            header_margin_v = 12
            header_spacing = 12
            title_spacing = 2
            header_right_spacing = 6
            panel_margin = 12
            panel_spacing = 8
            header_title_style = "font-size: 20px;"
            header_subtitle_style = "font-size: 11px;"
        elif small_screen:
            root_margin = 14
            root_spacing = 12
            header_margin_h = 18
            header_margin_v = 14
            header_spacing = 14
            title_spacing = 3
            header_right_spacing = 8
            panel_margin = 14
            panel_spacing = 9
            header_title_style = "font-size: 22px;"
            header_subtitle_style = "font-size: 11px;"
        elif height < 820:
            root_margin = 18
            root_spacing = 14
            header_margin_h = 20
            header_margin_v = 16
            header_spacing = 16
            title_spacing = 3
            header_right_spacing = 8
            panel_margin = 16
            panel_spacing = 10
            header_title_style = "font-size: 25px;"
            header_subtitle_style = "font-size: 12px;"
        else:
            root_margin = 24
            root_spacing = 20
            header_margin_h = 24
            header_margin_v = 20
            header_spacing = 18
            title_spacing = 4
            header_right_spacing = 10
            panel_margin = 18
            panel_spacing = 10
            header_title_style = ""
            header_subtitle_style = ""

        self.root_layout.setContentsMargins(root_margin, root_margin, root_margin, root_margin)
        self.root_layout.setSpacing(root_spacing)
        self.header_layout.setContentsMargins(header_margin_h, header_margin_v, header_margin_h, header_margin_v)
        self.header_layout.setSpacing(header_spacing)
        self.title_block.setSpacing(title_spacing)
        self.header_right.setSpacing(header_right_spacing)
        self.header_title_label.setStyleSheet(header_title_style)
        self.header_subtitle_label.setStyleSheet(header_subtitle_style)

        hint_visibility = not tiny_screen
        self.eyebrow_label.setVisible(not tiny_screen)
        self.header_subtitle_label.setVisible(not tiny_screen)
        self.preview_caption.setVisible(hint_visibility)
        self.workflow_detail_label.setVisible(hint_visibility)
        self.gallery_hint_label.setVisible(hint_visibility)
        self.result_gallery_hint_label.setVisible(hint_visibility)
        self.saved_result_gallery_hint_label.setVisible(hint_visibility)
        self.date_chip.setVisible(not ultra_tiny)
        self.saved_results_button.setText("Saved" if tiny_screen else "Saved Results")

        preview_layout = self.preview_surface.parentWidget().layout()
        preview_layout.setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        preview_layout.setSpacing(panel_spacing)
        self.control_panel_layout.setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.control_panel_layout.setSpacing(4 if ultra_tiny else 6 if tiny_screen else 8 if small_screen else 10)
        self.gallery_card.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.gallery_card.layout().setSpacing(panel_spacing)
        self.result_container.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.result_container.layout().setSpacing(panel_spacing)
        self.result_summary_card.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.result_summary_card.layout().setSpacing(panel_spacing)
        self.saved_results_container.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.saved_results_container.layout().setSpacing(panel_spacing)
        self.saved_results_list_card.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.saved_results_list_card.layout().setSpacing(panel_spacing)
        self.saved_result_detail_content.layout().setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        self.saved_result_detail_content.layout().setSpacing(panel_spacing)

        self.gallery_hint_label.setAlignment(Qt.AlignRight if width >= 860 and not tiny_screen else Qt.AlignLeft)
        self.gallery_layout.setSpacing(6 if tiny_screen else 8 if small_screen else 10)
        self.result_gallery_layout.setHorizontalSpacing(4 if tiny_screen else 6)
        self.result_gallery_layout.setVerticalSpacing(4 if tiny_screen else 6)
        self.saved_result_gallery_layout.setSpacing(8 if tiny_screen else 10 if small_screen else 12)
        self.saved_results_list_layout.setSpacing(8 if tiny_screen else 10)

        grid_spacing = 4 if ultra_tiny else 6 if tiny_screen else 8 if small_screen else 10
        self.metric_grid.setHorizontalSpacing(grid_spacing)
        self.metric_grid.setVerticalSpacing(grid_spacing)
        self.button_row_layout.setHorizontalSpacing(grid_spacing)
        self.button_row_layout.setVerticalSpacing(grid_spacing)

        compact_metric_height = 48 if ultra_tiny else 52 if tiny_screen else 56 if small_screen else 68
        for tile in self.metric_tiles:
            tile.setMinimumHeight(compact_metric_height)

        result_metric_compact = tiny_screen
        result_metric_height = 72 if tiny_screen else 84 if small_screen else 92
        for tile in (
            self.result_confidence_tile,
            self.result_images_tile,
            self.result_timestamp_tile,
            self.result_action_tile,
        ):
            tile.set_compact_mode(result_metric_compact)
            tile.setMinimumHeight(result_metric_height)

        saved_metric_height = 64 if ultra_tiny else 70 if tiny_screen else 78 if small_screen else 88
        self.saved_metrics_widget.setFixedHeight(saved_metric_height * 2 + (8 if tiny_screen else 12))
        self.saved_metrics_top_widget.setFixedHeight(saved_metric_height)
        self.saved_metrics_bottom_widget.setFixedHeight(saved_metric_height)
        for tile in self.saved_result_tiles:
            tile.setFixedHeight(saved_metric_height)

        if ultra_tiny:
            capture_height = 36
            secondary_height = 34
        elif tiny_screen:
            capture_height = 40
            secondary_height = 38
        elif small_screen:
            capture_height = 44
            secondary_height = 42
        else:
            capture_height = 48 if height < 820 else 52
            secondary_height = 44 if height < 820 else 46

        self.capture_button.setFixedHeight(capture_height)
        self.import_button.setFixedHeight(secondary_height)
        self.run_button.setFixedHeight(secondary_height)
        self.clear_button.setFixedHeight(secondary_height)

        button_columns = 2 if width >= 620 else 1
        self._rebuild_action_button_grid(button_columns)
        self.button_panel.setFixedHeight(self._button_panel_height(button_columns))

        if ultra_tiny:
            preview_min_height = 156 if acquisition_row else 170
        elif tiny_screen:
            preview_min_height = 180 if acquisition_row else 210
        elif small_screen:
            preview_min_height = 210 if acquisition_row else 240
        elif height >= 900:
            preview_min_height = 360
        elif height >= 820:
            preview_min_height = 320
        elif height >= 740:
            preview_min_height = 280 if width >= 900 else 250
        else:
            preview_min_height = 248 if width >= 900 else 232
        self.preview_surface.setMinimumHeight(preview_min_height)

        if ultra_tiny:
            result_gallery_height = 180
        elif tiny_screen:
            result_gallery_height = 196
        elif small_screen:
            result_gallery_height = 220
        elif height >= 860:
            result_gallery_height = 292
        elif height >= 720:
            result_gallery_height = 272
        else:
            result_gallery_height = 264
        self.result_gallery_scroll.setFixedHeight(result_gallery_height)

        if ultra_tiny:
            saved_gallery_height = 90
        elif tiny_screen:
            saved_gallery_height = 102
        elif small_screen:
            saved_gallery_height = 112
        else:
            saved_gallery_height = 138 if width >= 1180 else 126 if width >= 980 else 114
        self.saved_result_gallery_scroll.setFixedHeight(saved_gallery_height)

        self.result_label.setMaximumWidth(16777215 if not tiny_screen else max(width - 80, 260))
        section_title_size = _scaled_px(28, style_scale, 18)
        self.result_page_title_label.setStyleSheet(f"font-size: {section_title_size}px;")
        self.saved_results_page_title_label.setStyleSheet(f"font-size: {section_title_size}px;")
        self._apply_progress_panel_mode()
        if self.last_result is not None:
            risk_level = self.last_result.get("risk_level", "low")
            result_color = (
                COLORS["danger"]
                if risk_level == "high"
                else COLORS["warning"]
                if risk_level in {"warning", "indeterminate"}
                else COLORS["success"]
            )
        else:
            result_color = COLORS["text"]
        self._apply_result_label_style(result_color)
        self.saved_results_current_button.setEnabled(self.last_result is not None)
        self._update_acquisition_gallery_geometry()
        self._rebuild_metric_grid(2 if width >= 620 else 1)
        saved_entry_height = 60 if tiny_screen else 68 if small_screen else 74
        for button in self.saved_result_buttons:
            button.setMinimumHeight(saved_entry_height)
        self._refresh_gallery_grid()
        if self.last_result is not None:
            self._schedule_result_gallery_refresh()
        if (
            self.selected_saved_result_dir is not None
            and self.saved_result_records
            and self.saved_result_gallery_cache_dir == self.selected_saved_result_dir
        ):
            self._schedule_saved_result_gallery_refresh()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._screen_fit_applied:
            self._screen_fit_applied = True
            QTimer.singleShot(0, self._fit_window_to_screen)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layouts()

    def closeEvent(self, event):
        backend = getattr(getattr(self, "model_runner", None), "_backend", None)
        if backend is not None:
            backend.flush_artifacts()
            backend.close()
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
        self._disconnect_camera()
        self._discard_capture_dir()
        event.accept()


if __name__ == "__main__":
    # Importing ``cv2`` above can reset the Qt platform plugin path.
    _configure_qt_platform_plugins()

    force_hidpi = os.environ.get("LEUKEMIA_UI_ENABLE_HIDPI")
    if force_hidpi is None:
        enable_hidpi = platform.machine().lower() not in {"aarch64", "arm64"}
    else:
        enable_hidpi = force_hidpi.strip().lower() in {"1", "true", "yes", "on"}

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, enable_hidpi)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app_font = QFont()
    app_font.setPointSize(10)
    app.setFont(app_font)
    window = LeukemiaDetectionUI()
    window.showFullScreen()
    sys.exit(app.exec_())