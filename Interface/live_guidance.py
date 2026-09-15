"""Non-blocking live WBC detection used by the acquisition view."""
from __future__ import annotations

import threading
from time import perf_counter

from PyQt5.QtCore import QThread, pyqtSignal

from aster_pipeline.config import load_config
from aster_pipeline.device import resolve_device
from YOLO import WBCDetector


class LiveWBCWorker(QThread):
    """Infer from only the newest camera frame, avoiding an unbounded backlog."""
    detection_ready = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, config_path, device="auto", max_fps=4.0, parent=None):
        super().__init__(parent)
        self.config_path, self.device = config_path, device
        self.interval = 1.0 / max(max_fps, 0.1)
        self._lock, self._frame, self._running = threading.Lock(), None, True

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()

    def run(self) -> None:
        try:
            # Ultralytics accepts concrete device identifiers (``cpu``, ``0``,
            # …), but not this application's ``auto`` alias.  Resolve it here
            # just as the batch pipeline does, so systems without CUDA use CPU.
            detector = WBCDetector(
                load_config(self.config_path).yolo,
                str(resolve_device(self.device)),
            )
            next_run = 0.0
            while self._running:
                if perf_counter() < next_run:
                    self.msleep(10)
                    continue
                with self._lock:
                    frame, self._frame = self._frame, None
                if frame is None:
                    self.msleep(10)
                    continue
                started = perf_counter()
                boxes = detector.detect_frame_bgr(frame)
                elapsed = perf_counter() - started
                self.detection_ready.emit(boxes, elapsed)
                next_run = perf_counter() + self.interval
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
