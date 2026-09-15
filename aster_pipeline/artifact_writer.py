"""Ordered, crash-safe persistence for session artefacts.

Inference retains the artefacts in memory and hands them to this single worker.
Keeping one writer per pipeline preserves the order within and between sessions,
while atomic replacement makes a completed file either old or complete--never a
partially encoded PNG/JSON/CSV.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable


class SessionArtifactWriter:
    def __init__(self) -> None:
        # One worker keeps writes strictly ordered within and between sessions.
        # Image encoding (the historical persistence bottleneck) now happens in
        # YoloWBCDetector before submit(); this worker only does the fsync +
        # atomic rename of already-encoded bytes, which is cheap, so a single
        # thread is sufficient and preserves ordering.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aster-io")
        self._lock = threading.Lock()
        self._futures: list[Future] = []
        self._closed = False

    def submit(self, write: Callable[[], None]) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("Artifact writer is closed")
            future = self._executor.submit(write)
            self._futures.append(future)
            return future

    def flush(self, timeout: float | None = None) -> None:
        with self._lock:
            pending, self._futures = self._futures, []
        for future in pending:
            future.result(timeout=timeout)

    def close(self, timeout: float | None = None) -> None:
        self.flush(timeout)
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a complete file to the target filesystem before publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
