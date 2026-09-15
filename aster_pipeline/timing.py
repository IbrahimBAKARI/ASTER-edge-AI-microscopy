from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Callable


class Timings:
    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self.values: dict[str, float] = {}
        self.synchronize = synchronize

    @contextmanager
    def measure(self, key: str, gpu: bool = False):
        if gpu and self.synchronize is not None:
            self.synchronize()
        start = perf_counter_ns()
        try:
            yield
        finally:
            if gpu and self.synchronize is not None:
                self.synchronize()
            elapsed = (perf_counter_ns() - start) / 1_000_000.0
            self.values[key] = self.values.get(key, 0.0) + elapsed

    def rounded(self) -> dict[str, float]:
        return {key: round(value, 3) for key, value in self.values.items()}
