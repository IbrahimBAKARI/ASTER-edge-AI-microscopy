"""Stable application service between user interfaces and inference code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aster_pipeline.config import load_config
from aster_pipeline.pipeline import LeukemiaPipeline
from aster_pipeline.schemas import PipelineResult


@dataclass(frozen=True)
class AnalysisRequest:
    input_path: str | Path
    session_id: str
    output_dir: str | Path


class ASTERBackend:
    """Validate requests, load the configured models and run one analysis."""

    def __init__(self, config_path: str | Path, device: str = "auto") -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.device = device
        self._pipeline: LeukemiaPipeline | None = None

    def health(self) -> dict[str, object]:
        config = load_config(self.config_path)
        required = {
            "config": self.config_path,
            "yolo_weights": config.yolo.weights,
            "block2_encoder": config.block2.encoder_onnx,
            "block2_heads": config.block2.heads,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        return {"ready": not missing, "missing": missing, "paths": required}

    def analyze(self, request: AnalysisRequest) -> PipelineResult:
        input_path = Path(request.input_path).expanduser().resolve()
        session_id = request.session_id.strip()
        output_dir = Path(request.output_dir).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        if not session_id:
            raise ValueError("Session ID must not be empty")
        if input_path == output_dir or input_path in output_dir.parents:
            raise ValueError("Output directory must not be inside the input directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._pipeline is None:
            config = load_config(self.config_path)
            self._pipeline = LeukemiaPipeline(
                config,
                device=self.device,
                mil_backend="auto",
            )
        return self._pipeline.run(input_path, session_id, output_dir)

    def flush_artifacts(self, timeout: float | None = None) -> float | None:
        """Publish queued session artefacts; call during application shutdown.

        Returns the elapsed ``artifact_flush_ms``, or ``None`` if no pipeline
        has been loaded yet (nothing was ever queued).
        """
        if self._pipeline is not None:
            return self._pipeline.flush_artifacts(timeout)
        return None

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
