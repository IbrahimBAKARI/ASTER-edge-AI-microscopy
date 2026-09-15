from pathlib import Path

import pytest

from backend import ASTERBackend, AnalysisRequest


ROOT = Path(__file__).resolve().parents[1]


def test_health_finds_all_deployment_artifacts():
    health = ASTERBackend(ROOT / "config" / "inference.yaml").health()
    assert health["ready"] is True
    assert health["missing"] == []


def test_analyze_rejects_missing_input_before_loading_models(tmp_path):
    backend = ASTERBackend(ROOT / "config" / "inference.yaml")
    request = AnalysisRequest(tmp_path / "missing", "test", tmp_path / "output")
    with pytest.raises(FileNotFoundError, match="Input path not found"):
        backend.analyze(request)


def test_analyze_rejects_empty_session_before_loading_models(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    backend = ASTERBackend(ROOT / "config" / "inference.yaml")
    request = AnalysisRequest(input_dir, "  ", tmp_path / "output")
    with pytest.raises(ValueError, match="Session ID"):
        backend.analyze(request)


def test_analyze_rejects_output_nested_in_input(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    backend = ASTERBackend(ROOT / "config" / "inference.yaml")
    request = AnalysisRequest(input_dir, "test", input_dir / "output")
    with pytest.raises(ValueError, match="must not be inside"):
        backend.analyze(request)
