from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the requested inference device, preferring CUDA in auto mode."""
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device: {requested}")
