"""Build a block-2 v2 ``SessionScorer`` from ``config/inference.yaml``'s ``block2:``
section.

Equivalent of ``block2_development/jetson/load_kit.py``, adapted to read paths from the
deployed config instead of a fixed ``models/config.json`` next to the kit.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import Block2Config
from .calibration import TemperatureScaler
from .grid import load_thresholds
from .inference import SessionScorer
from .models import CellHead, Encoder, GatedAttentionMIL
from .ood import DomainGate
from .quantify import Quantifier


def load_block2_scorer(config: Block2Config, device: str, backend: str = "tensorrt") -> SessionScorer:
    """``backend="tensorrt"``: the engine built on this Jetson by ``build_engine.sh``.
    ``backend="torch"``: the fp32 reference weights, and the fallback when no engine
    is available. Everything after the encoder (cell head, MIL head, OOD gate, grid)
    is identical between the two.
    """
    weights = torch.load(config.heads, map_location="cpu")

    cell_head = CellHead()
    cell_head.load_state_dict(weights["cell_head"])
    mil_head = GatedAttentionMIL()
    mil_head.load_state_dict(weights["mil"])

    if backend == "tensorrt":
        from .trt_encoder import TensorRTEncoder

        encoder = TensorRTEncoder(config.encoder_engine)
    elif backend == "torch":
        encoder = Encoder(pretrained=False)  # no ImageNet download: weights follow
        encoder.load_state_dict(weights["encoder"])
    else:
        raise ValueError(f"unknown block2 backend {backend!r}")

    thresholds = load_thresholds(config.grid)
    for name, value in json.loads(Path(config.thresholds).read_text()).items():
        setattr(thresholds, name, value)
    if abs(thresholds.gamma - config.gamma) > 1e-12:
        raise RuntimeError(
            f"gamma {thresholds.gamma} in {config.grid} differs from config gamma {config.gamma}"
        )
    missing = thresholds.unresolved()
    if missing:
        raise RuntimeError(f"unresolved block2 thresholds: {missing}")

    return SessionScorer(
        encoder=encoder,
        cell_head=cell_head,
        mil_head=mil_head,
        thresholds=thresholds,
        domain_gate=DomainGate.load(config.ood_stats),
        calibrator=TemperatureScaler(thresholds.mil_temperature),
        quantifier=Quantifier.load(config.quantifier),
        device=device,
        bag_size=config.bag_size,
        encode_chunk=config.encode_chunk,
        image_size=config.image_size,
        seed=config.seed,
    )
