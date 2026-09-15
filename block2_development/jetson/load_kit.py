"""Build the block-2 SessionScorer from the Jetson kit.

    from load_kit import load_scorer
    scorer = load_scorer(".", backend="tensorrt")          # or backend="torch"
    result, detail = scorer.score(list_of_crop_paths_or_PIL_tensors, session_id="S1")

backend="tensorrt"  encoder = the engine built by build_engine.sh (deployment path)
backend="torch"     encoder = the fp32 PyTorch weights of heads.pt (reference path, and
                    the fallback if TensorRT is unavailable)
Everything after the encoder (cell head, MIL head, OOD gate, grid) is identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

KIT = Path(__file__).resolve().parent
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from aster_block2.calibration import TemperatureScaler  # noqa: E402
from aster_block2.grid import load_thresholds  # noqa: E402
from aster_block2.inference import SessionScorer  # noqa: E402
from aster_block2.models import CellHead, Encoder, GatedAttentionMIL  # noqa: E402
from aster_block2.ood import DomainGate  # noqa: E402
from aster_block2.quantify import Quantifier  # noqa: E402


def load_config(kit: Path = KIT) -> dict:
    return json.loads((Path(kit) / "models/config.json").read_text())


def load_scorer(kit: Path = KIT, backend: str = "tensorrt", engine: Path | None = None,
                device: str | None = None) -> SessionScorer:
    models = Path(kit) / "models"
    config = load_config(kit)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(models / "heads.pt", map_location="cpu")

    cell_head = CellHead()
    cell_head.load_state_dict(weights["cell_head"])
    mil = GatedAttentionMIL()
    mil.load_state_dict(weights["mil"])
    if backend == "tensorrt":
        from trt_encoder import TensorRTEncoder
        encoder = TensorRTEncoder(engine or models / "encoder_fp16.engine")
    elif backend == "torch":
        encoder = Encoder(pretrained=False)          # no ImageNet download: weights follow
        encoder.load_state_dict(weights["encoder"])
    else:
        raise ValueError(f"unknown backend {backend!r}")

    thresholds = load_thresholds(models / "decision_grid.yaml")
    for name, value in json.loads((models / "thresholds.json").read_text()).items():
        setattr(thresholds, name, value)
    if abs(thresholds.gamma - config["gamma"]) > 1e-12:
        raise RuntimeError(f"gamma {thresholds.gamma} differs from the run's {config['gamma']}")
    missing = thresholds.unresolved()
    if missing:
        raise RuntimeError(f"unresolved thresholds: {missing}")

    return SessionScorer(
        encoder=encoder, cell_head=cell_head, mil_head=mil, thresholds=thresholds,
        domain_gate=DomainGate.load(models / "ood_stats.npz"),
        calibrator=TemperatureScaler(thresholds.mil_temperature),
        quantifier=Quantifier.load(models / "quantifier.json"),
        device=device, bag_size=config["bag_size"], seed=config["seed"],
    )
