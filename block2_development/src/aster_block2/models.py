"""Block-2 v2 heads. One encoder pass per unique crop; everything else runs on the
cached [N, 512] feature matrix.

The deployed block 2 runs the encoder 5 folds x 3 repeats x 50 crops = 750 times per
session for ~183 unique crops (aster_pipeline/mil_inference.py). Here the encoder is
called once per crop and every head reads the cache, which is what pays for the cell
head, the attention head and the OOD test inside the same latency budget.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

EMBED_DIM = 512
N_CELL_CLASSES = 16  # 15 morphological classes + other_artifact


class Encoder(nn.Module):
    """ResNet18 trunk, ImageNet-initialised, global-pooled to 512-d.

    ResNet18 is kept deliberately: the deployed TensorRT FP16 engines, the parity
    harness and the on-device latency table are all built on it, and the design's
    claim is not 'a bigger encoder'.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(*list(backbone.children())[:-1])
        self.dim = EMBED_DIM

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.stem(images).flatten(1)


class CellHead(nn.Module):
    """Per-crop morphological class + an abnormality score."""

    def __init__(self, dim: int = EMBED_DIM, n_classes: int = N_CELL_CLASSES) -> None:
        super().__init__()
        self.classifier = nn.Linear(dim, n_classes)
        self.abnormality = nn.Linear(dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classifier(features)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "abnormality": torch.sigmoid(self.abnormality(features)).squeeze(-1),
        }


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al. 2018), with an entropy regulariser so the
    decision never rests on a single crop - the failure mode of the deployed MAX head.
    """

    def __init__(self, dim: int = EMBED_DIM, attention_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, attention_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(dim, attention_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = self.w(self.v(features) * self.u(features)).squeeze(-1)
        attention = torch.softmax(scores, dim=0)
        pooled = (attention.unsqueeze(-1) * features).sum(0)
        logit = self.classifier(pooled).squeeze()
        entropy = -(attention * torch.log(attention.clamp_min(1e-12))).sum()
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "attention": attention,
            "entropy": entropy,
            "pooled": pooled,
        }


def attention_entropy_penalty(entropy: torch.Tensor, n_instances: int, floor: float = 0.5) -> torch.Tensor:
    """Penalise attention concentrated on fewer than ``floor`` of the maximum entropy.

    Maximum entropy is log(n); a one-hot attention has entropy 0. The penalty is zero
    once the attention is spread over enough crops.
    """
    import math

    target = floor * math.log(max(n_instances, 2))
    return torch.clamp(target - entropy, min=0.0)


def partial_label_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Loss for cells whose class is known only up to a set.

    ``allowed`` is a 0/1 mask over classes. For a definite label the mask has a single
    1 and this reduces to cross-entropy. For 'this cell is one of the immature
    granulocytes' (PBC's merged ig class) or 'this cell is not a blast' (ALL-IDB
    non-cancer), it maximises the total probability mass inside the allowed set - the
    model is never forced to pick a class the annotation does not support.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    mass = torch.logsumexp(log_probabilities.masked_fill(allowed == 0, -1e30), dim=-1)
    return -mass.mean()


@torch.inference_mode()
def encode_bag(encoder: nn.Module, images: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """One pass per crop, chunked to bound peak memory."""
    return torch.cat([encoder(images[i:i + chunk]) for i in range(0, len(images), chunk)])
