"""Bag sampling, seeded and reproducible - the deployed contract.

`aster_pipeline/sampling.py` draws the bag with `numpy.random.default_rng(seed + repeat *
10007).choice(n, size, replace=False)`. Block 2 v2 must draw it the same way: taking the
first `bag_size` crops would take the first fields scanned, and a blood smear is not
homogeneous along the scan path (feathered edge vs body), so the head would see a
spatially biased sample and the session-to-session variance would be understated.
"""

from __future__ import annotations

import numpy as np

SEED = 42
REPEAT_STRIDE = 10007  # aster_pipeline/sampling.py


def deterministic_bag(n_available: int, bag_size: int, seed: int = SEED,
                      repeat: int = 0) -> list[int]:
    """Indices of one bag, identical to the deployed `deterministic_samples`."""
    if n_available <= 0:
        return []
    size = min(bag_size, n_available)
    rng = np.random.default_rng(seed + repeat * REPEAT_STRIDE)
    return rng.choice(n_available, size=size, replace=False).tolist()


def deterministic_bags(n_available: int, bag_size: int, repeats: int = 3,
                       seed: int = SEED) -> list[list[int]]:
    """The three sub-bags. In v2 they measure decision STABILITY rather than being three
    redundant passes: the encoder now runs once per crop, not once per bag."""
    return [deterministic_bag(n_available, bag_size, seed, r) for r in range(repeats)]
