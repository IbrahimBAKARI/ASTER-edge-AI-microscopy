"""Three-way proportion test used by the ASTER block-2 decision grid.

Every criterion in the grid has the form ``frac >= theta``. With a session bag of
50-500 classified leukocytes, a point estimate cannot support such a statement:
6 basophils out of 100 and 2 out of 100 are not the same evidence for "basophilia
above 2 %". The grid therefore never compares a point estimate to a threshold.

Given ``k`` cells of a class among ``n`` classified leukocytes, we use the Jeffreys
posterior ``Beta(k + 1/2, n - k + 1/2)`` and declare:

    ASSERT        if  P(p >= theta | k, n) >= gamma
    REJECT        if  P(p <  theta | k, n) >= gamma
    INDETERMINATE otherwise

``gamma`` is the single confidence parameter of the whole grid (frozen at 0.90,
sensitivity-analysed at 0.80 and 0.95 -- see PREREGISTRATION.md). A rule fires only
when every one of its criteria returns ASSERT; if any is INDETERMINATE and none is
REJECT, the session is reported ``indeterminate``.

Pure standard library on purpose: this module also runs inside the deployed
pipeline on the Jetson, where the block-2 heads are the only non-TensorRT code.
"""

from __future__ import annotations

from enum import Enum
from math import exp, lgamma, log

__all__ = ["Verdict", "posterior_exceeds", "test_proportion", "min_count_to_assert"]

_MAXIT = 300
_EPS = 3e-16
_FPMIN = 1e-300


class Verdict(str, Enum):
    ASSERT = "assert"
    REJECT = "reject"
    INDETERMINATE = "indeterminate"


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) = P(Beta(a, b) <= x)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def posterior_exceeds(k: int, n: int, theta: float) -> float:
    """P(p >= theta | k of n), Jeffreys prior Beta(1/2, 1/2)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")
    return 1.0 - _betainc(k + 0.5, n - k + 0.5, theta)


def test_proportion(k: int, n: int, theta: float, gamma: float = 0.90) -> Verdict:
    """Three-way verdict on the claim ``p >= theta``."""
    if not 0.5 < gamma < 1.0:
        raise ValueError("gamma must lie in (0.5, 1)")
    above = posterior_exceeds(k, n, theta)
    if above >= gamma:
        return Verdict.ASSERT
    if 1.0 - above >= gamma:
        return Verdict.REJECT
    return Verdict.INDETERMINATE


def min_count_to_assert(theta: float, n: int, gamma: float = 0.90) -> int | None:
    """Smallest observed count that asserts ``p >= theta`` in a bag of n cells.

    Returns None when the claim cannot be asserted at any count for that bag size.
    This is the function that sets the minimum session sizes in the grid.
    """
    for k in range(n + 1):
        if posterior_exceeds(k, n, theta) >= gamma:
            return k
    return None


if __name__ == "__main__":  # resolution table, reproduces PREREGISTRATION.md 3.3
    gamma = 0.90
    sizes = (50, 100, 200, 400, 500)
    print(f"gamma = {gamma}   (observed count and % needed to assert)")
    print(f"{'theta':>7} | " + " | ".join(f"N_c={n:<10}" for n in sizes))
    for theta in (0.02, 0.05, 0.10, 0.20, 0.50):
        cells = []
        for n in sizes:
            k = min_count_to_assert(theta, n, gamma)
            cells.append(f"{k:>4} ({k / n * 100:4.1f}%)" if k is not None else "  unreachable")
        print(f"{theta * 100:6.0f}% | " + " | ".join(cells))
