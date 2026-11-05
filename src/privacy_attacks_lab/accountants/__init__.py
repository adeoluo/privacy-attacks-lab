"""Privacy accountants for DP-SGD (the Poisson subsampled Gaussian mechanism).

Three accountants answer the same question, how much privacy ``steps`` noisy steps
with sampling rate ``q`` and noise multiplier ``sigma`` spend:

- ``rdp``: Renyi DP (an upper bound; Rust core with a Python reference);
- ``prv``: privacy random variables with numerical composition (an upper bound
  within ``eps_error``, tighter than RDP);
- ``gdp``: the Gaussian DP central limit approximation (not a bound).

Example:
    >>> from privacy_attacks_lab.accountants import epsilon
    >>> round(epsilon(q=256 / 60000, sigma=1.1, steps=14062, delta=1e-5), 3)
    2.597
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from privacy_attacks_lab._backend import Backend, resolve, rust
from privacy_attacks_lab.accountants import gdp as _gdp
from privacy_attacks_lab.accountants import prv as _prv
from privacy_attacks_lab.accountants import rdp as _rdp
from privacy_attacks_lab.accountants.rdp import DEFAULT_ORDERS

Mechanism = Literal["rdp", "prv", "gdp"]
MECHANISMS: tuple[Mechanism, ...] = ("rdp", "prv", "gdp")

__all__ = [
    "DEFAULT_ORDERS",
    "MECHANISMS",
    "Accountant",
    "Mechanism",
    "calibrate_noise",
    "compute_rdp",
    "epsilon",
    "max_tpr",
    "rdp_epsilon",
]


def _validate(q: float, sigma: float, steps: int, delta: float) -> None:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"sample rate must be in [0, 1], got {q}")
    if not sigma >= 0.0:
        raise ValueError(f"noise multiplier must be non-negative, got {sigma}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def compute_rdp(
    q: float,
    sigma: float,
    steps: int,
    orders: Sequence[float] = DEFAULT_ORDERS,
    backend: Backend = "auto",
) -> np.ndarray:
    """RDP after ``steps`` steps at each order."""
    if resolve(backend) == "rust":
        return np.asarray(rust().compute_rdp(q, sigma, steps, list(orders)), dtype=np.float64)
    return _rdp.compute_rdp(q, sigma, steps, orders)


def rdp_epsilon(
    q: float,
    sigma: float,
    steps: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    backend: Backend = "auto",
) -> tuple[float, float]:
    """``(epsilon, best order)`` under RDP accounting."""
    _validate(q, sigma, steps, delta)
    if resolve(backend) == "rust":
        eps, order = rust().epsilon(q, sigma, steps, delta, list(orders))
        return float(eps), float(order)
    return _rdp.rdp_to_epsilon(orders, _rdp.compute_rdp(q, sigma, steps, orders), delta)


def epsilon(
    q: float,
    sigma: float,
    steps: int,
    delta: float,
    *,
    accountant: Mechanism = "rdp",
    eps_error: float = 0.01,
    backend: Backend = "auto",
) -> float:
    """Epsilon spent at ``delta`` after ``steps`` DP-SGD steps.

    For ``prv`` this is the upper bound (estimate plus ``eps_error``), as Opacus reports.
    """
    _validate(q, sigma, steps, delta)
    if steps == 0 or q == 0.0:
        return 0.0
    if sigma == 0.0:
        return math.inf
    if accountant == "rdp":
        return rdp_epsilon(q, sigma, steps, delta, backend=backend)[0]
    if accountant == "prv":
        return _prv.prv_epsilon([(sigma, q, steps)], delta, eps_error=eps_error).upper
    if accountant == "gdp":
        return _gdp.gdp_epsilon(q, sigma, steps, delta)
    raise ValueError(f"unknown accountant {accountant!r}")


def calibrate_noise(
    target_epsilon: float,
    delta: float,
    q: float,
    steps: int,
    *,
    accountant: Mechanism = "rdp",
    rtol: float = 1e-4,
    eps_error: float = 0.01,
    backend: Backend = "auto",
) -> float:
    """Smallest noise multiplier (to relative tolerance ``rtol``) with epsilon at most the target.

    Doubling to bracket the answer, then bisection on sigma (epsilon decreases in sigma).
    """
    if target_epsilon <= 0.0:
        raise ValueError("target epsilon must be positive")
    _validate(q, 1.0, steps, delta)
    if accountant == "rdp" and resolve(backend) == "rust":
        found = rust().calibrate_sigma(target_epsilon, delta, q, steps, list(DEFAULT_ORDERS), rtol)
        if found is None:
            raise ValueError("the privacy budget is too small: sigma would exceed 1e6")
        return float(found)

    def eps_at(s: float) -> float:
        return epsilon(
            q, s, steps, delta, accountant=accountant, eps_error=eps_error, backend=backend
        )

    hi = 1.0
    while eps_at(hi) > target_epsilon:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("the privacy budget is too small: sigma would exceed 1e6")
    lo = hi / 2.0 if hi > 1.0 else 0.0
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if eps_at(mid) <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class Accountant:
    """Tracks a training run step by step, like Opacus' accountants.

    ``history`` holds ``(sigma, q, steps)`` segments; consecutive steps with the same
    parameters are merged.
    """

    mechanism: Mechanism = "rdp"
    history: list[tuple[float, float, int]] = field(default_factory=list)

    def step(self, sigma: float, q: float, n: int = 1) -> None:
        if self.history and self.history[-1][:2] == (sigma, q):
            s, r, k = self.history[-1]
            self.history[-1] = (s, r, k + n)
        else:
            self.history.append((sigma, q, n))

    @property
    def steps(self) -> int:
        return sum(k for _, _, k in self.history)

    def epsilon(self, delta: float, eps_error: float = 0.01) -> float:
        if not self.history:
            return 0.0
        if self.mechanism == "prv":
            return _prv.prv_epsilon(self.history, delta, eps_error=eps_error).upper
        if self.mechanism == "gdp":
            if len(self.history) != 1:
                raise ValueError("the GDP accountant needs a constant sigma and sample rate")
            sigma, q, k = self.history[0]
            return _gdp.gdp_epsilon(q, sigma, k, delta)
        total = sum(
            (compute_rdp(q, s, k) for s, q, k in self.history), np.zeros(len(DEFAULT_ORDERS))
        )
        return _rdp.rdp_to_epsilon(DEFAULT_ORDERS, total, delta)[0]


def max_tpr(fpr: float | np.ndarray, eps: float, delta: float) -> np.ndarray:
    """Highest true-positive rate any membership test can reach at ``fpr`` under (eps, delta)-DP.

    From the hypothesis-testing view of DP (Kairouz, Oh and Viswanath 2015):
    ``TPR <= min(1, e^eps FPR + delta, 1 - e^-eps (1 - delta - FPR))``.
    """
    f = np.asarray(fpr, dtype=np.float64)
    if math.isinf(eps):
        return np.ones_like(f)
    a = math.exp(eps) * f + delta
    b = 1.0 - math.exp(-eps) * (1.0 - delta - f)
    return np.clip(np.minimum(a, b), 0.0, 1.0)
