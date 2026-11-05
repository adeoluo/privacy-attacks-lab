"""Gaussian differential privacy (GDP) accounting by the central limit theorem.

An approximation, not an upper bound: it can underestimate epsilon for few steps or
large sampling rates (Gopi et al. 2021 show cases). Included for comparison with the
RDP and PRV accountants.

Reference: Bu, Dong, Long and Su, "Deep Learning with Gaussian Differential
Privacy", Harvard Data Science Review 2020.
"""

from __future__ import annotations

import math

from scipy import optimize
from scipy.stats import norm


def mu_poisson(q: float, sigma: float, steps: int) -> float:
    """GDP parameter mu of ``steps`` Poisson subsampled Gaussian steps (CLT)."""
    return math.sqrt(math.expm1(sigma**-2)) * math.sqrt(steps) * q


def delta_from_mu(eps: float, mu: float) -> float:
    """Delta of a mu-GDP mechanism at ``eps`` (Dong, Roth and Su 2019, Corollary 2.13)."""
    return float(norm.cdf(-eps / mu + mu / 2.0) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0))


def epsilon_from_mu(mu: float, delta: float) -> float:
    """Smallest epsilon with ``delta_from_mu(epsilon, mu) <= delta`` (Brent's method)."""
    if delta_from_mu(0.0, mu) <= delta:
        return 0.0
    root = optimize.brentq(lambda e: delta_from_mu(e, mu) - delta, 0.0, 500.0)
    return float(root)


def gdp_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """Epsilon at ``delta`` for Poisson subsampling under the GDP CLT approximation."""
    return epsilon_from_mu(mu_poisson(q, sigma, steps), delta)
