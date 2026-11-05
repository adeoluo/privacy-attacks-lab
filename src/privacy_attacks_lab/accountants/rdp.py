"""Renyi differential privacy (RDP) of the Poisson subsampled Gaussian mechanism.

This is the readable reference implementation. The default entry points in
:mod:`privacy_attacks_lab.accountants` call the Rust core when it is available and
fall back to these functions otherwise; the parity tests check that both agree.

References:
    Mironov, Talwar and Zhang, "Renyi Differential Privacy of the Sampled Gaussian
    Mechanism", 2019 (arXiv:1908.10530), Section 3.3 for ``log A_alpha``.
    Balle, Barthe, Gaboardi, Hsu and Sato, "Hypothesis Testing Interpretations and
    Renyi Differential Privacy", AISTATS 2020, Theorem 21 for the conversion to
    (epsilon, delta).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import special

#: Orders used when none are given: 1.1 to 10.9 in steps of 0.1, then 12 to 63
#: (the same grid as Opacus' ``RDPAccountant.DEFAULT_ALPHAS``).
DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1.0 + x / 10.0 for x in range(1, 100)] + [float(a) for a in range(12, 64)]
)

_LOG_HALF = math.log(0.5)
_SERIES_CUTOFF = -30.0


def log_add(logx: float, logy: float) -> float:
    """Return ``log(exp(logx) + exp(logy))`` without overflow."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -math.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def log_sub(logx: float, logy: float) -> float:
    """Return ``log(exp(logx) - exp(logy))``; requires ``logx >= logy``."""
    if logx < logy:
        raise ValueError("log_sub needs logx >= logy")
    if logy == -math.inf:
        return logx
    if logx == logy:
        return -math.inf
    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def log_erfc(x: float) -> float:
    """``log(erfc(x))``, accurate for large ``x`` where ``erfc`` underflows."""
    return math.log(2.0) + float(special.log_ndtr(-x * math.sqrt(2.0)))


def _log_binom(n: float, k: float) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def log_a_int(q: float, sigma: float, alpha: int) -> float:
    """``log A_alpha`` for an integer order: a finite binomial sum in log space."""
    log_a = -math.inf
    log_q, log_1q = math.log(q), math.log1p(-q)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    for i in range(alpha + 1):
        term = _log_binom(alpha, i) + i * log_q + (alpha - i) * log_1q + (i * i - i) * inv_2s2
        log_a = log_add(log_a, term)
    return log_a


def log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """``log A_alpha`` for a fractional order: two series, stopped once terms fall below e^-30."""
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    log_q, log_1q = math.log(q), math.log1p(-q)
    s2 = sigma * sigma
    sqrt2_sigma = math.sqrt(2.0) * sigma
    i = 0
    while True:
        coef = float(special.binom(alpha, i))
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * log_q + j * log_1q
        log_t1 = log_coef + j * log_q + i * log_1q
        log_e0 = _LOG_HALF + log_erfc((i - z0) / sqrt2_sigma)
        log_e1 = _LOG_HALF + log_erfc((z0 - j) / sqrt2_sigma)
        log_s0 = log_t0 + (i * i - i) / (2.0 * s2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2.0 * s2) + log_e1
        if coef > 0:
            log_a0 = log_add(log_a0, log_s0)
            log_a1 = log_add(log_a1, log_s1)
        else:
            log_a0 = log_sub(log_a0, log_s0)
            log_a1 = log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < _SERIES_CUTOFF:
            break
    return log_add(log_a0, log_a1)


def rdp_single(q: float, sigma: float, alpha: float) -> float:
    """RDP at order ``alpha`` of one step of the subsampled Gaussian mechanism."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"sample rate must be in [0, 1], got {q}")
    if sigma < 0.0:
        raise ValueError(f"noise multiplier must be non-negative, got {sigma}")
    if alpha <= 1.0:
        raise ValueError(f"RDP orders must be greater than 1, got {alpha}")
    if q == 0.0:
        return 0.0
    if sigma == 0.0 or math.isinf(alpha):
        return math.inf
    if q == 1.0:
        return alpha / (2.0 * sigma * sigma)
    if float(alpha).is_integer():
        return log_a_int(q, sigma, int(alpha)) / (alpha - 1.0)
    return log_a_frac(q, sigma, alpha) / (alpha - 1.0)


def compute_rdp(
    q: float, sigma: float, steps: int, orders: Sequence[float] = DEFAULT_ORDERS
) -> np.ndarray:
    """RDP after ``steps`` compositions, one value per order."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return np.array([rdp_single(q, sigma, a) for a in orders], dtype=np.float64) * steps


def rdp_to_epsilon(
    orders: Sequence[float], rdp: Sequence[float] | np.ndarray, delta: float
) -> tuple[float, float]:
    """Convert RDP curves to (epsilon, best order) at ``delta`` (Balle et al. 2020, Thm. 21)."""
    a = np.asarray(orders, dtype=np.float64)
    r = np.asarray(rdp, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError("orders and rdp must have the same length")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    with np.errstate(invalid="ignore"):
        eps = r - (np.log(delta) + np.log(a)) / (a - 1.0) + np.log((a - 1.0) / a)
    if np.isnan(eps).all():
        return math.inf, math.nan
    idx = int(np.nanargmin(eps))
    return float(eps[idx]), float(a[idx])
