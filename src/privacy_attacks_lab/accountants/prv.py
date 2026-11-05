"""Privacy random variable (PRV) accountant for the Poisson subsampled Gaussian mechanism.

The privacy loss of one step is a random variable Y; composing T steps adds T
independent copies. The accountant discretizes Y on a uniform grid, composes by
raising the discrete Fourier transform to the power T, and reads (epsilon, delta)
off the composed distribution with explicit error bounds.

Reference: Gopi, Lee and Wutschitz, "Numerical Composition of Differential
Privacy", NeurIPS 2021 (arXiv:2106.02848). The grid size, truncation and mean
correction follow that paper, as does Opacus' ``PRVAccountant``, against which
:mod:`tests.test_accountants_opacus` checks this module.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import integrate, signal, special

from privacy_attacks_lab._backend import resolve, rust
from privacy_attacks_lab.accountants import rdp as _rdp

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Grid:
    """A uniform grid of ``size`` points from ``t_min`` to ``t_max`` (size is even)."""

    t_min: float
    t_max: float
    size: int
    shift: float = 0.0

    @classmethod
    def aligned(cls, t_min: float, t_max: float, dt: float) -> Grid:
        lo = math.floor(t_min / dt) * dt
        hi = math.ceil(t_max / dt) * dt
        size = round((hi - lo) / dt) + 1
        if size % 2 == 1:
            size += 1
            hi += dt
        return cls(lo, hi, size)

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.size - 1)

    @property
    def ts(self) -> np.ndarray:
        return np.linspace(self.t_min, self.t_max, self.size)

    def shifted(self, by: float) -> Grid:
        return Grid(self.t_min + by, self.t_max + by, self.size, self.shift + by)


class SubsampledGaussianPRV:
    """Privacy loss of one step of the subsampled Gaussian mechanism with rate ``q``."""

    def __init__(self, q: float, sigma: float) -> None:
        if not 0.0 < q <= 1.0:
            raise ValueError(f"sample rate must be in (0, 1], got {q}")
        if sigma <= 0.0:
            raise ValueError(f"noise multiplier must be positive, got {sigma}")
        self.q = q
        self.sigma = sigma

    def cdf(self, t: np.ndarray | float) -> np.ndarray:
        """P(Y <= t); zero below ``log(1 - q)``, where the loss is bounded."""
        q, s = self.q, self.sigma
        t = np.asarray(t, dtype=np.float64)
        lower = math.log(1.0 - q) if q < 1.0 else -math.inf
        inside = t > lower
        ratio = np.where(inside, (np.exp(t) + q - 1.0) / q, 1.0)
        z = np.log(ratio)
        a = (2.0 * z * s * s - 1.0) / (2.0 * _SQRT2 * s)
        b = (2.0 * z * s * s + 1.0) / (2.0 * _SQRT2 * s)
        value = 1.0 - q * special.erfc(a) / 2.0 - (1.0 - q) * special.erfc(b) / 2.0
        return np.where(inside, value, 0.0)


class TruncatedPRV:
    """A PRV conditioned on ``t_min <= Y < t_max``."""

    def __init__(self, prv: SubsampledGaussianPRV, t_min: float, t_max: float) -> None:
        self.prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self._lo = float(prv.cdf(t_min))
        self._mass = float(prv.cdf(t_max)) - self._lo

    def cdf(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        inner = (self.prv.cdf(t) - self._lo) / self._mass
        return np.where(t < self.t_min, 0.0, np.where(t < self.t_max, inner, 1.0))

    def mean(self) -> float:
        """E[Y] by parts, ``E[Y] = [t F(t)] - integral of F``, on a log-spaced partition."""
        lo_exp = int(math.log10(abs(self.t_min)))
        hi_exp = int(math.log10(self.t_max))
        points = np.concatenate(
            [
                [self.t_min],
                -np.logspace(lo_exp, -5, num=10),
                [0.0],
                np.logspace(-5, hi_exp, num=10),
                [self.t_max],
            ]
        )
        total = 0.0
        for left, right in itertools.pairwise(points):
            area, _ = integrate.quad(lambda x: float(self.cdf(x)), left, right, limit=500)
            total += right * float(self.cdf(right)) - left * float(self.cdf(left)) - area
        return total


@dataclass(frozen=True)
class DiscretePRV:
    pmf: np.ndarray
    grid: Grid

    def epsilon(self, delta: float, delta_error: float, eps_error: float) -> PRVEpsilon:
        """Epsilon at ``delta`` with a lower bound, an estimate and an upper bound."""
        if delta <= 0.0:
            return PRVEpsilon(math.inf, math.inf, math.inf)
        if np.finfo(np.longdouble).eps * self.grid.size > delta - delta_error:
            raise ValueError("delta is too small for this grid; floating point error dominates")
        t, p = self.grid.ts, self.pmf
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p * np.exp(-t)).cumsum())
        ndelta = np.exp(t) * d2 - d1

        def find(target: float) -> float:
            i = int(np.searchsorted(ndelta, -target, side="left"))
            if i <= 0:
                raise RuntimeError("cannot compute epsilon on this grid")
            return float(np.log((d1[i] - target) / d2[i]))

        return PRVEpsilon(
            lower=find(delta + delta_error) - eps_error,
            estimate=find(delta),
            upper=find(delta - delta_error) + eps_error,
        )

    def delta(self, eps: float) -> float:
        """Delta at ``eps`` (estimate, no error bound)."""
        t = self.grid.ts
        return float(np.where(t >= eps, self.pmf * (1.0 - np.exp(eps - t)), 0.0).sum())


@dataclass(frozen=True)
class PRVEpsilon:
    lower: float
    estimate: float
    upper: float


def discretize(prv: TruncatedPRV, grid: Grid) -> DiscretePRV:
    """Bin the PRV onto the grid and shift the grid so the discrete mean is exact."""
    tc = grid.ts
    pmf = prv.cdf(tc + grid.dt / 2.0) - prv.cdf(tc - grid.dt / 2.0)
    shift = prv.mean() - float(np.dot(tc, pmf))
    if abs(shift) >= grid.dt / 2.0:
        raise RuntimeError("discrete mean differs from the continuous mean by half a bin")
    return DiscretePRV(pmf=pmf, grid=grid.shifted(shift))


def self_compose(d: DiscretePRV, n: int) -> DiscretePRV:
    """Sum of ``n`` independent copies, by powering the real FFT of the pmf."""
    if d.grid.size % 2 != 0:
        raise ValueError("can only compose evenly sized grids")
    pmf = np.fft.irfft(np.fft.rfft(d.pmf) ** n, n=d.grid.size)
    m = n - 1
    if n % 2 == 0:
        m += d.grid.size // 2
    pmf = np.roll(pmf, m)
    return DiscretePRV(pmf=pmf, grid=d.grid.shifted(d.grid.shift * (n - 1)))


def _compose_pair(left: DiscretePRV, right: DiscretePRV) -> DiscretePRV:
    pmf = signal.convolve(left.pmf, right.pmf, mode="same")
    return DiscretePRV(pmf=pmf, grid=left.grid.shifted(right.grid.shift))


def _rdp_curve(q: float, sigma: float, steps: int) -> np.ndarray:
    orders = _rdp.DEFAULT_ORDERS
    if resolve("auto") == "rust":
        return np.asarray(rust().compute_rdp(q, sigma, steps, list(orders)), dtype=np.float64)
    return _rdp.compute_rdp(q, sigma, steps, orders)


def _safe_half_width(
    history: Sequence[tuple[float, float, int]], eps_error: float, delta_error: float
) -> float:
    total = sum(steps for _, _, steps in history)
    orders = _rdp.DEFAULT_ORDERS
    rdp_total = sum(
        (_rdp_curve(q, sigma, steps) for sigma, q, steps in history),
        np.zeros(len(orders)),
    )
    width, _ = _rdp.rdp_to_epsilon(orders, rdp_total, delta_error / 4.0)
    for sigma, q, _ in history:
        one = _rdp_curve(q, sigma, 1)
        eps_one, _ = _rdp.rdp_to_epsilon(orders, one, delta_error / (8.0 * total))
        width = max(width, eps_one)
    return max(width, eps_error) + 3.0


def compose(
    history: Sequence[tuple[float, float, int]], eps_error: float, delta_error: float
) -> DiscretePRV:
    """Discretize and compose a history of ``(sigma, q, steps)`` segments."""
    if not history:
        raise ValueError("empty history")
    total = sum(steps for _, _, steps in history)
    half = _safe_half_width(history, eps_error, delta_error)
    dt = eps_error / math.sqrt(total * math.log(12.0 / delta_error) / 2.0)
    grid = Grid.aligned(-half, half, dt)
    parts = []
    for sigma, q, steps in history:
        tprv = TruncatedPRV(SubsampledGaussianPRV(q, sigma), grid.t_min, grid.t_max)
        parts.append(self_compose(discretize(tprv, grid), steps))
    while len(parts) > 1:
        merged = [parts.pop()] if len(parts) % 2 == 1 else []
        merged.extend(_compose_pair(a, b) for a, b in zip(parts[:-1:2], parts[1::2], strict=True))
        parts = merged
    return parts[0]


def prv_epsilon(
    history: Sequence[tuple[float, float, int]],
    delta: float,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> PRVEpsilon:
    """Epsilon bounds for a history of ``(sigma, q, steps)`` segments at ``delta``."""
    if delta_error is None:
        delta_error = delta / 1000.0
    return compose(history, eps_error, delta_error).epsilon(delta, delta_error, eps_error)
