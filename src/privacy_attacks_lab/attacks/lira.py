"""The Likelihood Ratio Attack, LiRA (Carlini, Chien, Nasr, Song, Terzis and Tramer, S&P 2022).

For each example ``x`` the attacker trains reference models with and without ``x``
and records the logit-scaled confidence ``phi = log(p_y / (1 - p_y))`` of each.
Across models, ``phi`` is close to Gaussian, so two per-example Gaussians are fit:

    Q_in(x)  = N(mu_in(x),  sigma_in(x)^2)    from models that trained on x
    Q_out(x) = N(mu_out(x), sigma_out(x)^2)   from models that did not

Online LiRA scores the target's confidence by the log likelihood ratio

    Lambda(x) = log N(phi; mu_in, sigma_in^2) - log N(phi; mu_out, sigma_out^2),

the test the Neyman-Pearson lemma says is most powerful between the two
hypotheses. Offline LiRA uses only OUT models (which can be trained before the
query arrives) and a one-sided test: how far above the OUT distribution the
target's confidence lies, ``(phi - mu_out) / sigma_out``.

With ``queries > 1`` (for images, the identity and a horizontal flip), the score
is averaged over queries, each with its own Gaussians, as in the paper.

Choices that follow the authors' reference code: per-example centres use the
median (robust to the occasional diverged model) and ``fix_variance`` replaces
per-example variances by one pooled variance, which the paper recommends when
fewer than about 64 reference models are available. Pooling here is over
per-example deviations from each example's own centre.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_EPS = 1e-30
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


@dataclass(frozen=True)
class GaussianFit:
    """Per-example Gaussian parameters, each ``(n, queries)``; ``count_*`` is ``(n,)``."""

    mu_in: np.ndarray
    sigma_in: np.ndarray
    mu_out: np.ndarray
    sigma_out: np.ndarray
    count_in: np.ndarray
    count_out: np.ndarray


def _as3d(phi: np.ndarray) -> np.ndarray:
    a = np.asarray(phi, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError("reference phi must be (refs, n) or (refs, n, queries)")
    return a


def _masked_center_scale(
    values: np.ndarray, keep: np.ndarray, center: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and standard deviation over axis 0 of ``values`` where ``keep`` is true."""
    v = np.where(keep[:, :, None], values, np.nan)
    count = keep.sum(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid = np.nanmedian(v, axis=0) if center == "median" else np.nanmean(v, axis=0)
        sd = np.nanstd(v, axis=0)
    return mid, sd, count


def _pooled_std(values: np.ndarray, keep: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """One standard deviation per query, pooled over all examples' deviations."""
    dev = np.where(keep[:, :, None], values - mid[None], 0.0)
    dev = np.nan_to_num(dev)
    dof = max(int(keep.sum()), 1)
    return np.asarray(np.sqrt((dev**2).sum(axis=(0, 1)) / dof))


def fit_gaussians(
    ref_phi: np.ndarray,
    ref_masks: np.ndarray,
    fix_variance: bool = False,
    center: str = "median",
) -> GaussianFit:
    """Fit the IN and OUT Gaussians of every example from reference models.

    Args:
        ref_phi: ``(refs, n)`` or ``(refs, n, queries)`` logit-scaled confidences.
        ref_masks: ``(refs, n)`` booleans, true where the example trained the reference.
        fix_variance: use one pooled variance per query instead of per-example ones.
        center: ``"median"`` (reference code) or ``"mean"``.
    """
    if center not in ("median", "mean"):
        raise ValueError("center must be 'median' or 'mean'")
    phi = _as3d(ref_phi)
    masks = np.asarray(ref_masks, dtype=bool)
    if masks.shape != phi.shape[:2]:
        raise ValueError("ref_masks must be (refs, n)")
    mu_in, sd_in, c_in = _masked_center_scale(phi, masks, center)
    mu_out, sd_out, c_out = _masked_center_scale(phi, ~masks, center)
    pooled_in = _pooled_std(phi, masks, np.nan_to_num(mu_in))
    pooled_out = _pooled_std(phi, ~masks, np.nan_to_num(mu_out))
    if fix_variance:
        sd_in = np.broadcast_to(pooled_in, mu_in.shape).copy()
        sd_out = np.broadcast_to(pooled_out, mu_out.shape).copy()
    else:
        # Fewer than two models give no variance estimate: fall back to the pooled one.
        sd_in = np.where(c_in[:, None] >= 2, sd_in, pooled_in)
        sd_out = np.where(c_out[:, None] >= 2, sd_out, pooled_out)
    return GaussianFit(mu_in, sd_in, mu_out, sd_out, c_in, c_out)


def _logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    s = sigma + _EPS
    return np.asarray(-0.5 * ((x - mu) / s) ** 2 - np.log(s) - _LOG_SQRT_2PI)


def _target(phi: np.ndarray) -> np.ndarray:
    t = np.asarray(phi, dtype=np.float64)
    return t[:, None] if t.ndim == 1 else t


def lira_online(
    target_phi: np.ndarray,
    ref_phi: np.ndarray,
    ref_masks: np.ndarray,
    fix_variance: bool = False,
    center: str = "median",
) -> np.ndarray:
    """Online LiRA score per example (higher means member)."""
    fit = fit_gaussians(ref_phi, ref_masks, fix_variance, center)
    x = _target(target_phi)
    if x.shape != fit.mu_in.shape:
        raise ValueError("target_phi does not match the reference shape")
    # Without any IN (or OUT) model the test degenerates; use the other side only.
    mu_in = np.where(np.isnan(fit.mu_in), fit.mu_out, fit.mu_in)
    mu_out = np.where(np.isnan(fit.mu_out), fit.mu_in, fit.mu_out)
    score = _logpdf(x, mu_in, fit.sigma_in) - _logpdf(x, mu_out, fit.sigma_out)
    return np.asarray(score.mean(axis=1))


def lira_offline(
    target_phi: np.ndarray,
    ref_phi: np.ndarray,
    ref_masks: np.ndarray,
    fix_variance: bool = False,
    center: str = "median",
    one_sided: bool = True,
) -> np.ndarray:
    """Offline LiRA score per example from OUT reference models only.

    ``one_sided=True`` is the paper's test, ``(phi - mu_out) / sigma_out``. With
    ``one_sided=False`` the score is the negative OUT log density, as in the
    authors' released code, which also flags examples far *below* the OUT mean.
    """
    fit = fit_gaussians(ref_phi, ref_masks, fix_variance, center)
    x = _target(target_phi)
    if x.shape != fit.mu_out.shape:
        raise ValueError("target_phi does not match the reference shape")
    mu = np.nan_to_num(fit.mu_out)
    score = (x - mu) / (fit.sigma_out + _EPS) if one_sided else -_logpdf(x, mu, fit.sigma_out)
    return np.asarray(score.mean(axis=1))
