"""ROC metrics for membership inference: AUC, balanced accuracy and TPR at low FPR.

Carlini et al. (2022) argue that average-case metrics (AUC, accuracy) hide what
matters for privacy, whether an attack can confidently identify *some* members,
so every result here reports the true-positive rate at 0.1% and 1% false-positive
rate next to AUC, and plots ROC curves on log-log axes.

Higher scores mean "member". The Rust core computes curves and bootstrap
replicates; the numpy functions below are the reference the parity tests use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from privacy_attacks_lab._backend import Backend, resolve, rust

#: False-positive rates reported throughout (0.1%, 1% and 10%).
DEFAULT_FPRS: tuple[float, ...] = (0.001, 0.01, 0.1)


def _prepare(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.ascontiguousarray(np.asarray(scores, dtype=np.float64).ravel())
    lab = np.asarray(labels).ravel()
    if s.shape != lab.shape:
        raise ValueError("scores and labels must have the same length")
    if lab.dtype != bool and not np.isin(lab, (0, 1)).all():
        raise ValueError("labels must be 0/1 or boolean")
    return s, np.ascontiguousarray(lab.astype(np.uint8))


def roc_curve(
    scores: np.ndarray, labels: np.ndarray, backend: Backend = "auto"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, tpr, thresholds)`` over distinct thresholds, starting at ``(0, 0, +inf)``."""
    s, lab = _prepare(scores, labels)
    if resolve(backend) == "rust":
        f, t, thr = rust().roc_curve(s.tobytes(), lab.tobytes())
        return np.asarray(f), np.asarray(t), np.asarray(thr)
    return _roc_curve_numpy(s, lab.astype(bool))


def _roc_curve_numpy(
    s: np.ndarray, lab: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="stable")
    # Put NaN scores last, as the Rust core does.
    order = np.concatenate([order[~np.isnan(s[order])], order[np.isnan(s[order])]])
    s_sorted = s[order]
    w = np.ones_like(s) if weights is None else weights
    w_sorted = w[order]
    pos_w = np.where(lab[order], w_sorted, 0.0)
    neg_w = w_sorted - pos_w
    tp, fp = np.cumsum(pos_w), np.cumsum(neg_w)
    nan = np.isnan(s_sorted)
    same = (s_sorted[1:] == s_sorted[:-1]) | (nan[1:] & nan[:-1])
    last = np.r_[~same, True] if len(s) else np.zeros(0, dtype=bool)
    p, n = pos_w.sum(), neg_w.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = np.r_[0.0, tp[last] / p] if p > 0 else np.r_[0.0, np.full(last.sum(), np.nan)]
        fpr = np.r_[0.0, fp[last] / n] if n > 0 else np.r_[0.0, np.full(last.sum(), np.nan)]
    return fpr, tpr, np.r_[np.inf, s_sorted[last]]


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the curve by the trapezoid rule."""
    f, t = np.asarray(fpr), np.asarray(tpr)
    return float(np.sum(np.diff(f) * (t[1:] + t[:-1]) * 0.5))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """Largest TPR over curve points with FPR at most ``target``."""
    f, t = np.asarray(fpr), np.asarray(tpr)
    ok = f <= target
    return float(t[ok].max()) if ok.any() else 0.0


@dataclass(frozen=True)
class RocSummary:
    """AUC, best balanced accuracy and TPR at each FPR in ``fprs``."""

    auc: float
    accuracy: float
    fprs: tuple[float, ...]
    tprs: tuple[float, ...]

    def tpr(self, fpr: float) -> float:
        return self.tprs[self.fprs.index(fpr)]

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"auc": self.auc, "accuracy": self.accuracy}
        for f, t in zip(self.fprs, self.tprs, strict=True):
            out[f"tpr@{f:g}"] = t
        return out


def _summary_from_curve(fpr: np.ndarray, tpr: np.ndarray, fprs: Sequence[float]) -> RocSummary:
    acc = float(np.max(1.0 - (fpr + (1.0 - tpr)) / 2.0))
    return RocSummary(auc(fpr, tpr), acc, tuple(fprs), tuple(tpr_at_fpr(fpr, tpr, f) for f in fprs))


def summarize(
    scores: np.ndarray,
    labels: np.ndarray,
    fprs: Sequence[float] = DEFAULT_FPRS,
    backend: Backend = "auto",
) -> RocSummary:
    """Summary metrics of member scores ``scores`` with membership ``labels``."""
    s, lab = _prepare(scores, labels)
    if resolve(backend) == "rust":
        a, acc, tprs = rust().roc_summary(s.tobytes(), lab.tobytes(), list(fprs))
        return RocSummary(float(a), float(acc), tuple(fprs), tuple(float(x) for x in tprs))
    fpr, tpr, _ = _roc_curve_numpy(s, lab.astype(bool))
    return _summary_from_curve(fpr, tpr, fprs)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    fprs: Sequence[float] = DEFAULT_FPRS,
    reps: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    backend: Backend = "auto",
) -> dict[str, Interval]:
    """Percentile bootstrap intervals of every summary metric (examples resampled).

    Returns ``{"auc": ..., "accuracy": ..., "tpr@0.001": ...}`` with the full-sample
    estimate and the ``level`` interval.
    """
    if reps < 1:
        raise ValueError("reps must be positive")
    s, lab = _prepare(scores, labels)
    point = summarize(s, lab, fprs, backend=backend).to_dict()
    if resolve(backend) == "rust":
        raw = rust().roc_bootstrap(s.tobytes(), lab.tobytes(), list(fprs), reps, seed)
        table = np.array([[a, acc, *t] for a, acc, t in raw], dtype=np.float64)
    else:
        table = _bootstrap_numpy(s, lab.astype(bool), fprs, reps, seed)
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(table, alpha, axis=0)
    hi = np.quantile(table, 1.0 - alpha, axis=0)
    return {key: Interval(point[key], float(lo[j]), float(hi[j])) for j, key in enumerate(point)}


def _bootstrap_numpy(
    s: np.ndarray, lab: np.ndarray, fprs: Sequence[float], reps: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(s)
    rows = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        fpr, tpr, _ = _roc_curve_numpy(s, lab, w)
        r = _summary_from_curve(fpr, tpr, fprs)
        rows.append([r.auc, r.accuracy, *r.tprs])
    return np.asarray(rows)


def log_roc_points(
    fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The curve sampled on a log-spaced FPR grid (for plots and averaging curves)."""
    g = np.logspace(-5, 0, 200) if grid is None else np.asarray(grid)
    return g, np.array([tpr_at_fpr(fpr, tpr, x) for x in g])
