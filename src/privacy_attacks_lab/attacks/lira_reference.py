"""LiRA scoring exactly as in the authors' released code, kept as a parity reference.

A loop-based transcription of ``generate_ours`` and ``generate_ours_offline`` from
``research/mi_lira_2021/plot.py`` in tensorflow/privacy (Apache-2.0), the code
behind Carlini et al. (2022). Differences from :mod:`privacy_attacks_lab.attacks.lira`:

- the reference truncates every example's IN and OUT sets to the smallest count
  over all examples (identical results when masks are balanced, as ours are);
- with ``fix_variance`` it uses the global standard deviation of all IN scores
  (around the global mean) for *both* Gaussians, whereas
  :func:`~privacy_attacks_lab.attacks.lira.fit_gaussians` pools each side's
  deviations from the per-example medians;
- offline it scores the negative OUT log density (two-sided), which
  :func:`~privacy_attacks_lab.attacks.lira.lira_offline` exposes as
  ``one_sided=False``; the paper's text describes the one-sided test.

Scores are returned with "higher means member", the sign the authors' ``sweep``
applies before computing the ROC curve.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _split(keep: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dat_in = [scores[keep[:, j], j, :] for j in range(scores.shape[1])]
    dat_out = [scores[~keep[:, j], j, :] for j in range(scores.shape[1])]
    in_size = min(len(x) for x in dat_in)
    out_size = min(len(x) for x in dat_out)
    return (
        np.array([x[:in_size] for x in dat_in]),
        np.array([x[:out_size] for x in dat_out]),
    )


def reference_online(
    keep: np.ndarray, scores: np.ndarray, check_scores: np.ndarray, fix_variance: bool = False
) -> np.ndarray:
    """``keep`` ``(refs, n)``, ``scores`` ``(refs, n, q)``, ``check_scores`` ``(n, q)``."""
    dat_in, dat_out = _split(keep, scores)
    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)
    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)
    pr_in = -stats.norm.logpdf(check_scores, mean_in, std_in + 1e-30)
    pr_out = -stats.norm.logpdf(check_scores, mean_out, std_out + 1e-30)
    return np.asarray(-(pr_in - pr_out).mean(1))


def reference_offline(
    keep: np.ndarray, scores: np.ndarray, check_scores: np.ndarray, fix_variance: bool = False
) -> np.ndarray:
    """Offline score of the released code (negative OUT log density)."""
    _, dat_out = _split(keep, scores)
    mean_out = np.median(dat_out, 1)
    std_out = np.std(dat_out) if fix_variance else np.std(dat_out, 1)
    score = stats.norm.logpdf(check_scores, mean_out, std_out + 1e-30)
    return np.asarray(-score.mean(1))
