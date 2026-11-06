"""Difficulty-calibrated loss attack (Watson, Guo, Cormode and Sablayrolles, ICLR 2022).

Some examples are hard (high loss) whether or not they were trained on, others are
easy. Subtracting each example's typical loss under reference models that did
*not* train on it removes that per-example difficulty:

    score(x) = -(loss_target(x) - mean_{r : x not in D_r} loss_r(x))

This is the offline variant (reference models never contain the target example),
so the reference models can be trained once, before the attack queries arrive.
With ``online=True`` all reference models are averaged regardless of membership.
"""

from __future__ import annotations

import numpy as np


def calibrated_attack(
    target_loss: np.ndarray,
    ref_loss: np.ndarray,
    ref_masks: np.ndarray,
    online: bool = False,
) -> np.ndarray:
    """Per-example calibrated score.

    Args:
        target_loss: ``(n,)`` or ``(n, queries)`` losses of the target model.
        ref_loss: ``(refs, n)`` or ``(refs, n, queries)`` losses of reference models.
        ref_masks: ``(refs, n)`` booleans, true where the example trained the reference.
        online: average over all references instead of only the OUT ones.
    """
    t = np.asarray(target_loss, dtype=np.float64)
    r = np.asarray(ref_loss, dtype=np.float64)
    m = np.asarray(ref_masks, dtype=bool)
    if t.ndim == 2:
        t = t.mean(axis=1)
    if r.ndim == 3:
        r = r.mean(axis=2)
    if r.ndim != 2 or r.shape != m.shape or r.shape[1] != t.shape[0]:
        raise ValueError("shapes of target_loss, ref_loss and ref_masks disagree")
    weights = np.ones_like(m, dtype=np.float64) if online else (~m).astype(np.float64)
    count = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        baseline = (weights * r).sum(axis=0) / count
    # Examples with no OUT reference fall back to the mean over all references.
    baseline = np.where(count > 0, baseline, r.mean(axis=0))
    return np.asarray(-(t - baseline))
