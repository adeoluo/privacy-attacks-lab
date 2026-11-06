"""The loss-threshold attack (Yeom, Giacomelli, Fredrikson and Jha, CSF 2018).

A model fits its training examples better than unseen ones, so the negative loss
is a membership score. The attack needs no shadow models; thresholding it at the
average training loss is the "global threshold" baseline that LiRA improves on
by calibrating per example.
"""

from __future__ import annotations

import numpy as np


def loss_attack(target_loss: np.ndarray) -> np.ndarray:
    """Score ``-loss`` per example; ``target_loss`` is ``(n,)`` or ``(n, queries)``."""
    loss = np.asarray(target_loss, dtype=np.float64)
    if loss.ndim == 2:
        loss = loss.mean(axis=1)
    if loss.ndim != 1:
        raise ValueError("target_loss must be (n,) or (n, queries)")
    return -loss
