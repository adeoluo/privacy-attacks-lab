"""Per-example signals that membership inference attacks read off a model's logits.

All functions take logits of shape ``(..., classes)`` and integer labels broadcast
against the leading axes, and are numerically stable for large logits.

- ``loss``: cross-entropy ``-log p_y`` (Yeom et al. 2018).
- ``logit_scaled_confidence``: ``phi = log(p_y / (1 - p_y))``, computed as
  ``z_y - logsumexp_{j != y} z_j`` so it stays finite when ``p_y`` rounds to 1.
  Carlini et al. (2022, Section 4.2) show that ``phi`` is close to Gaussian
  across models, which is what makes the LiRA likelihood ratio work.
- ``top_k_probs``: the sorted largest softmax probabilities (Shokri et al. 2017
  feed the sorted probability vector to their attack model).
"""

from __future__ import annotations

import numpy as np


def _check(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim < 1 or z.shape[-1] < 2:
        raise ValueError("logits need a class axis with at least 2 classes")
    y = np.broadcast_to(y, z.shape[:-1])
    if y.size and (y.min() < 0 or y.max() >= z.shape[-1]):
        raise ValueError("label out of range")
    return z, y


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy of the true label."""
    z, y = _check(logits, labels)
    return -np.take_along_axis(log_softmax(z), y[..., None], axis=-1)[..., 0]


def logit_scaled_confidence(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``log p_y - log(1 - p_y)``, stable for confident predictions."""
    z, y = _check(logits, labels)
    zy = np.take_along_axis(z, y[..., None], axis=-1)[..., 0]
    others = z.copy()
    np.put_along_axis(others, y[..., None], -np.inf, axis=-1)
    m = others.max(axis=-1)
    lse = m + np.log(np.exp(others - m[..., None]).sum(axis=-1))
    return np.asarray(zy - lse)


def top_k_probs(logits: np.ndarray, k: int = 3) -> np.ndarray:
    """The ``k`` largest softmax probabilities in decreasing order."""
    p = softmax(logits)
    k = min(k, p.shape[-1])
    return -np.sort(-p, axis=-1)[..., :k]


def correct(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    z, y = _check(logits, labels)
    return np.asarray(z.argmax(axis=-1) == y)


def summarize(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> dict[str, np.ndarray]:
    """The compact per-example record stored for every trained model."""
    return {
        "phi": logit_scaled_confidence(logits, labels).astype(np.float32),
        "loss": loss(logits, labels).astype(np.float32),
        "topk": top_k_probs(logits, k).astype(np.float32),
        "correct": correct(logits, labels),
    }
