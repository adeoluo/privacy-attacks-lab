"""The shadow-model attack (Shokri, Stronati, Song and Shmatikov, S&P 2017).

The attacker trains shadow models that imitate the target on data whose
membership it knows, then trains an attack classifier that maps a model's output
on an example (and the example's label) to IN or OUT. Shokri et al. train one
attack model per class; Salem et al. (NDSS 2019) show the three largest sorted
probabilities carry nearly all of the signal, which keeps the attack model small.

Features per query: the log of the three largest softmax probabilities and the
logit-scaled confidence of the true label (the information in "prediction vector
plus label"). The attack model is L2-regularised logistic regression fitted by
L-BFGS, one per class when the class has enough rows, else one global model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize


def shadow_features(topk: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """``(..., n, queries, k)`` probabilities and ``(..., n, queries)`` phi to ``(..., n, f)``."""
    p = np.asarray(topk, dtype=np.float64)
    c = np.asarray(phi, dtype=np.float64)
    if p.shape[:-1] != c.shape:
        raise ValueError("topk and phi disagree in shape")
    logp = np.log(np.clip(p, 1e-12, 1.0))
    feats = np.concatenate([logp, c[..., None]], axis=-1)
    return feats.reshape(*feats.shape[:-2], -1)


@dataclass
class LogisticRegression:
    """Binary logistic regression with an L2 penalty on the weights (not the bias)."""

    l2: float = 1e-3
    max_iter: int = 200
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticRegression:
        n, d = x.shape
        yy = y.astype(np.float64)

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            z = x @ w[:d] + w[d]
            # log(1 + e^z) - y z, stable for large |z|
            loss = np.logaddexp(0.0, z) - yy * z
            p = 0.5 * (1.0 + np.tanh(0.5 * z))
            g = p - yy
            grad = np.concatenate([x.T @ g / n + self.l2 * w[:d], [g.mean()]])
            return float(loss.mean() + 0.5 * self.l2 * w[:d] @ w[:d]), grad

        res = optimize.minimize(
            objective,
            np.zeros(d + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coef, self.bias = res.x[:d], float(res.x[d])
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x @ self.coef + self.bias)


@dataclass
class ShadowAttack:
    """Per-class attack models trained on shadow models' outputs."""

    per_class: bool = True
    l2: float = 1e-3
    min_rows_per_class: int = 200
    max_rows: int = 400_000
    seed: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    models: dict[int, LogisticRegression] = field(default_factory=dict)
    fallback: LogisticRegression | None = None

    def fit(self, feats: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> ShadowAttack:
        """``feats`` ``(shadows, n, f)``, ``masks`` ``(shadows, n)``, ``labels`` ``(n,)``."""
        f = np.asarray(feats, dtype=np.float64)
        m = np.asarray(masks, dtype=bool)
        if f.ndim != 3 or m.shape != f.shape[:2] or len(labels) != f.shape[1]:
            raise ValueError("feats must be (shadows, n, f) with matching masks and labels")
        x = f.reshape(-1, f.shape[2])
        y = m.reshape(-1)
        cls = np.broadcast_to(np.asarray(labels), m.shape).reshape(-1)
        if len(y) > self.max_rows:
            keep = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            x, y, cls = x[keep], y[keep], cls[keep]
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1e-8
        z = (x - self.mean) / self.scale
        self.fallback = LogisticRegression(self.l2).fit(z, y)
        self.models = {}
        if self.per_class:
            for c in np.unique(cls):
                rows = cls == c
                if rows.sum() >= self.min_rows_per_class and 0 < y[rows].mean() < 1:
                    self.models[int(c)] = LogisticRegression(self.l2).fit(z[rows], y[rows])
        return self

    def score(self, feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Attack logit for each example of the target (higher means member)."""
        if self.fallback is None:
            raise RuntimeError("fit the attack first")
        z = (np.asarray(feats, dtype=np.float64) - self.mean) / self.scale
        out = self.fallback.decision_function(z)
        lab = np.asarray(labels)
        for c, model in self.models.items():
            rows = lab == c
            if rows.any():
                out[rows] = model.decision_function(z[rows])
        return out
