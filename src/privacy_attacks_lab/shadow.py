"""Train many models on random halves of a data pool and store their per-example signals.

This is the experiment harness behind every membership inference result: LiRA,
the calibrated attack and the shadow-model attack all need models whose training
sets are known. Following Carlini et al. (2022), model ``i`` trains on the pool
examples where ``mask[i]`` is true, and the masks are balanced so every example is
IN for exactly half of the models. Any model can then play the target while the
others serve as shadow or reference models.

Each model is saved as ``model_XXX.npz`` (signals for every pool example, see
:func:`privacy_attacks_lab.signals.summarize`) next to ``spec.json`` and
``masks.npy``, so runs can be resumed and extended model by model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from privacy_attacks_lab.data import Dataset, DatasetName, load


@dataclass
class ShadowSpec:
    """Everything that defines a family of models (stored as ``spec.json``)."""

    dataset: DatasetName = "cifar10"
    arch: str = "cnn"
    num_models: int = 64
    pool_size: int | None = None
    seed: int = 0
    queries: str = "flip"
    epochs: int = 15
    batch_size: int = 256
    lr: float = 0.05
    weight_decay: float = 5e-4
    augment: bool = False
    # DP-SGD (used when noise_multiplier is set): per-example clipping norm and noise.
    noise_multiplier: float | None = None
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShadowSpec:
        return cls(**raw)


def make_masks(num_models: int, n: int, seed: int, pkeep: float = 0.5) -> np.ndarray:
    """Boolean ``(num_models, n)`` membership masks.

    Each column has exactly ``round(pkeep * num_models)`` true entries (the LiRA
    construction: rank a uniform draw per example across models), so every example
    has the same number of IN and OUT models.
    """
    if num_models < 1 or n < 0:
        raise ValueError("need at least one model")
    if not 0.0 < pkeep < 1.0:
        raise ValueError("pkeep must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.uniform(size=(num_models, n)).argsort(axis=0).argsort(axis=0)
    return np.asarray(order < round(pkeep * num_models))


def split_pool(ds: Dataset, spec: ShadowSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(pool indices, held-out test indices)`` within ``ds``.

    CIFAR-10 uses its 50,000 training images as the pool and the 10,000 test images
    as the test set. Tabular data uses every record (or a seeded subset of
    ``pool_size``) as the pool; each model's OUT half doubles as its test set.
    """
    n = len(ds)
    if ds.name == "cifar10" and n == 60000:
        pool, test = np.arange(50000), np.arange(50000, 60000)
    else:
        pool, test = np.arange(n), np.arange(0)
    if spec.pool_size is not None and spec.pool_size < len(pool):
        rng = np.random.default_rng(spec.seed + 7919)
        pool = np.sort(rng.choice(pool, size=spec.pool_size, replace=False))
    return pool, test


class ShadowRun:
    """A directory of trained models sharing one :class:`ShadowSpec`."""

    def __init__(self, root: Path, spec: ShadowSpec | None = None) -> None:
        self.root = Path(root)
        spec_path = self.root / "spec.json"
        if spec is None:
            if not spec_path.exists():
                raise FileNotFoundError(f"{spec_path} does not exist")
            spec = ShadowSpec.from_dict(json.loads(spec_path.read_text()))
        elif spec_path.exists():
            stored = json.loads(spec_path.read_text())
            if stored != asdict(spec):
                raise ValueError(f"{self.root} holds a different spec; use a new directory")
        self.spec = spec

    def model_path(self, i: int) -> Path:
        return self.root / f"model_{i:03d}.npz"

    def completed(self) -> list[int]:
        return [i for i in range(self.spec.num_models) if self.model_path(i).exists()]

    def init(self, ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Write the spec, pool, labels and masks once; return ``(pool, test, masks)``."""
        self.root.mkdir(parents=True, exist_ok=True)
        pool, test = split_pool(ds, self.spec)
        masks_path = self.root / "masks.npy"
        if masks_path.exists():
            masks = np.load(masks_path)
        else:
            masks = make_masks(self.spec.num_models, len(pool), self.spec.seed)
            np.save(masks_path, masks)
            np.save(self.root / "pool.npy", pool)
            np.save(self.root / "labels.npy", ds.y[pool])
            (self.root / "spec.json").write_text(json.dumps(asdict(self.spec), indent=2))
        return pool, test, masks

    def train(
        self,
        models: range | list[int] | None = None,
        data_dir: Path | None = None,
        device: str = "auto",
        log: Callable[[str], None] = print,
        dataset: Dataset | None = None,
    ) -> list[dict[str, Any]]:
        """Train the missing models among ``models`` (default: all) and save their signals."""
        from privacy_attacks_lab import signals
        from privacy_attacks_lab.models import build, num_parameters
        from privacy_attacks_lab.training import (
            DeviceData,
            TrainConfig,
            accuracy,
            get_device,
            predict_logits,
            train,
        )

        spec = self.spec
        ds = dataset if dataset is not None else load(spec.dataset, data_dir)
        pool, test, masks = self.init(ds)
        dev = get_device(device)
        pool_data = DeviceData(ds.subset(pool), dev)
        test_data = DeviceData(ds.subset(test), dev) if len(test) else None
        labels = ds.y[pool]
        todo = list(range(spec.num_models)) if models is None else list(models)
        rows: list[dict[str, Any]] = []
        for i in todo:
            if self.model_path(i).exists():
                continue
            model = build(spec.arch, ds.input_shape, ds.num_classes)  # type: ignore[arg-type]
            cfg = TrainConfig(
                epochs=spec.epochs,
                batch_size=spec.batch_size,
                lr=spec.lr,
                weight_decay=spec.weight_decay,
                augment=spec.augment and ds.is_image,
                seed=spec.seed * 100_003 + i,
            )
            train_idx = np.flatnonzero(masks[i])
            row: dict[str, Any] = {"model": i, "params": num_parameters(model)}
            if spec.noise_multiplier is None:
                res = train(model, pool_data, train_idx, cfg)
                row.update(seconds=res.seconds, final_loss=res.final_loss)
            else:
                from privacy_attacks_lab.dpsgd import DPConfig, train_dp

                dp = DPConfig(
                    noise_multiplier=spec.noise_multiplier,
                    max_grad_norm=spec.max_grad_norm,
                    delta=spec.delta,
                )
                dres = train_dp(model, pool_data, train_idx, cfg, dp)
                row.update(seconds=dres.seconds, final_loss=dres.final_loss, epsilon=dres.epsilon)
            t0 = time.perf_counter()
            logits = predict_logits(model, pool_data, queries=spec.queries)  # type: ignore[arg-type]
            sig = signals.summarize(logits, labels[:, None])
            row["train_acc"] = accuracy(logits[masks[i]], labels[masks[i]])
            row["out_acc"] = accuracy(logits[~masks[i]], labels[~masks[i]])
            if test_data is not None:
                row["test_acc"] = accuracy(predict_logits(model, test_data), ds.y[test])
            row["inference_seconds"] = time.perf_counter() - t0
            np.savez(
                self.model_path(i),
                phi=sig["phi"],
                loss=sig["loss"],
                topk=sig["topk"].astype(np.float16),
                correct=sig["correct"],
                meta=json.dumps(row),
            )
            with (self.root / "log.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            shown = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
            log(shown)
            rows.append(row)
        return rows

    def load(self, models: list[int] | None = None) -> Signals:
        """Stack the stored signals of the completed models."""
        done = self.completed() if models is None else models
        if not done:
            raise FileNotFoundError(f"no trained models in {self.root}")
        masks = np.load(self.root / "masks.npy")[done]
        parts: dict[str, list[np.ndarray]] = {"phi": [], "loss": [], "topk": [], "correct": []}
        meta = []
        for i in done:
            with np.load(self.model_path(i)) as f:
                for key in parts:
                    parts[key].append(f[key])
                meta.append(json.loads(str(f["meta"])))
        return Signals(
            phi=np.stack(parts["phi"]),
            loss=np.stack(parts["loss"]),
            topk=np.stack(parts["topk"]).astype(np.float32),
            correct=np.stack(parts["correct"]),
            masks=masks,
            labels=np.load(self.root / "labels.npy"),
            meta=meta,
        )


@dataclass
class Signals:
    """Stacked per-model signals: arrays are ``(models, examples, queries[, k])``."""

    phi: np.ndarray
    loss: np.ndarray
    topk: np.ndarray
    correct: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    meta: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_models(self) -> int:
        return int(self.masks.shape[0])

    def select(self, models: np.ndarray | list[int]) -> Signals:
        idx = np.asarray(models)
        return Signals(
            self.phi[idx],
            self.loss[idx],
            self.topk[idx],
            self.correct[idx],
            self.masks[idx],
            self.labels,
            [self.meta[i] for i in idx] if self.meta else [],
        )
