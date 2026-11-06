"""Training and inference loops for target, shadow and reference models.

The whole training set lives on the accelerator as uint8 and is converted to float
per batch, so a shadow model's epoch is a sequence of device-side gathers with no
host transfers. Image augmentation (random crop with 4 pixels of reflect padding
and horizontal flip, as in the LiRA paper) runs batched on the device too.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from privacy_attacks_lab.data import Dataset

#: CIFAR-10 channel statistics (on the 50,000 training images).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

Queries = Literal["none", "flip"]


def get_device(prefer: str = "auto") -> torch.device:
    """``mps`` when available (Apple GPU), else ``cuda``, else ``cpu``; or the named device."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the test machines
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    return torch.Generator().manual_seed(seed)


@dataclass
class TrainConfig:
    """Optimisation settings (SGD with Nesterov momentum and a cosine schedule)."""

    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    augment: bool = True
    seed: int = 0
    device: str = "auto"
    label_smoothing: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeviceData:
    """A dataset resident on ``device`` as uint8 with on-device normalisation."""

    def __init__(self, ds: Dataset, device: torch.device) -> None:
        self.device = device
        self.is_image = ds.is_image
        x = torch.from_numpy(np.ascontiguousarray(ds.x))
        if self.is_image:
            x = x.permute(0, 3, 1, 2).contiguous()
            channels = x.shape[1]
            mean = CIFAR_MEAN if channels == 3 else (0.5,) * channels
            std = CIFAR_STD if channels == 3 else (0.25,) * channels
            self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
            self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.x = x.to(device)
        self.y = torch.from_numpy(ds.y).to(device)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def batch(
        self, idx: torch.Tensor, augment: bool = False, flip: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xb = self.x[idx]
        if not self.is_image:
            return xb.float(), self.y[idx]
        xf = (xb.float() / 255.0 - self.mean) / self.std
        if augment:
            xf = random_crop_flip(xf)
        elif flip:
            xf = torch.flip(xf, dims=(3,))
        return xf, self.y[idx]


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-example random translation by up to ``pad`` pixels (reflect padding) and h-flip."""
    b, c, h, w = x.shape
    dev = x.device
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (dy[:, None] + torch.arange(h, device=dev)[None, :])[:, None, :, None]
    cols = (dx[:, None] + torch.arange(w, device=dev)[None, :])[:, None, None, :]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], torch.flip(out, dims=(3,)), out)


@dataclass
class TrainResult:
    seconds: float
    epochs: int
    final_loss: float
    history: list[float]


def train(
    model: nn.Module,
    data: DeviceData,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Train ``model`` in place on ``data[train_idx]`` with SGD and a cosine schedule."""
    dev = data.device
    gen = seed_everything(cfg.seed)
    model.to(dev).train()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.momentum > 0,
    )
    n = len(train_idx)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        total_steps=cfg.epochs * steps_per_epoch,
        pct_start=0.15,
        anneal_strategy="cos",
    )
    idx_all = torch.from_numpy(np.asarray(train_idx, dtype=np.int64)).to(dev)
    history: list[float] = []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        perm = idx_all[torch.randperm(n, generator=gen).to(dev)]
        total = torch.zeros((), device=dev)
        for s in range(steps_per_epoch):
            bidx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size]
            xb, yb = data.batch(bidx, augment=cfg.augment)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach() * len(bidx)
        history.append(float(total) / n)
        if on_epoch is not None:
            on_epoch(epoch, history[-1])
    _sync(dev)
    return TrainResult(time.perf_counter() - start, cfg.epochs, history[-1], history)


def _sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    data: DeviceData,
    idx: np.ndarray | None = None,
    queries: Queries = "flip",
    batch_size: int = 2048,
) -> np.ndarray:
    """Logits of shape ``(n, q, classes)``; ``q`` is 2 (identity and h-flip) for images."""
    model.eval()
    n = len(data) if idx is None else len(idx)
    ids = torch.arange(len(data)) if idx is None else torch.from_numpy(np.asarray(idx))
    ids = ids.to(data.device)
    views = [False, True] if (queries == "flip" and data.is_image) else [False]
    out: list[np.ndarray] = []
    for flip in views:
        parts = []
        for s in range(0, n, batch_size):
            xb, _ = data.batch(ids[s : s + batch_size], flip=flip)
            parts.append(model(xb).float().cpu().numpy())
        out.append(np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32))
    return np.stack(out, axis=1).astype(np.float32)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy of ``(n, classes)`` or ``(n, q, classes)`` logits (first query)."""
    first = logits[:, 0] if logits.ndim == 3 else logits
    if len(labels) == 0:
        return math.nan
    return float((first.argmax(-1) == labels).mean())
