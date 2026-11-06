"""DP-SGD (Abadi et al., CCS 2016), implemented directly on ``torch.func``.

One step with sampling rate ``q``, clipping norm ``C`` and noise multiplier ``sigma``:

1. Poisson sampling: every training example joins the batch independently with
   probability ``q`` (the sampling the RDP and PRV accountants analyse; fixed-size
   shuffled batches would need a different analysis).
2. Per-example gradients ``g_i`` with ``vmap(grad(loss))``, so no example's gradient
   is ever mixed with another's before clipping.
3. Clipping: ``g_i <- g_i * min(1, C / ||g_i||_2)``, the norm taken over all
   parameters jointly (flat clipping), which bounds each example's contribution.
4. Noise: ``g = (sum_i g_i + N(0, sigma^2 C^2 I)) / (q n)``, dividing by the
   *expected* batch size so the noise scale does not depend on the realised one.
5. Any first-order update (here SGD with momentum); post-processing keeps DP.

The privacy spent by ``T`` such steps is ``epsilon(q, sigma, T, delta)`` from
:mod:`privacy_attacks_lab.accountants`. Models must not use BatchNorm (it mixes
examples); :mod:`privacy_attacks_lab.models` uses GroupNorm.

Per-example gradients are computed in physical chunks sized from the parameter
count so peak memory stays bounded (about 256 MB of gradients at a time).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

from privacy_attacks_lab.accountants import Accountant, epsilon
from privacy_attacks_lab.training import DeviceData, TrainConfig, _sync, seed_everything

#: Floats of per-example gradient held at once (64M floats = 256 MB in float32).
GRAD_BUDGET = 64_000_000


@dataclass
class DPConfig:
    noise_multiplier: float
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    accountant: str = "rdp"
    engine: str = "ghost"

    def __post_init__(self) -> None:
        if self.noise_multiplier < 0:
            raise ValueError("noise multiplier must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("clipping norm must be positive")


Engine = Literal["vmap", "hooks", "ghost"]


def per_sample_grads(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, engine: Engine = "vmap"
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Gradient of each example's cross-entropy loss, ``{name: (batch, *param.shape)}``,
    and the per-example losses.

    ``vmap`` differentiates a single-example loss and vectorises it over the batch
    (works for any model). ``hooks`` runs one ordinary batched backward pass and
    rebuilds each example's weight gradient from the layer's input activations
    and output gradients (supports Linear, Conv2d and GroupNorm, the layers of
    :mod:`privacy_attacks_lab.models`).
    """
    if engine in ("hooks", "ghost"):
        return _per_sample_grads_hooks(model, x, y)
    if engine != "vmap":
        raise ValueError(f"unknown engine {engine!r}")
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(p: dict[str, torch.Tensor], xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        out = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    grads, losses = vmap(grad_and_value(loss_one), in_dims=(None, 0, 0))(params, x, y)
    return dict(grads), losses


_SUPPORTED = (nn.Linear, nn.Conv2d, nn.GroupNorm)


def _capture(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[list[tuple[str, nn.Module, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """One batched forward and backward pass recording, for every layer with
    parameters, its input activation and the gradient of the summed loss with
    respect to its output. Returns ``[(name, module, activation, out_grad)]`` and
    the per-example losses."""
    acts: dict[nn.Module, torch.Tensor] = {}
    outs: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def hook(m: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        acts[m] = inp[0].detach()
        if out.requires_grad:
            out.register_hook(lambda g: outs.__setitem__(m, g.detach()))

    for mod in model.modules():
        if isinstance(mod, _SUPPORTED):
            handles.append(mod.register_forward_hook(hook))
        elif any(True for _ in mod.parameters(recurse=False)):
            for h in handles:
                h.remove()
            raise ValueError(f"the hooks engines do not support {type(mod).__name__}")
    try:
        losses = F.cross_entropy(model(x), y, reduction="none")
        params = [p for p in model.parameters() if p.requires_grad]
        torch.autograd.grad(losses.sum(), params)
    finally:
        for h in handles:
            h.remove()
    layers = [
        (name, mod, acts[mod], outs[mod]) for name, mod in model.named_modules() if mod in acts
    ]
    return layers, losses.detach()


def _layer_grads(mod: nn.Module, a: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-example gradients of one layer's parameters.

    For a linear layer ``z = W a + b`` the loss of example ``i`` alone has
    ``dL_i/dW = g_i a_i^T`` where ``g_i = dL/dz_i`` of the summed loss (the other
    examples do not touch row ``i``). A convolution is a linear layer applied to
    every unfolded patch, so ``dL_i/dW = sum_l g_il patch_il^T``. GroupNorm's affine
    weights see the normalised input ``xhat``: ``dL_i/dgamma_c = sum g xhat``.
    """
    out: dict[str, torch.Tensor] = {}
    if isinstance(mod, nn.Linear):
        out["weight"] = torch.einsum("bo,bi->boi", g, a)
        if mod.bias is not None:
            out["bias"] = g
    elif isinstance(mod, nn.Conv2d):
        patches = F.unfold(a, mod.kernel_size, mod.dilation, mod.padding, mod.stride)  # type: ignore[arg-type]
        g_flat = g.flatten(2)
        gw = torch.einsum("bol,bkl->bok", g_flat, patches)
        out["weight"] = gw.view(a.shape[0], *mod.weight.shape)
        if mod.bias is not None:
            out["bias"] = g_flat.sum(dim=2)
    else:
        assert isinstance(mod, nn.GroupNorm)
        xhat = F.group_norm(a, mod.num_groups, eps=mod.eps)
        if mod.weight is not None:
            out["weight"] = (g * xhat).flatten(2).sum(dim=2)
        if mod.bias is not None:
            out["bias"] = g.flatten(2).sum(dim=2)
    return out


def _per_sample_grads_hooks(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    layers, losses = _capture(model, x, y)
    grads: dict[str, torch.Tensor] = {}
    for name, mod, a, g in layers:
        prefix = f"{name}." if name else ""
        for k, v in _layer_grads(mod, a, g).items():
            grads[prefix + k] = v
    names = [k for k, p in model.named_parameters() if p.requires_grad]
    return {k: grads[k] for k in names}, losses


def clipped_sum_ghost(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, max_norm: float
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Summed clipped gradients without materialising linear layers' per-example gradients.

    Ghost clipping (Goodfellow 2015; Li, Tramer, Liang and Hashimoto, ICLR 2022): for
    a linear layer on unbatched-in-time inputs, ``||g_i a_i^T||_F^2 = ||g_i||^2 ||a_i||^2``,
    so per-example norms cost O(b (in + out)) instead of O(b in out), and the clipped
    sum ``sum_i c_i g_i a_i^T`` is the single matrix product ``(c * G)^T A``.
    Convolutions and GroupNorm have few parameters here and are materialised.

    Returns ``(summed clipped gradients, per-example norms, per-example losses)``.
    """
    layers, losses = _capture(model, x, y)
    b = x.shape[0]
    sq = torch.zeros(b, device=x.device, dtype=torch.float32)
    dense: dict[str, torch.Tensor] = {}
    for name, mod, a, g in layers:
        prefix = f"{name}." if name else ""
        if isinstance(mod, nn.Linear) and a.dim() == 2:
            g2 = (g * g).sum(dim=1)
            sq += g2 * (a * a).sum(dim=1)
            if mod.bias is not None:
                sq += g2
        else:
            for k, v in _layer_grads(mod, a, g).items():
                dense[prefix + k] = v
                sq += v.reshape(b, -1).pow(2).sum(dim=1)
    norms = sq.sqrt()
    factor = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    summed: dict[str, torch.Tensor] = {}
    for name, mod, a, g in layers:
        prefix = f"{name}." if name else ""
        if isinstance(mod, nn.Linear) and a.dim() == 2:
            cg = g * factor[:, None]
            summed[prefix + "weight"] = cg.T @ a
            if mod.bias is not None:
                summed[prefix + "bias"] = cg.sum(dim=0)
    for k, v in dense.items():
        summed[k] = torch.einsum("b,b...->...", factor, v)
    names = [k for k, p in model.named_parameters() if p.requires_grad]
    return {k: summed[k] for k in names}, norms, losses


def clip_and_sum(
    grads: dict[str, torch.Tensor], max_norm: float
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip each example's gradient to ``max_norm`` (flat L2) and sum over the batch.

    Returns the summed clipped gradients and the per-example norms before clipping.
    """
    flat = torch.stack([g.reshape(g.shape[0], -1).norm(dim=1) for g in grads.values()], dim=1)
    norms = flat.norm(dim=1)
    factor = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    summed = {k: torch.einsum("b,b...->...", factor, g) for k, g in grads.items()}
    return summed, norms


def physical_batch(model: nn.Module, budget: int = GRAD_BUDGET) -> int:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return max(8, budget // max(params, 1))


def noisy_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dp: DPConfig,
    expected_batch: float,
    generator: torch.Generator | None = None,
    chunk: int | None = None,
    engine: Engine = "ghost",
) -> tuple[dict[str, torch.Tensor], float]:
    """The privatised gradient of one logical batch (steps 2 to 4 above) and the summed
    loss of its examples."""
    chunk = chunk or physical_batch(model)
    total: dict[str, torch.Tensor] = {
        k: torch.zeros_like(p) for k, p in model.named_parameters() if p.requires_grad
    }
    loss_sum = torch.zeros((), device=x.device)
    for s in range(0, x.shape[0], chunk):
        xc, yc = x[s : s + chunk], y[s : s + chunk]
        if engine == "ghost":
            summed, _, losses = clipped_sum_ghost(model, xc, yc, dp.max_grad_norm)
        else:
            grads, losses = per_sample_grads(model, xc, yc, engine)
            summed, _ = clip_and_sum(grads, dp.max_grad_norm)
        loss_sum += losses.detach().sum()
        for k in total:
            total[k] += summed[k]
    std = dp.noise_multiplier * dp.max_grad_norm
    out = {}
    for k, g in total.items():
        if std > 0:
            noise = torch.randn(g.shape, generator=generator, device="cpu").to(g.device)
            g = g + std * noise
        out[k] = g / expected_batch
    return out, float(loss_sum)


def poisson_batches(n: int, q: float, steps: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Index sets of ``steps`` Poisson-sampled batches (each example kept with prob. ``q``)."""
    return [torch.nonzero(torch.rand(n, generator=generator) < q).flatten() for _ in range(steps)]


@dataclass
class DPTrainResult:
    seconds: float
    steps: int
    epsilon: float
    epsilon_prv: float
    final_loss: float
    sample_rate: float
    history: list[float] = field(default_factory=list)


def train_dp(
    model: nn.Module,
    data: DeviceData,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    dp: DPConfig,
) -> DPTrainResult:
    """Train with DP-SGD for ``cfg.epochs`` epochs of Poisson batches of expected size
    ``cfg.batch_size``; returns the epsilon spent at ``dp.delta`` (RDP and PRV)."""
    dev = data.device
    gen = seed_everything(cfg.seed)
    model.to(dev).train()
    n = len(train_idx)
    q = min(1.0, cfg.batch_size / n)
    steps_per_epoch = max(1, round(1.0 / q))
    total_steps = cfg.epochs * steps_per_epoch
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    idx_all = torch.from_numpy(np.asarray(train_idx, dtype=np.int64)).to(dev)
    acc = Accountant()
    chunk = physical_batch(model)
    history: list[float] = []
    start = time.perf_counter()
    for _epoch in range(cfg.epochs):
        running, seen = 0.0, 0
        for local in poisson_batches(n, q, steps_per_epoch, gen):
            acc.step(dp.noise_multiplier, q)
            # An empty Poisson batch still spends privacy: its update is pure noise.
            xb, yb = data.batch(idx_all[local.to(dev)], augment=cfg.augment)
            grads, loss_sum = noisy_gradient(
                model,
                xb,
                yb,
                dp,
                q * n,
                gen,
                chunk,
                dp.engine,  # type: ignore[arg-type]
            )
            opt.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.grad = grads[name]
            opt.step()
            sched.step()
            running += loss_sum
            seen += int(local.numel())
        history.append(running / max(seen, 1))
    _sync(dev)
    seconds = time.perf_counter() - start
    eps_rdp = acc.epsilon(dp.delta) if dp.noise_multiplier > 0 else math.inf
    eps_prv = (
        epsilon(q, dp.noise_multiplier, total_steps, dp.delta, accountant="prv")
        if dp.noise_multiplier > 0
        else math.inf
    )
    return DPTrainResult(
        seconds, total_steps, eps_rdp, eps_prv, history[-1] if history else math.nan, q, history
    )
