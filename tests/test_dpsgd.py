"""DP-SGD: per-example gradients, clipping, noise and the training loop."""

from __future__ import annotations

import math

import numpy as np
import pytest

from .conftest import needs_opacus, needs_torch

pytestmark = needs_torch

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from privacy_attacks_lab import dpsgd  # noqa: E402
from privacy_attacks_lab.accountants import epsilon  # noqa: E402
from privacy_attacks_lab.data import synthetic  # noqa: E402
from privacy_attacks_lab.models import build  # noqa: E402
from privacy_attacks_lab.training import DeviceData, TrainConfig  # noqa: E402

CASES = [("cnn-small", (3, 8, 8), 4), ("mlp", (20,), 5), ("mlp-small", (12,), 3)]


def _model_batch(arch: str, shape: tuple[int, ...], classes: int, b: int = 6, seed: int = 0):  # type: ignore[no-untyped-def]
    torch.manual_seed(seed)
    model = build(arch, shape, classes)  # type: ignore[arg-type]
    x = torch.randn(b, *shape)
    y = torch.randint(0, classes, (b,))
    return model, x, y


def _loop_grads(model, x, y):  # type: ignore[no-untyped-def]
    """Reference: one backward pass per example."""
    out = {n: [] for n, _ in model.named_parameters()}
    for i in range(x.shape[0]):
        model.zero_grad()
        F.cross_entropy(model(x[i : i + 1]), y[i : i + 1]).backward()
        for n, p in model.named_parameters():
            out[n].append(p.grad.detach().clone())
    return {n: torch.stack(v) for n, v in out.items()}


@pytest.mark.parametrize(("arch", "shape", "classes"), CASES)
@pytest.mark.parametrize("engine", ["vmap", "hooks"])
def test_per_sample_grads_match_loop(arch, shape, classes, engine) -> None:  # type: ignore[no-untyped-def]
    model, x, y = _model_batch(arch, shape, classes)
    ref = _loop_grads(model, x, y)
    got, losses = dpsgd.per_sample_grads(model, x, y, engine)
    assert set(got) == set(ref)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(losses, F.cross_entropy(model(x), y, reduction="none"))


@pytest.mark.parametrize(("arch", "shape", "classes"), CASES)
@pytest.mark.parametrize("max_norm", [1e-3, 0.5, 100.0])
def test_ghost_clipping_matches_materialised(arch, shape, classes, max_norm) -> None:  # type: ignore[no-untyped-def]
    model, x, y = _model_batch(arch, shape, classes, b=9)
    grads, _ = dpsgd.per_sample_grads(model, x, y, "vmap")
    ref, ref_norms = dpsgd.clip_and_sum(grads, max_norm)
    got, norms, _ = dpsgd.clipped_sum_ghost(model, x, y, max_norm)
    torch.testing.assert_close(norms, ref_norms, rtol=1e-4, atol=1e-6)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], rtol=1e-4, atol=1e-6)


def test_clip_and_sum_bounds_every_example() -> None:
    model, x, y = _model_batch("mlp", (20,), 5, b=16)
    grads, _ = dpsgd.per_sample_grads(model, x, y)
    single = {k: v[:1] for k, v in grads.items()}
    summed, norms = dpsgd.clip_and_sum(single, 0.01)
    flat = torch.cat([v.flatten() for v in summed.values()])
    assert float(flat.norm()) == pytest.approx(0.01, rel=1e-3)
    assert float(norms[0]) > 0.01
    # A large clipping norm leaves the plain sum.
    summed, _ = dpsgd.clip_and_sum(grads, 1e9)
    for k, v in grads.items():
        torch.testing.assert_close(summed[k], v.sum(0))


def test_unknown_engine_and_layer() -> None:
    model, x, y = _model_batch("mlp", (20,), 5)
    with pytest.raises(ValueError, match="engine"):
        dpsgd.per_sample_grads(model, x, y, "loop")  # type: ignore[arg-type]
    bn = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
    with pytest.raises(ValueError, match="BatchNorm1d"):
        dpsgd.per_sample_grads(bn, torch.randn(3, 4), torch.tensor([0, 1, 0]), "hooks")


def test_dpconfig_validation() -> None:
    with pytest.raises(ValueError, match="noise"):
        dpsgd.DPConfig(noise_multiplier=-1.0)
    with pytest.raises(ValueError, match="clipping"):
        dpsgd.DPConfig(noise_multiplier=1.0, max_grad_norm=0.0)


def test_physical_batch_respects_budget() -> None:
    model = build("mlp", (600,), 100)
    params = sum(p.numel() for p in model.parameters())
    assert dpsgd.physical_batch(model, budget=params * 100) == 100
    assert dpsgd.physical_batch(model, budget=1) == 8


@pytest.mark.parametrize("engine", ["ghost", "vmap", "hooks"])
def test_noiseless_gradient_is_clipped_mean(engine: str) -> None:
    model, x, y = _model_batch("mlp", (20,), 5, b=20)
    dp = dpsgd.DPConfig(noise_multiplier=0.0, max_grad_norm=0.1)
    got, loss_sum = dpsgd.noisy_gradient(
        model, x, y, dp, expected_batch=25.0, chunk=7, engine=engine
    )  # type: ignore[arg-type]
    grads, losses = dpsgd.per_sample_grads(model, x, y)
    ref, _ = dpsgd.clip_and_sum(grads, 0.1)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k] / 25.0, rtol=1e-4, atol=1e-7)
    assert loss_sum == pytest.approx(float(losses.sum()), rel=1e-5)


def test_noise_has_the_calibrated_scale() -> None:
    model, x, y = _model_batch("mlp-small", (12,), 3, b=4)
    dp = dpsgd.DPConfig(noise_multiplier=2.0, max_grad_norm=0.5)
    clean, _ = dpsgd.noisy_gradient(model, x, y, dpsgd.DPConfig(0.0, 0.5), 10.0)
    noisy, _ = dpsgd.noisy_gradient(model, x, y, dp, 10.0, torch.Generator().manual_seed(0))
    resid = torch.cat([(noisy[k] - clean[k]).flatten() for k in clean])
    # std of each coordinate = sigma * C / expected batch = 0.1
    assert float(resid.std()) == pytest.approx(0.1, rel=0.05)
    assert abs(float(resid.mean())) < 0.01


def test_poisson_batches_rate() -> None:
    gen = torch.Generator().manual_seed(0)
    batches = dpsgd.poisson_batches(10_000, 0.02, 50, gen)
    sizes = np.array([len(b) for b in batches])
    assert sizes.mean() == pytest.approx(200, rel=0.05)
    assert sizes.std() > 0
    assert all(int(b.max()) < 10_000 for b in batches)


def test_train_dp_reports_accountant_epsilon() -> None:
    ds = synthetic(n=600, features=16, num_classes=3)
    data = DeviceData(ds, torch.device("cpu"))
    model = build("mlp-small", ds.input_shape, ds.num_classes)
    cfg = TrainConfig(epochs=3, batch_size=60, lr=0.5, augment=False)
    dp = dpsgd.DPConfig(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    res = dpsgd.train_dp(model, data, np.arange(600), cfg, dp)
    assert res.steps == 30
    assert res.sample_rate == pytest.approx(0.1)
    assert res.epsilon == pytest.approx(epsilon(0.1, 1.0, 30, 1e-5), rel=1e-9)
    assert res.epsilon_prv <= res.epsilon + 0.05
    assert len(res.history) == 3
    assert res.history[-1] < res.history[0]


def test_train_dp_without_noise_has_infinite_epsilon() -> None:
    ds = synthetic(n=200, features=8, num_classes=2)
    data = DeviceData(ds, torch.device("cpu"))
    model = build("mlp-small", ds.input_shape, ds.num_classes)
    res = dpsgd.train_dp(
        model, data, np.arange(200), TrainConfig(epochs=1, batch_size=50), dpsgd.DPConfig(0.0)
    )
    assert math.isinf(res.epsilon)
    assert math.isinf(res.epsilon_prv)


@needs_opacus
@pytest.mark.validation
@pytest.mark.parametrize(("arch", "shape", "classes"), CASES)
def test_per_sample_grads_match_opacus(arch, shape, classes) -> None:  # type: ignore[no-untyped-def]
    import warnings

    from opacus import GradSampleModule

    model, x, y = _model_batch(arch, shape, classes, b=12)
    ours, _ = dpsgd.per_sample_grads(model, x, y, "hooks")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsm = GradSampleModule(model, loss_reduction="sum")
        F.cross_entropy(gsm(x), y, reduction="sum").backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(ours[name], p.grad_sample, rtol=1e-4, atol=1e-6)
