"""Models, training loops, the shadow-model harness and leave-one-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from privacy_attacks_lab.data import Dataset, synthetic
from privacy_attacks_lab.shadow import ShadowRun, ShadowSpec, Signals, make_masks, split_pool

from .conftest import needs_torch

torch = pytest.importorskip("torch")

from privacy_attacks_lab import evaluation, training  # noqa: E402
from privacy_attacks_lab.models import build, num_parameters  # noqa: E402

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- models


@needs_torch
@pytest.mark.parametrize(
    ("arch", "shape", "classes", "params"),
    [
        ("cnn", (3, 32, 32), 10, 114_186),
        ("cnn-small", (3, 32, 32), 10, None),
        ("mlp", (600,), 100, 484_836),
        ("mlp-small", (600,), 100, None),
    ],
)
def test_build_output_shapes(arch, shape, classes, params) -> None:  # type: ignore[no-untyped-def]
    m = build(arch, shape, classes)
    assert m(torch.zeros(2, *shape)).shape == (2, classes)
    if params is not None:
        assert num_parameters(m) == params


def test_build_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="architecture"):
        build("resnet", (3, 32, 32), 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="8 pixels"):
        build("cnn", (3, 4, 4), 10)


def test_models_have_no_batchnorm() -> None:
    for arch in ("cnn", "cnn-small", "mlp", "mlp-small"):
        shape = (3, 32, 32) if arch.startswith("cnn") else (600,)
        m = build(arch, shape, 10)  # type: ignore[arg-type]
        assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())


# --------------------------------------------------------------------------- training


def test_get_device_and_seed() -> None:
    assert training.get_device("cpu").type == "cpu"
    assert training.get_device("auto").type in ("cpu", "mps", "cuda")
    g1 = training.seed_everything(3)
    g2 = training.seed_everything(3)
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))


def test_random_crop_flip_preserves_shape_and_values() -> None:
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
    torch.manual_seed(0)
    out = training.random_crop_flip(x, pad=2)
    assert out.shape == x.shape
    # No padding: the only change is a horizontal flip.
    torch.manual_seed(0)
    same = training.random_crop_flip(x, pad=0)
    for i in range(2):
        assert torch.equal(same[i], x[i]) or torch.equal(same[i], torch.flip(x[i], dims=(2,)))


def test_device_data_normalises_images() -> None:
    ds = synthetic(n=20, num_classes=3, image=True)
    data = training.DeviceData(ds, CPU)
    xb, yb = data.batch(torch.arange(4))
    assert xb.shape == (4, 3, 8, 8)
    assert xb.dtype == torch.float32
    flipped, _ = data.batch(torch.arange(4), flip=True)
    assert torch.equal(flipped, torch.flip(xb, dims=(3,)))
    aug, _ = data.batch(torch.arange(4), augment=True)
    assert aug.shape == xb.shape
    assert torch.equal(yb, torch.from_numpy(ds.y[:4]))
    assert len(data) == 20


def test_train_reduces_loss_and_predicts() -> None:
    ds = synthetic(n=800, features=24, num_classes=4)
    data = training.DeviceData(ds, CPU)
    model = build("mlp-small", ds.input_shape, ds.num_classes)
    seen: list[float] = []
    res = training.train(
        model,
        data,
        np.arange(600),
        training.TrainConfig(epochs=4, batch_size=64, lr=0.1, augment=False),
        on_epoch=lambda e, loss: seen.append(loss),
    )
    assert res.epochs == 4
    assert len(seen) == 4
    assert res.history[-1] < res.history[0]
    logits = training.predict_logits(model, data, np.arange(600, 800))
    assert logits.shape == (200, 1, 4)
    assert training.accuracy(logits, ds.y[600:]) > 0.8
    assert training.accuracy(logits[:, 0], ds.y[600:]) > 0.8
    assert np.isnan(training.accuracy(logits[:0], ds.y[:0]))
    assert training.TrainConfig().to_dict()["epochs"] == 20


def test_predict_logits_flip_queries_for_images() -> None:
    ds = synthetic(n=40, num_classes=2, image=True)
    data = training.DeviceData(ds, CPU)
    model = build("cnn-small", ds.input_shape, 2)
    out = training.predict_logits(model, data, queries="flip", batch_size=16)
    assert out.shape == (40, 2, 2)
    none = training.predict_logits(model, data, queries="none")
    np.testing.assert_allclose(none[:, 0], out[:, 0], rtol=1e-5, atol=1e-6)


def test_training_is_deterministic_on_cpu() -> None:
    ds = synthetic(n=300, features=16, num_classes=3)
    data = training.DeviceData(ds, CPU)
    cfg = training.TrainConfig(epochs=2, batch_size=50, seed=11, augment=False)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = build("mlp-small", ds.input_shape, 3)
        training.train(m, data, np.arange(300), cfg)
        outs.append(training.predict_logits(m, data))
    np.testing.assert_array_equal(outs[0], outs[1])


# --------------------------------------------------------------------------- shadow harness


@settings(max_examples=40, deadline=None)
@given(models=st.integers(2, 40), n=st.integers(0, 200), seed=st.integers(0, 99))
def test_masks_are_balanced_per_example(models: int, n: int, seed: int) -> None:
    m = make_masks(models, n, seed)
    assert m.shape == (models, n)
    assert (m.sum(axis=0) == round(models / 2)).all()


def test_make_masks_validation() -> None:
    with pytest.raises(ValueError):
        make_masks(0, 5, 0)
    with pytest.raises(ValueError, match="pkeep"):
        make_masks(4, 5, 0, pkeep=1.0)
    m = make_masks(10, 50, 0, pkeep=0.3)
    assert (m.sum(0) == 3).all()


def test_split_pool_cifar_and_tabular() -> None:
    cifar = Dataset("cifar10", np.zeros((60000, 1, 1, 1), np.uint8), np.zeros(60000, np.int64), 10)
    pool, test = split_pool(cifar, ShadowSpec())
    assert (len(pool), len(test)) == (50000, 10000)
    tab = synthetic(n=500)
    pool, test = split_pool(tab, ShadowSpec(dataset="synthetic", pool_size=100))
    assert len(pool) == 100
    assert len(test) == 0
    assert np.all(np.diff(pool) > 0)


def _spec(**kw) -> ShadowSpec:  # type: ignore[no-untyped-def]
    base = {
        "dataset": "synthetic",
        "arch": "mlp-small",
        "num_models": 6,
        "epochs": 3,
        "batch_size": 100,
        "lr": 0.1,
        "queries": "none",
    }
    base.update(kw)
    return ShadowSpec(**base)  # type: ignore[arg-type]


@needs_torch
def test_shadow_run_train_resume_and_load(tmp_path: Path) -> None:
    ds = synthetic(n=600, features=32, num_classes=4)
    run = ShadowRun(tmp_path / "run", _spec())
    lines: list[str] = []
    rows = run.train(range(4), device="cpu", dataset=ds, log=lines.append)
    assert [r["model"] for r in rows] == [0, 1, 2, 3]
    assert len(lines) == 4
    assert run.completed() == [0, 1, 2, 3]
    # Resuming skips finished models and trains the rest.
    again = ShadowRun(tmp_path / "run")
    rows2 = again.train(device="cpu", dataset=ds, log=lambda s: None)
    assert [r["model"] for r in rows2] == [4, 5]
    sig = again.load()
    assert sig.num_models == 6
    assert sig.phi.shape == (6, 600, 1)
    assert sig.topk.shape == (6, 600, 1, 3)
    assert sig.masks.shape == (6, 600)
    assert len(sig.meta) == 6
    assert sig.meta[0]["train_acc"] > sig.meta[0]["out_acc"]
    sub = sig.select([0, 2])
    assert sub.num_models == 2
    assert sub.meta[1]["model"] == 2
    log = (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    assert len(log) == 6
    assert json.loads(log[0])["model"] == 0


@needs_torch
def test_shadow_run_dp(tmp_path: Path) -> None:
    ds = synthetic(n=300, features=16, num_classes=3)
    run = ShadowRun(tmp_path / "dp", _spec(num_models=2, noise_multiplier=1.0, epochs=2))
    rows = run.train(device="cpu", dataset=ds, log=lambda s: None)
    assert all(np.isfinite(r["epsilon"]) for r in rows)


def test_shadow_run_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ShadowRun(tmp_path / "missing")
    (tmp_path / "x").mkdir()
    run = ShadowRun(tmp_path / "x", _spec())
    with pytest.raises(FileNotFoundError, match="no trained"):
        run.load()
    run.init(synthetic(n=50))
    with pytest.raises(ValueError, match="different spec"):
        ShadowRun(tmp_path / "x", _spec(epochs=99))
    assert ShadowSpec.from_dict(_spec().__dict__) == _spec()


# --------------------------------------------------------------------------- evaluation


def _signals(models: int = 12, n: int = 300, seed: int = 0) -> Signals:
    rng = np.random.default_rng(seed)
    masks = make_masks(models, n, seed)
    d = rng.normal(0, 3, size=n)
    phi = (d + rng.normal(size=(models, n)) + 2.0 * masks)[:, :, None]
    p = 1 / (1 + np.exp(-phi))
    loss = -np.log(p)
    topk = np.concatenate([p, (1 - p) * 0.6, (1 - p) * 0.4], axis=-1)[:, :, None, :]
    return Signals(
        phi=phi,
        loss=loss,
        topk=topk,
        correct=p > 0.5,
        masks=masks,
        labels=rng.integers(0, 2, size=n),
    )


def test_attack_scores_every_variant() -> None:
    sig = _signals()
    extra = ["calibrated-online", "lira-offline-pdf"]
    for v in [*evaluation.VARIANTS, *extra]:
        s = evaluation.attack_scores(sig, 0, v, shadow_models=4)
        assert s.shape == (300,)
        assert np.isfinite(s).all(), v


def test_attack_scores_errors() -> None:
    sig = _signals()
    with pytest.raises(ValueError, match="own reference"):
        evaluation.attack_scores(sig, 0, "loss", refs=[0, 1])
    with pytest.raises(ValueError, match="unknown"):
        evaluation.attack_scores(sig, 0, "magic")
    with pytest.raises(ValueError, match="unknown"):
        evaluation.attack_scores(sig, 0, "lira-sideways")


def test_evaluate_aggregates_targets() -> None:
    sig = _signals(models=10, n=400)
    res = evaluation.evaluate(sig, targets=[0, 1, 2], num_refs=6, shadow_models=4)
    agg = {v: r.aggregate() for v, r in res.items()}
    assert agg["lira-online"]["targets"] == 3
    assert agg["lira-online"]["auc"] > agg["loss"]["auc"]
    assert agg["calibrated"]["auc"] > agg["loss"]["auc"]
    assert "tpr@0.001_std" in agg["loss"]
    one = evaluation.evaluate(sig, targets=[0], variants=["loss"])
    assert one["loss"].aggregate()["auc_std"] == 0.0
