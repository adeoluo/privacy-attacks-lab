"""Dataset conversion, packing and the restricted unpickler."""

from __future__ import annotations

import io
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from privacy_attacks_lab import data


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _fake_cifar(path: Path, per_batch: int = 4) -> None:
    rng = np.random.default_rng(0)
    with tarfile.open(path, "w:gz") as tar:
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            batch = {
                b"data": rng.integers(0, 256, size=(per_batch, 3072), dtype=np.uint8),
                b"labels": rng.integers(0, 10, size=per_batch).tolist(),
            }
            _add(tar, f"cifar-10-batches-py/{name}", pickle.dumps(batch))


def test_prepare_cifar10_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "cifar.tgz"
    _fake_cifar(raw)
    ds = data.prepare_cifar10(raw, tmp_path / "cifar10.npz")
    assert ds.x.shape == (24, 32, 32, 3) and ds.is_image
    assert ds.input_shape == (3, 32, 32)
    again = data.load_npz(tmp_path / "cifar10.npz", "cifar10")
    np.testing.assert_array_equal(again.x, ds.x)
    np.testing.assert_array_equal(again.y, ds.y)


def test_cifar_channel_layout(tmp_path: Path) -> None:
    raw = tmp_path / "c.tgz"
    row = np.concatenate([np.full(1024, 10), np.full(1024, 20), np.full(1024, 30)]).astype(np.uint8)
    with tarfile.open(raw, "w:gz") as tar:
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            _add(tar, name, pickle.dumps({b"data": row[None, :], b"labels": [3]}))
    ds = data.prepare_cifar10(raw, tmp_path / "o.npz")
    assert tuple(ds.x[0, 5, 7]) == (10, 20, 30)


class _Evil:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.getcwd, ())  # inert stand-in for a malicious callable


def test_safe_unpickle_refuses_globals() -> None:
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        data.safe_unpickle(pickle.dumps(_Evil()))
    arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(data.safe_unpickle(pickle.dumps({b"a": arr}))[b"a"], arr)


def test_prepare_purchase(tmp_path: Path) -> None:
    raw = tmp_path / "p.tgz"
    rng = np.random.default_rng(1)
    rows = [
        f"{rng.integers(1, 101)}," + ",".join(map(str, rng.integers(0, 2, 600))) for _ in range(7)
    ]
    with tarfile.open(raw, "w:gz") as tar:
        _add(tar, "dataset_purchase", ("\n".join(rows) + "\n\n").encode())
    ds = data.prepare_purchase100(raw, tmp_path / "purchase100.npz")
    assert ds.x.shape == (7, 600) and ds.num_classes == 100
    assert ds.y.min() >= 0 and ds.y.max() <= 99
    assert ds.y[0] == int(rows[0].split(",")[0]) - 1
    back = data.load_npz(tmp_path / "purchase100.npz", "purchase100")
    np.testing.assert_array_equal(back.x, ds.x)


def test_prepare_texas(tmp_path: Path) -> None:
    raw = tmp_path / "t.tgz"
    rng = np.random.default_rng(2)
    feats = rng.integers(0, 2, size=(5, 6169))
    with tarfile.open(raw, "w:gz") as tar:
        _add(tar, "texas/100/feats", "\n".join(",".join(map(str, r)) for r in feats).encode())
        _add(tar, "texas/100/labels", b"1\n100\n5\n7\n9\n")
    ds = data.prepare_texas100(raw, tmp_path / "texas100.npz")
    np.testing.assert_array_equal(ds.x, feats)
    assert list(ds.y) == [0, 99, 4, 6, 8]


def test_bad_rows_are_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "p.tgz"
    with tarfile.open(raw, "w:gz") as tar:
        _add(tar, "dataset_purchase", b"3,0,1,2\n")
    with pytest.raises(ValueError, match="binary features"):
        data.prepare_purchase100(raw, tmp_path / "x.npz")


def test_load_dispatch_and_errors(tmp_path: Path) -> None:
    assert len(data.load("synthetic")) == 2000
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load("cifar10", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        data.load("mnist", tmp_path)  # type: ignore[arg-type]


def test_load_converts_once(tmp_path: Path) -> None:
    _fake_cifar(tmp_path / data.RAW_FILES["cifar10"], per_batch=2)
    first = data.load("cifar10", tmp_path)
    assert (tmp_path / "cifar10.npz").exists()
    (tmp_path / data.RAW_FILES["cifar10"]).unlink()
    second = data.load("cifar10", tmp_path)
    np.testing.assert_array_equal(first.x, second.x)


def test_dataset_validation_and_subset() -> None:
    ds = data.synthetic(n=50, features=8, num_classes=3)
    sub = ds.subset(np.arange(10))
    assert len(sub) == 10 and sub.input_shape == (8,)
    with pytest.raises(ValueError):
        data.Dataset("x", np.zeros((3, 2)), np.zeros(2, dtype=np.int64), 2)
    with pytest.raises(ValueError):
        data.Dataset("x", np.zeros((2, 2)), np.array([0, 5]), 2)


def test_synthetic_images_are_learnable_shape() -> None:
    ds = data.synthetic(n=20, image=True, num_classes=2)
    assert ds.x.shape == (20, 8, 8, 3) and ds.x.dtype == np.uint8
