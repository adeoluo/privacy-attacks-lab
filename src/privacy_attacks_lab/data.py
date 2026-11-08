"""Datasets: CIFAR-10, Purchase100 and Texas100, converted once to compact ``.npz`` files.

The raw archives come from their public sources (``scripts/download_data.sh``):

- CIFAR-10 (Krizhevsky 2009): 60,000 32x32 colour images in 10 classes, stored as
  pickled numpy batches. They are read with a restricted unpickler that only
  resolves the numpy constructors the archive needs, so a tampered archive cannot
  execute code while loading.
- Purchase100 (Shokri et al. 2017): 197,324 shopping records with 600 binary
  features, 100 classes.
- Texas100 (Shokri et al. 2017): 67,330 hospital discharge records with 6,169
  binary features, 100 classes (the ``texas/100`` split).

Both tabular files are comma separated text (about 0.2 GB and 0.8 GB); they are
parsed line by line so memory stays at the size of the packed result, and the
binary features of Texas100 are stored bit packed (52 MB instead of 415 MB).
"""

from __future__ import annotations

import io
import os
import pickle
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

DatasetName = Literal["cifar10", "purchase100", "texas100", "synthetic"]
DATASETS: tuple[DatasetName, ...] = ("cifar10", "purchase100", "texas100", "synthetic")

#: Default location of raw archives and prepared files (outside the repository);
#: override with the ``PAL_DATA_DIR`` environment variable.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "PAL_DATA_DIR",
        Path(__file__).resolve().parents[3] / ".bench-data" / "privacy-attacks-lab",
    )
)

RAW_FILES: dict[str, str] = {
    "cifar10": "cifar-10-python.tar.gz",
    "purchase100": "dataset_purchase.tgz",
    "texas100": "dataset_texas.tgz",
}


@dataclass(frozen=True)
class Dataset:
    """Features ``x`` (uint8) and labels ``y`` (int64, 0-based).

    Images are ``(n, 32, 32, 3)``; tabular records are ``(n, features)`` with 0/1 values.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    num_classes: int

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        if len(self.y) and (self.y.min() < 0 or self.y.max() >= self.num_classes):
            raise ValueError("labels out of range")

    def __len__(self) -> int:
        return len(self.y)

    @property
    def is_image(self) -> bool:
        return self.x.ndim == 4

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Model input shape without the batch axis (channels first for images)."""
        if self.is_image:
            return (int(self.x.shape[3]), int(self.x.shape[1]), int(self.x.shape[2]))
        return (int(self.x.shape[1]),)

    def subset(self, idx: np.ndarray) -> Dataset:
        return Dataset(self.name, self.x[idx], self.y[idx], self.num_classes)


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Resolves only the globals a pickled numpy array needs.

    Protocols up to 4 rebuild arrays with ``_reconstruct``; protocol 5 (the default from
    Python 3.14) uses ``_frombuffer``, which only wraps a buffer and runs no other code.
    """

    _ALLOWED: ClassVar[set[tuple[str, str]]] = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")
        return super().find_class(module, name)


def safe_unpickle(data: bytes) -> Any:
    """Unpickle ``data`` allowing numpy arrays and builtin containers only."""
    return _NumpyOnlyUnpickler(io.BytesIO(data), encoding="bytes").load()


def prepare_cifar10(archive: Path, out: Path) -> Dataset:
    """Convert the CIFAR-10 python archive: 50,000 training images then 10,000 test images."""
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for name in names:
            handle = tar.extractfile(members[name])
            if handle is None:  # pragma: no cover - members are regular files
                raise ValueError(f"cannot read {name}")
            batch = safe_unpickle(handle.read())
            data = np.asarray(batch[b"data"], dtype=np.uint8)
            xs.append(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.asarray(batch[b"labels"], dtype=np.int64))
    ds = Dataset("cifar10", np.ascontiguousarray(np.concatenate(xs)), np.concatenate(ys), 10)
    save(ds, out)
    return ds


def _parse_binary_csv(lines: Iterator[bytes], width: int, label_first: bool) -> Iterator[Any]:
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if label_first:
            head, _, rest = line.partition(b",")
            label: int | None = int(head)
        else:
            label, rest = None, line
        # Every feature is a single digit, so the digits sit at even offsets.
        row = np.frombuffer(rest, dtype=np.uint8)[::2] - ord("0")
        if row.shape[0] != width or bool((row > 1).any()):
            raise ValueError(f"expected {width} binary features, got {row.shape[0]}")
        yield label, row


def prepare_purchase100(archive: Path, out: Path) -> Dataset:
    """Convert Purchase100: ``label,f1,...,f600`` per line, labels 1..100."""
    xs: list[np.ndarray] = []
    labels: list[int] = []
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.isfile())
        handle = tar.extractfile(member)
        if handle is None:  # pragma: no cover
            raise ValueError("cannot read the purchase file")
        for label, row in _parse_binary_csv(iter(handle), 600, label_first=True):
            labels.append(int(label) - 1)
            xs.append(row)
    ds = Dataset("purchase100", np.stack(xs), np.asarray(labels, dtype=np.int64), 100)
    save(ds, out)
    return ds


def prepare_texas100(archive: Path, out: Path) -> Dataset:
    """Convert Texas100: ``texas/100/feats`` (6,169 binary columns) and ``texas/100/labels``."""
    with tarfile.open(archive, "r:gz") as tar:
        lab = tar.extractfile("texas/100/labels")
        feats = tar.extractfile("texas/100/feats")
        if lab is None or feats is None:  # pragma: no cover
            raise ValueError("texas/100 is missing from the archive")
        y = np.array([int(v) - 1 for v in lab.read().split()], dtype=np.int64)
        packed = [np.packbits(row) for _, row in _parse_binary_csv(iter(feats), 6169, False)]
    x = np.unpackbits(np.stack(packed), axis=1, count=6169)
    ds = Dataset("texas100", x, y, 100)
    save(ds, out)
    return ds


def synthetic(
    n: int = 2000, features: int = 32, num_classes: int = 4, seed: int = 0, image: bool = False
) -> Dataset:
    """A small labelled dataset for tests and CI (binary features or 8x8 images)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=n)
    if image:
        centers = rng.integers(0, 256, size=(num_classes, 8, 8, 3))
        noise = rng.normal(0, 40, size=(n, 8, 8, 3))
        x = np.clip(centers[y] + noise, 0, 255).astype(np.uint8)
    else:
        protos = rng.random((num_classes, features)) < 0.5
        flips = rng.random((n, features)) < 0.25
        x = (protos[y] ^ flips).astype(np.uint8)
    return Dataset("synthetic", x, y.astype(np.int64), num_classes)


def save(ds: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ds.is_image:
        np.savez(path, x=ds.x, y=ds.y, num_classes=ds.num_classes, packed=False)
    else:
        np.savez(
            path,
            x=np.packbits(ds.x, axis=1),
            y=ds.y,
            num_classes=ds.num_classes,
            packed=True,
            width=ds.x.shape[1],
        )


def load_npz(path: Path, name: str) -> Dataset:
    with np.load(path, allow_pickle=False) as f:
        x = f["x"]
        if bool(f["packed"]):
            x = np.unpackbits(x, axis=1, count=int(f["width"]))
        return Dataset(name, x, f["y"].astype(np.int64), int(f["num_classes"]))


_PREPARE = {
    "cifar10": prepare_cifar10,
    "purchase100": prepare_purchase100,
    "texas100": prepare_texas100,
}


def load(name: DatasetName, data_dir: Path | None = None) -> Dataset:
    """Load a prepared dataset, converting the raw archive on first use."""
    if name == "synthetic":
        return synthetic()
    if name not in _PREPARE:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    prepared = root / f"{name}.npz"
    if prepared.exists():
        return load_npz(prepared, name)
    raw = root / RAW_FILES[name]
    if not raw.exists():
        raise FileNotFoundError(f"{raw} not found; run scripts/download_data.sh")
    return _PREPARE[name](raw, prepared)
