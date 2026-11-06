"""Privacy/utility curves of DP-SGD: test accuracy against epsilon.

Usage:
    uv run --no-sync python benchmarks/dp_utility.py cifar10 --eps 1 2 4 8 --seeds 0 1 2
    uv run --no-sync python benchmarks/dp_utility.py purchase100 --eps 1 2 4 8 --seeds 0 1 2

For every target epsilon (at delta = 1e-5) the noise multiplier is calibrated
with the RDP accountant for the fixed schedule (epochs, expected batch size),
then the model is trained from scratch with DP-SGD (Poisson sampling, flat
per-example clipping, Gaussian noise; ghost clipping engine). The PRV epsilon of
the same run is recorded next to it. A non-private model with the same
architecture, optimiser and epochs is the utility ceiling; ``clip only`` (clipping
without noise) separates the cost of clipping from the cost of noise.

CIFAR-10 trains on the 50,000 training images and tests on the 10,000 test images.
Purchase100 trains on a seeded 80% of the 197,324 records and tests on the rest.
Results append to benchmarks/results/dp_utility_<dataset>.jsonl (one line per run),
so the script can be resumed; finished (setting, seed) pairs are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from privacy_attacks_lab.accountants import calibrate_noise, epsilon
from privacy_attacks_lab.data import load
from privacy_attacks_lab.dpsgd import DPConfig, train_dp
from privacy_attacks_lab.models import build
from privacy_attacks_lab.training import (
    DeviceData,
    TrainConfig,
    accuracy,
    get_device,
    predict_logits,
    train,
)

RESULTS = Path(__file__).resolve().parent / "results"

PRESETS: dict[str, dict[str, Any]] = {
    # GroupNorm CNN (0.11M parameters); DP-SGD favours large batches and few epochs.
    "cifar10": {
        "arch": "cnn",
        "epochs": 30,
        "batch": 1024,
        "lr": 2.0,
        "lr_np": 0.05,
        "momentum": 0.9,
    },
    "purchase100": {
        "arch": "mlp",
        "epochs": 30,
        "batch": 1024,
        "lr": 1.0,
        "lr_np": 0.05,
        "momentum": 0.9,
    },
}


def split(name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "cifar10":
        return np.arange(50000), np.arange(50000, 60000)
    rng = np.random.default_rng(12345)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def clean(row: dict[str, Any]) -> dict[str, Any]:
    """Infinite epsilons (no noise) are stored as null."""
    return {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in row.items()}


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(r["setting"], r["seed"]) for r in map(json.loads, path.read_text().splitlines())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(PRESETS))
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    p = dict(PRESETS[args.dataset])
    if args.epochs:
        p["epochs"] = args.epochs
    ds = load(args.dataset)  # type: ignore[arg-type]
    tr, te = split(args.dataset, len(ds))
    dev = get_device()
    data = DeviceData(ds, dev)
    n = len(tr)
    q = p["batch"] / n
    steps = p["epochs"] * max(1, round(1 / q))
    out_path = RESULTS / f"dp_utility_{args.dataset}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out_path)

    settings: list[tuple[str, float | None]] = [(f"eps={e:g}", e) for e in args.eps]
    if not args.no_baselines:
        settings += [("clip only", 0.0), ("non-private", None)]
    for setting, target in settings:
        for seed in args.seeds:
            if (setting, seed) in finished:
                continue
            model = build(p["arch"], ds.input_shape, ds.num_classes)
            row: dict[str, Any] = {
                "dataset": args.dataset,
                "setting": setting,
                "seed": seed,
                "arch": p["arch"],
                "epochs": p["epochs"],
                "expected_batch": p["batch"],
                "train_size": n,
                "test_size": len(te),
                "delta": args.delta,
                "clip": args.clip,
            }
            t0 = time.perf_counter()
            if target is None:
                cfg = TrainConfig(
                    epochs=p["epochs"], batch_size=256, lr=p["lr_np"], seed=seed, augment=False
                )
                res = train(model, data, tr, cfg)
                row.update(sigma=None, epsilon_rdp=math.inf, epsilon_prv=math.inf, steps=None)
                row["final_loss"] = res.final_loss
            else:
                sigma = (
                    calibrate_noise(target, args.delta, q, steps, accountant="rdp")
                    if target > 0
                    else 0.0
                )
                cfg = TrainConfig(
                    epochs=p["epochs"],
                    batch_size=p["batch"],
                    lr=p["lr"],
                    momentum=p["momentum"],
                    seed=seed,
                    augment=False,
                )
                dres = train_dp(model, data, tr, cfg, DPConfig(sigma, args.clip, args.delta))
                row.update(
                    sigma=sigma,
                    steps=dres.steps,
                    sample_rate=dres.sample_rate,
                    epsilon_rdp=dres.epsilon,
                    epsilon_prv=dres.epsilon_prv,
                    epsilon_gdp=epsilon(q, sigma, dres.steps, args.delta, accountant="gdp")
                    if sigma > 0
                    else math.inf,
                    final_loss=dres.final_loss,
                )
            row["train_seconds"] = time.perf_counter() - t0
            logits = predict_logits(model, data, te, queries="none")
            row["test_acc"] = accuracy(logits, ds.y[te])
            sub = np.random.default_rng(seed).choice(tr, size=min(10000, n), replace=False)
            row["train_acc"] = accuracy(predict_logits(model, data, sub, queries="none"), ds.y[sub])
            row["machine"] = f"{platform.machine()} {dev.type}"
            row = clean(row)
            print(json.dumps(row), flush=True)
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
