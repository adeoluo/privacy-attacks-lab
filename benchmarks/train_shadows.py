"""Train the model families behind the membership inference benchmarks.

Usage:
    uv run --no-sync python benchmarks/train_shadows.py cifar10 [--models 0-63]

Runs are written to $PAL_DATA_DIR/runs/<preset>/ (outside the repository) and can be
resumed: finished models are skipped. Per-model timings and accuracies are appended
to log.jsonl there and summarised by benchmarks/mia_benchmark.py.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from privacy_attacks_lab.data import DEFAULT_DATA_DIR
from privacy_attacks_lab.shadow import ShadowRun, ShadowSpec

PRESETS: dict[str, ShadowSpec] = {
    # SmallCNN (0.11M parameters), no augmentation, 15 epochs on 25,000 images.
    "cifar10": ShadowSpec("cifar10", "cnn", num_models=64, epochs=15, lr=0.05, weight_decay=5e-4),
    # Same, trained with random crop and flip augmentation (ablation).
    "cifar10-aug": ShadowSpec(
        "cifar10", "cnn", num_models=16, epochs=15, lr=0.1, weight_decay=5e-4, augment=True
    ),
    # MLP 600-512-256-128-100 (Tanh) on 98,662 of the 197,324 records.
    "purchase100": ShadowSpec(
        "purchase100",
        "mlp",
        num_models=64,
        epochs=20,
        lr=0.05,
        weight_decay=1e-5,
        queries="none",
    ),
    # Smaller pools, the usual setting in the literature (ablation of training set size).
    "purchase100-20k": ShadowSpec(
        "purchase100",
        "mlp",
        num_models=64,
        pool_size=20000,
        epochs=40,
        lr=0.05,
        weight_decay=1e-5,
        queries="none",
    ),
    # MLP 6169-512-256-128-100 on 33,665 of the 67,330 records.
    "texas100": ShadowSpec(
        "texas100",
        "mlp",
        num_models=64,
        epochs=30,
        lr=0.1,
        batch_size=128,
        weight_decay=1e-5,
        queries="none",
    ),
}


def parse_models(text: str | None) -> list[int] | None:
    if text is None:
        return None
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--models", help="range such as 0-31")
    ap.add_argument("--num-models", type=int)
    args = ap.parse_args()
    spec = PRESETS[args.preset]
    if args.num_models:
        spec = replace(spec, num_models=args.num_models)
    run = ShadowRun(DEFAULT_DATA_DIR / "runs" / args.preset, spec)
    run.train(parse_models(args.models), log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
