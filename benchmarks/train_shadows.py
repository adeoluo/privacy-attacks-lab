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

from privacy_attacks_lab.accountants import calibrate_noise
from privacy_attacks_lab.data import DEFAULT_DATA_DIR, load
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
    # DP-SGD families for attack success against epsilon (same pool and masks scheme):
    # expected batch 1,024, 30 epochs; sigma is calibrated to the target epsilon at delta 1e-5.
    **{
        f"purchase100-dp{eps:g}": ShadowSpec(
            "purchase100",
            "mlp",
            num_models=16,
            epochs=30,
            batch_size=1024,
            lr=1.0,
            weight_decay=0.0,
            queries="none",
            extra={"target_epsilon": eps},
        )
        for eps in (1.0, 8.0)
    },
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
    if "target_epsilon" in spec.extra and spec.noise_multiplier is None:
        n = round(len(load(spec.dataset)) * 0.5) if spec.pool_size is None else spec.pool_size // 2
        q = min(1.0, spec.batch_size / n)
        steps = spec.epochs * max(1, round(1.0 / q))
        sigma = calibrate_noise(spec.extra["target_epsilon"], spec.delta, q, steps)
        print(f"target epsilon {spec.extra['target_epsilon']}: sigma {sigma:.6f}", flush=True)
        spec = replace(spec, noise_multiplier=sigma)
    run = ShadowRun(DEFAULT_DATA_DIR / "runs" / args.preset, spec)
    run.train(parse_models(args.models), log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
