"""Membership inference benchmark on a trained model family (leave-one-out).

Usage:
    uv run --no-sync python benchmarks/mia_benchmark.py cifar10 [--targets 16]

Every one of the first ``--targets`` models plays the target in turn; all other
models of the family are its reference (shadow) models. The script writes
``benchmarks/results/mia_<preset>.json`` with

- ``table``: mean and standard deviation over targets of AUC, balanced accuracy
  and TPR at 0.1%, 1% and 10% FPR for every attack variant, including the
  authors' released LiRA scoring as a reference;
- ``bootstrap``: 95% bootstrap intervals (1,000 resamples of examples) for target 0;
- ``scaling``: LiRA TPR at low FPR as the number of reference models grows;
- ``roc``: target 0's ROC curves sampled on a log-spaced FPR grid (for the figures);
- ``models``: training cost and accuracy of the family (from log.jsonl).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from privacy_attacks_lab import metrics
from privacy_attacks_lab.data import DEFAULT_DATA_DIR
from privacy_attacks_lab.evaluation import REFERENCE_VARIANTS, VARIANTS, attack_scores, evaluate
from privacy_attacks_lab.shadow import ShadowRun

RESULTS = Path(__file__).resolve().parent / "results"
SCALING_VARIANTS = ("lira-online", "lira-online-fixed", "lira-offline", "lira-offline-fixed")
SCALING_REFS = (1, 2, 4, 8, 16, 32, 63)


def model_summary(run: ShadowRun) -> dict[str, Any]:
    rows = [json.loads(x) for x in (run.root / "log.jsonl").read_text().splitlines()]
    out: dict[str, Any] = {"count": len(rows), "params": rows[0]["params"]}
    for key in ("seconds", "train_acc", "out_acc", "test_acc", "epsilon"):
        vals = np.array([r[key] for r in rows if key in r], dtype=np.float64)
        if len(vals):
            out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    out["total_train_hours"] = float(sum(r["seconds"] for r in rows) / 3600)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--scaling-targets", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--shadow-models", type=int, default=8)
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument(
        "--queries",
        choices=["all", "first"],
        default="all",
        help="'first' keeps only the identity view (images); 'all' averages over views",
    )
    args = ap.parse_args()

    run = ShadowRun(DEFAULT_DATA_DIR / "runs" / args.preset)
    sig = run.load()
    views = sig.num_queries
    if args.queries == "first":
        sig = sig.query(0)
    suffix = "_firstquery" if args.queries == "first" and views > 1 else ""
    m = sig.num_models
    print(f"{args.preset}: {m} models, {sig.masks.shape[1]} examples", flush=True)
    variants = list(VARIANTS) + ([] if args.no_reference else list(REFERENCE_VARIANTS))
    targets = list(range(min(args.targets, m)))

    t0 = time.perf_counter()
    res = evaluate(sig, targets, variants, shadow_models=args.shadow_models)
    table = {v: r.aggregate() for v, r in res.items()}
    for v, row in table.items():
        print(
            f"  {v:<24} auc={row['auc']:.4f} tpr@0.1%={row['tpr@0.001']:.4f} "
            f"tpr@1%={row['tpr@0.01']:.4f}",
            flush=True,
        )
    eval_seconds = time.perf_counter() - t0

    labels0 = sig.masks[0]
    boot: dict[str, Any] = {}
    roc: dict[str, Any] = {}
    for v in variants:
        s = attack_scores(sig, 0, v, shadow_models=args.shadow_models)
        boot[v] = {
            k: iv.to_dict() for k, iv in metrics.bootstrap(s, labels0, reps=args.reps).items()
        }
        fpr, tpr, _ = metrics.roc_curve(s, labels0)
        g, t = metrics.log_roc_points(fpr, tpr)
        roc[v] = {"fpr": g.tolist(), "tpr": t.tolist()}

    scaling: dict[str, Any] = {v: [] for v in SCALING_VARIANTS}
    st = list(range(min(args.scaling_targets, m)))
    for k in SCALING_REFS:
        if k > m - 1:
            continue
        for v in SCALING_VARIANTS:
            rows = []
            for t in st:
                refs = [i for i in range(m) if i != t][:k]
                s = attack_scores(sig, t, v, refs)
                rows.append(metrics.summarize(s, sig.masks[t]).to_dict())
            agg = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            scaling[v].append({"refs": k, **agg})
        print(
            f"  scaling refs={k}: lira-online "
            f"tpr@0.1%={scaling['lira-online'][-1]['tpr@0.001']:.4f}"
        )

    out = {
        "preset": args.preset,
        "queries": args.queries if views > 1 else "all",
        "spec": json.loads((run.root / "spec.json").read_text()),
        "num_models": m,
        "pool_size": int(sig.masks.shape[1]),
        "targets": targets,
        "shadow_models_for_shadow_attack": args.shadow_models,
        "variants": {**VARIANTS, **REFERENCE_VARIANTS},
        "table": table,
        "bootstrap_target0": boot,
        "scaling": scaling,
        "roc_target0": roc,
        "models": model_summary(run),
        "eval_seconds": eval_seconds,
        "machine": f"{platform.machine()} {platform.system()} {platform.python_version()}",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"mia_{args.preset}{suffix}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
