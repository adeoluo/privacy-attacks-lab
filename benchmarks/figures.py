"""Figures for the README and the evaluation report, from benchmarks/results only.

Usage:
    uv run --no-sync --group plots python benchmarks/figures.py

Writes PNGs to docs/figures/. Every figure reads committed result files, so it can
be regenerated without retraining anything; missing result files are skipped.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT.parent / "docs" / "figures"

ATTACK_STYLE = {
    "loss": ("Loss threshold", "#94A3B8"),
    "calibrated": ("Calibrated loss", "#0EA5E9"),
    "shadow": ("Shadow models", "#10B981"),
    "lira-offline-fixed": ("LiRA offline", "#F59E0B"),
    "lira-online-fixed": ("LiRA online", "#6366F1"),
}


def _load(name: str) -> Any:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(name: str) -> list[dict[str, Any]]:
    path = RESULTS / name
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def roc_figure(presets: list[str]) -> None:
    data = [(p, _load(f"mia_{p}.json")) for p in presets]
    data = [(p, d) for p, d in data if d]
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.6), squeeze=False)
    for ax, (preset, d) in zip(axes[0], data, strict=True):
        for key, (label, color) in ATTACK_STYLE.items():
            roc = d["roc_target0"].get(key)
            if roc is None:
                continue
            auc = d["table"][key]["auc"]
            t = d["table"][key]["tpr@0.001"]
            ax.plot(
                roc["fpr"],
                roc["tpr"],
                color=color,
                lw=1.8,
                label=f"{label} (AUC {auc:.3f}, TPR@0.1% {100 * t:.2f}%)",
            )
        grid = np.logspace(-5, 0, 50)
        ax.plot(grid, grid, "k--", lw=0.8, label="Chance")
        ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1e-5, 1))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{preset}: {d['num_models']} models, target 0")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3, which="both", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "roc_log.png", dpi=140)
    plt.close(fig)


def scaling_figure(presets: list[str]) -> None:
    data = [(p, _load(f"mia_{p}.json")) for p in presets]
    data = [(p, d) for p, d in data if d and d.get("scaling")]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {"lira-online-fixed": "-", "lira-offline-fixed": "--"}
    colors = ["#6366F1", "#10B981", "#F59E0B", "#0EA5E9"]
    for (preset, d), color in zip(data, colors, strict=False):
        for variant, ls in styles.items():
            rows = d["scaling"].get(variant, [])
            if not rows:
                continue
            refs = [r["refs"] for r in rows]
            tpr = [100 * r["tpr@0.001"] for r in rows]
            name = "online" if "online" in variant else "offline"
            ax.plot(refs, tpr, ls, marker="o", ms=3, color=color, label=f"{preset}, LiRA {name}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Reference models")
    ax.set_ylabel("TPR at 0.1% FPR (%)")
    ax.set_title("LiRA strength against the number of reference models")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "lira_scaling.png", dpi=140)
    plt.close(fig)


def dp_utility_figure(datasets: list[str]) -> None:
    present = [(d, _jsonl(f"dp_utility_{d}.jsonl")) for d in datasets]
    present = [(d, rows) for d, rows in present if rows]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.0), squeeze=False)
    for ax, (name, rows) in zip(axes[0], present, strict=True):
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by[r["setting"]].append(r)
        eps_rows = sorted(
            (k for k in by if k.startswith("eps=")), key=lambda k: float(k.split("=")[1])
        )
        xs = [float(np.mean([r["epsilon_rdp"] for r in by[k]])) for k in eps_rows]
        ys = [100 * np.mean([r["test_acc"] for r in by[k]]) for k in eps_rows]
        es = [
            100 * np.std([r["test_acc"] for r in by[k]], ddof=1) if len(by[k]) > 1 else 0.0
            for k in eps_rows
        ]
        ax.errorbar(xs, ys, yerr=es, marker="o", color="#6366F1", capsize=3, label="DP-SGD")
        for key, color in (("non-private", "#10B981"), ("clip only", "#F59E0B")):
            if key in by:
                acc = 100 * np.mean([r["test_acc"] for r in by[key]])
                ax.axhline(acc, color=color, ls="--", lw=1.2, label=f"{key} ({acc:.1f}%)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("epsilon (RDP accountant, delta = 1e-5)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{name}: privacy and utility")
        ax.grid(alpha=0.3, lw=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "dp_utility.png", dpi=140)
    plt.close(fig)


def dp_attack_figure(preset: str, dp_presets: dict[str, str]) -> None:
    """LiRA TPR at low FPR against epsilon, with the bound DP itself implies."""
    from privacy_attacks_lab.accountants import max_tpr

    points = []
    for name, label in dp_presets.items():
        d = _load(f"mia_{name}.json")
        if d is None:
            continue
        eps = d["models"].get("epsilon", {}).get("mean", math.inf)
        points.append((eps, label, d["table"]))
    base = _load(f"mia_{preset}.json")
    if not points or base is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    finite = sorted(p[0] for p in points if math.isfinite(p[0]))
    grid = np.logspace(math.log10(min(finite) / 1.5), math.log10(max(finite) * 4), 60)
    for fpr, color in ((0.001, "#6366F1"), (0.01, "#10B981")):
        bound = 100 * np.array([float(max_tpr(fpr, e, 1e-5)) for e in grid])
        ax.plot(grid, bound, ":", color=color, label=f"DP bound at {100 * fpr:g}% FPR")
        xs = [p[0] for p in sorted(points)]
        ys = [100 * p[2]["lira-online-fixed"][f"tpr@{fpr:g}"] for p in sorted(points)]
        ax.plot(xs, ys, "o-", color=color, label=f"LiRA online at {100 * fpr:g}% FPR")
        nonpriv = 100 * base["table"]["lira-online-fixed"][f"tpr@{fpr:g}"]
        ax.axhline(nonpriv, color=color, ls="--", lw=1, label=f"non-private ({nonpriv:.2f}%)")
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("epsilon (RDP accountant, delta = 1e-5)")
    ax.set_ylabel("TPR (%)")
    ax.set_title(f"{preset}: attack success under DP-SGD")
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "dp_attack.png", dpi=140)
    plt.close(fig)


def exposure_figure() -> None:
    d = _load("lm_canaries.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = {"base": "#94A3B8", "ft": "#6366F1", "dp8": "#10B981", "dp1": "#F59E0B"}
    labels = {
        "base": "Pretrained",
        "ft": "Fine-tuned",
        "dp8": "DP-SGD eps 8",
        "dp1": "DP-SGD eps 1",
    }
    for name in colors:
        rows = d.get(name, {}).get("canaries")
        if not rows:
            continue
        by: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            by[r["repeats"]].append(r["exposure"])
        reps = sorted(by)
        ax.plot(
            reps,
            [np.mean(by[k]) for k in reps],
            "o-",
            color=colors.get(name, "k"),
            label=labels[name],
        )
    ax.axhline(math.log2(1e6), color="k", ls=":", lw=1, label="log2(10^6): fully extracted")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Times the canary was inserted")
    ax.set_ylabel("Exposure (bits, exact)")
    ax.set_title("Pythia-70M fine-tuned on WikiText-2 with canaries")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "exposure.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    roc_figure(["cifar10", "purchase100"])
    scaling_figure(["cifar10", "purchase100"])
    dp_utility_figure(["cifar10", "purchase100"])
    dp_attack_figure("purchase100", {"purchase100-dp8": "eps 8", "purchase100-dp1": "eps 1"})
    exposure_figure()
    print("figures in", FIGS)


if __name__ == "__main__":
    main()
