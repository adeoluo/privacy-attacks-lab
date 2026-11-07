"""Accountant parity with Opacus and speed of the Rust core.

Usage:
    uv run --no-sync python benchmarks/parity_speed.py

Part 1 (parity): on a grid of 48 DP-SGD settings (sample rate, noise multiplier,
steps) at delta = 1e-5, epsilon from this repo's RDP accountant (Rust core and
NumPy reference) and PRV accountant (41 of the settings; the widest PRV grids are skipped to
bound memory) is compared with Opacus 1.6
(``RDPAccountant`` and ``PRVAccountant``) and noise calibration with
``opacus.accountants.utils.get_noise_multiplier``.

Part 2 (speed): wall time per call of the RDP epsilon (Rust, NumPy, Opacus), the
ROC summary (Rust, NumPy, scikit-learn ``roc_curve`` + ``auc``) and a 1,000
replicate bootstrap (Rust, NumPy) on 100,000 scores. Medians of repeated runs.

Writes benchmarks/results/parity_speed.json.
"""

from __future__ import annotations

import itertools
import json
import platform
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from privacy_attacks_lab import metrics
from privacy_attacks_lab.accountants import DEFAULT_ORDERS, calibrate_noise, epsilon, rdp_epsilon

OUT = Path(__file__).resolve().parent / "results" / "parity_speed.json"
DELTA = 1e-5


def opacus_eps(kind: str, q: float, sigma: float, steps: int) -> float:
    from opacus.accountants import PRVAccountant, RDPAccountant

    acc = RDPAccountant() if kind == "rdp" else PRVAccountant()
    acc.history = [(sigma, q, steps)]
    if kind == "rdp":
        return float(acc.get_epsilon(DELTA, alphas=list(DEFAULT_ORDERS)))
    return float(acc.get_epsilon(DELTA, eps_error=0.01))


def timeit(fn: Callable[[], Any], repeat: int, number: int = 1) -> float:
    fn()
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        runs.append((time.perf_counter() - t) / number)
    return statistics.median(runs)


def parity() -> dict[str, Any]:
    from opacus.accountants.utils import get_noise_multiplier

    rows = []
    grid = itertools.product((0.001, 0.004, 0.01, 0.05), (0.6, 1.0, 2.0, 4.0), (100, 1000, 10000))
    for q, sigma, steps in grid:
        r = {
            "q": q,
            "sigma": sigma,
            "steps": steps,
            "rdp_rust": rdp_epsilon(q, sigma, steps, DELTA, backend="rust")[0],
            "rdp_numpy": rdp_epsilon(q, sigma, steps, DELTA, backend="python")[0],
            "rdp_opacus": opacus_eps("rdp", q, sigma, steps),
        }
        # PRV grids grow with sqrt(steps) and the privacy loss range; the largest
        # settings are skipped to bound memory (both libraries use the same grid).
        if steps == 10000 and (q > 0.01 or sigma < 1.0):
            rows.append(r)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r["prv"] = epsilon(q, sigma, steps, DELTA, accountant="prv")
                r["prv_opacus"] = opacus_eps("prv", q, sigma, steps)
            except (ValueError, RuntimeError) as e:
                r["prv_error"] = str(e)[:120]
        rows.append(r)

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-300)

    rr = [rel(r["rdp_rust"], r["rdp_opacus"]) for r in rows]
    rn = [rel(r["rdp_numpy"], r["rdp_opacus"]) for r in rows]
    pv = [rel(r["prv"], r["prv_opacus"]) for r in rows if "prv_opacus" in r]
    cal = []
    for target, q, epochs in itertools.product((1.0, 3.0, 8.0), (0.004, 0.02), (10, 30)):
        steps = int(epochs / q)
        ours = calibrate_noise(target, DELTA, q, steps, rtol=1e-6)
        ref = get_noise_multiplier(
            target_epsilon=target,
            target_delta=DELTA,
            sample_rate=q,
            steps=steps,
            accountant="rdp",
            epsilon_tolerance=1e-4,
        )
        cal.append(
            {
                "target": target,
                "q": q,
                "steps": steps,
                "sigma": ours,
                "sigma_opacus": float(ref),
                "eps_at_sigma": epsilon(q, ours, steps, DELTA),
                "eps_at_sigma_opacus": opacus_eps("rdp", q, float(ref), steps),
            }
        )
    return {
        "settings": len(rows),
        "rdp_rust_max_rel_diff": max(rr),
        "rdp_numpy_max_rel_diff": max(rn),
        "prv_settings": len(pv),
        "prv_max_rel_diff": max(pv) if pv else None,
        "prv_median_rel_diff": statistics.median(pv) if pv else None,
        "calibration_max_rel_sigma_diff": max(rel(c["sigma"], c["sigma_opacus"]) for c in cal),
        "rows": rows,
        "calibration": cal,
    }


def speed() -> dict[str, Any]:
    from opacus.accountants import RDPAccountant
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_curve as sk_roc

    q, sigma, steps = 0.004, 1.1, 7500
    out: dict[str, Any] = {"rdp_setting": {"q": q, "sigma": sigma, "steps": steps}}
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    orders = list(DEFAULT_ORDERS)
    out["rdp_epsilon_seconds"] = {
        "rust": timeit(lambda: rdp_epsilon(q, sigma, steps, DELTA, backend="rust"), 50, 20),
        "numpy": timeit(lambda: rdp_epsilon(q, sigma, steps, DELTA, backend="python"), 50, 5),
        "opacus": timeit(lambda: acc.get_epsilon(DELTA, alphas=orders), 50, 5),
    }
    out["calibrate_seconds"] = {
        "rust": timeit(lambda: calibrate_noise(3.0, DELTA, q, steps, backend="rust"), 10),
        "numpy": timeit(lambda: calibrate_noise(3.0, DELTA, q, steps, backend="python"), 5),
    }
    rng = np.random.default_rng(0)
    n = 100_000
    lab = rng.random(n) < 0.5
    s = rng.normal(size=n) + 0.3 * lab
    out["roc_n"] = n
    out["roc_summary_seconds"] = {
        "rust": timeit(lambda: metrics.summarize(s, lab, backend="rust"), 20),
        "numpy": timeit(lambda: metrics.summarize(s, lab, backend="python"), 20),
        "sklearn_roc_auc": timeit(lambda: sk_auc(*sk_roc(lab, s)[:2]), 20),
    }
    a_rust = metrics.summarize(s, lab, backend="rust").auc
    f, t, _ = sk_roc(lab, s, drop_intermediate=False)
    out["roc_auc_abs_diff_vs_sklearn"] = abs(a_rust - float(sk_auc(f, t)))
    out["bootstrap_reps"] = 1000
    out["bootstrap_seconds"] = {
        "rust": timeit(lambda: metrics.bootstrap(s, lab, reps=1000, backend="rust"), 3),
        "numpy": timeit(lambda: metrics.bootstrap(s, lab, reps=1000, backend="python"), 1),
    }
    return out


def main() -> int:
    t0 = time.time()
    res = {"parity": parity()}
    p = res["parity"]
    print(
        f"RDP max rel diff vs Opacus: rust {p['rdp_rust_max_rel_diff']:.2e}, "
        f"numpy {p['rdp_numpy_max_rel_diff']:.2e}; PRV max {p['prv_max_rel_diff']:.2e} "
        f"over {p['prv_settings']} settings; calibration {p['calibration_max_rel_sigma_diff']:.2e}",
        flush=True,
    )
    res["speed"] = speed()
    for k, v in res["speed"].items():
        print(k, v, flush=True)
    import opacus
    import sklearn

    res["versions"] = {"opacus": opacus.__version__, "scikit-learn": sklearn.__version__}
    res["machine"] = f"{platform.machine()} {platform.system()} {platform.python_version()}"
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
