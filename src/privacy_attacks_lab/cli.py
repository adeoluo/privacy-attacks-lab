"""Command line interface: ``python -m privacy_attacks_lab <command>``.

Commands that need torch import it lazily, so the accountant and metric commands
work with only numpy and scipy installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from privacy_attacks_lab import __version__
from privacy_attacks_lab.accountants import MECHANISMS, calibrate_noise, epsilon, max_tpr


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    width = max(len(k) for k in payload)
    for key, value in payload.items():
        shown = f"{value:.6g}" if isinstance(value, float) else value
        print(f"{key:<{width}}  {shown}")


def _steps(args: argparse.Namespace) -> int:
    if args.steps is not None:
        return int(args.steps)
    if args.epochs is None:
        raise SystemExit("give --steps or --epochs")
    return round(float(args.epochs) / float(args.q))


def _add_dp_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--q", type=float, required=True, help="Poisson sampling rate (batch / n)")
    p.add_argument("--steps", type=int, help="number of noisy steps")
    p.add_argument("--epochs", type=float, help="epochs; steps = epochs / q")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--eps-error", type=float, default=0.01, help="PRV epsilon tolerance")
    p.add_argument("--json", action="store_true", help="print JSON")


def cmd_epsilon(args: argparse.Namespace) -> int:
    steps = _steps(args)
    payload: dict[str, Any] = {
        "q": args.q,
        "sigma": args.sigma,
        "steps": steps,
        "delta": args.delta,
    }
    for mech in args.accountant or list(MECHANISMS):
        payload[f"epsilon_{mech}"] = epsilon(
            args.q, args.sigma, steps, args.delta, accountant=mech, eps_error=args.eps_error
        )
    _emit(payload, args.json)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    steps = _steps(args)
    sigma = calibrate_noise(
        args.target_epsilon,
        args.delta,
        args.q,
        steps,
        accountant=args.accountant,
        eps_error=args.eps_error,
    )
    payload = {
        "target_epsilon": args.target_epsilon,
        "accountant": args.accountant,
        "sigma": sigma,
        "steps": steps,
        "q": args.q,
        "delta": args.delta,
    }
    _emit(payload, args.json)
    return 0


def cmd_max_tpr(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"epsilon": args.epsilon, "delta": args.delta}
    for fpr in args.fpr:
        payload[f"max_tpr@{fpr:g}"] = float(max_tpr(fpr, args.epsilon, args.delta))
    _emit(payload, args.json)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover
    from privacy_attacks_lab.service import serve

    serve(args.host, args.port)
    return 0


REPORT_METRICS = ("auc", "accuracy", "tpr@0.001", "tpr@0.01", "tpr@0.1")


def cmd_report(args: argparse.Namespace) -> int:
    """Print the attack table of a ``benchmarks/results/mia_<preset>.json`` file."""
    with open(args.path) as fh:
        data = json.load(fh)
    table = data.get("table")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{args.path} has no attack table")
    rows = {k: v for k, v in table.items() if args.all or not k.endswith(("-ref", "-ref-fixed"))}
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    models = data.get("num_models", "?")
    targets = len(data.get("targets", [])) or "?"
    print(f"{data.get('preset', '?')}: {models} models, {targets} targets (mean over targets)")
    names = ("AUC", "bal acc", "TPR@0.1%", "TPR@1%", "TPR@10%")
    width = max(len(k) for k in rows)
    print(f"{'attack':<{width}}  " + "  ".join(f"{n:>8}" for n in names))
    order = sorted(rows, key=lambda k: rows[k].get(args.sort, 0.0))
    for name in order:
        vals = "  ".join(f"{100 * rows[name][m]:>7.2f}%" for m in REPORT_METRICS)
        print(f"{name:<{width}}  {vals}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacy-attacks-lab",
        description="Membership inference, DP-SGD accounting and canary exposure.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("epsilon", help="privacy spent by DP-SGD (RDP, PRV and GDP)")
    _add_dp_args(p)
    p.add_argument("--sigma", type=float, required=True, help="noise multiplier")
    p.add_argument("--accountant", action="append", choices=MECHANISMS)
    p.set_defaults(func=cmd_epsilon)

    p = sub.add_parser("calibrate", help="smallest noise multiplier for a target epsilon")
    _add_dp_args(p)
    p.add_argument("--target-epsilon", type=float, required=True)
    p.add_argument("--accountant", choices=MECHANISMS, default="rdp")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("max-tpr", help="highest attack TPR at a given FPR allowed by (eps, delta)")
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--fpr", type=float, action="append", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_max_tpr, fpr_default=(0.001, 0.01, 0.1))

    p = sub.add_parser("serve", help="HTTP service (accounting, ROC metrics, exposure)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="print the attack table of a mia_<preset>.json result")
    p.add_argument("path")
    p.add_argument("--sort", choices=REPORT_METRICS, default="tpr@0.001")
    p.add_argument("--all", action="store_true", help="include the reference LiRA scoring")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    for register in _EXTRA_COMMANDS:
        register(sub)
    return parser


_EXTRA_COMMANDS: list[Callable[[Any], None]] = []


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "fpr", 1) is None:
        args.fpr = list(args.fpr_default)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
