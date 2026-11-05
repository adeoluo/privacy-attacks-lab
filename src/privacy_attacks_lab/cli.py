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
