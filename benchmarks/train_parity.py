"""Check that fine-tuning Pythia-70M gives the same model under two transformers releases.

A short deterministic fine-tune on the CPU with the canary benchmark's data pipeline and
training loop: a seeded subset of the training sequences, the same trainable layers,
optimizer and schedule as ``lm_canaries.py``. Each run records its per-epoch training loss
and per-sequence losses on held-out paragraphs, and saves the trained parameters; the
compare step reports the largest differences between two runs.

    uv run --no-sync python benchmarks/train_parity.py run --tag new
    uv run --no-sync --with transformers==4.57.6 python benchmarks/train_parity.py run --tag old
    uv run --no-sync python benchmarks/train_parity.py compare old new
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from lm_canaries import MAX_LEN, TRAIN_CHUNK, build_data

from privacy_attacks_lab import lm
from privacy_attacks_lab.data import DEFAULT_DATA_DIR

WORK = DEFAULT_DATA_DIR / "train-parity"
OUT = Path(__file__).resolve().parent / "results" / "train_parity.json"


def run(tag: str, n_train: int, epochs: int, n_heldout: int) -> None:
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    model, tok = lm.load_model(dev)
    d = build_data(tok)
    idx = np.sort(np.random.default_rng(0).choice(len(d["train"]), n_train, replace=False))
    train = d["train"].subset(idx)
    held = lm.encode(tok, d["nonmembers"][:n_heldout], MAX_LEN)
    trainable = lm.freeze_for_finetune(model)
    cfg = lm.LMTrainConfig(epochs=epochs, batch_size=16, lr=2e-4, chunk=TRAIN_CHUNK)
    t0 = time.perf_counter()
    res = lm.finetune(model, train, cfg)
    seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        held_loss = lm_loss_per_sequence(model, held)
    WORK.mkdir(parents=True, exist_ok=True)
    trained = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trained, WORK / f"{tag}.pt")
    info = {
        "transformers": version("transformers"),
        "torch": version("torch"),
        "train_sequences": n_train,
        "epochs": epochs,
        "steps": res.steps,
        "trainable_params": trainable,
        "train_loss_per_epoch": res.history,
        "heldout_sequences": n_heldout,
        "heldout_loss": held_loss.tolist(),
        "seconds": seconds,
    }
    (WORK / f"{tag}.json").write_text(json.dumps(info) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "heldout_loss"}), flush=True)


def lm_loss_per_sequence(model: torch.nn.Module, data: lm.Tokenized) -> np.ndarray:
    out = []
    for s in range(0, len(data), 8):
        out.append(lm.lm_loss(model, data.x[s : s + 8], data.y[s : s + 8]).numpy())
    return np.concatenate(out)


def compare(old: str, new: str) -> None:
    a = json.loads((WORK / f"{old}.json").read_text())
    b = json.loads((WORK / f"{new}.json").read_text())
    pa = torch.load(WORK / f"{old}.pt")
    pb = torch.load(WORK / f"{new}.pt")
    if set(pa) != set(pb):
        raise SystemExit(f"parameter names differ: {sorted(set(pa) ^ set(pb))[:5]}")
    weight_diff = max(float((pa[k] - pb[k]).abs().max()) for k in pa)
    update = max(float(pa[k].abs().max()) for k in pa)
    result = {
        "old": {k: a[k] for k in a if k != "heldout_loss"},
        "new": {k: b[k] for k in b if k != "heldout_loss"},
        "train_loss_max_abs_diff": float(
            np.max(np.abs(np.subtract(a["train_loss_per_epoch"], b["train_loss_per_epoch"])))
        ),
        "heldout_loss_max_abs_diff": float(
            np.max(np.abs(np.subtract(a["heldout_loss"], b["heldout_loss"])))
        ),
        "heldout_loss_mean": [float(np.mean(a["heldout_loss"])), float(np.mean(b["heldout_loss"]))],
        "trained_weights_max_abs_diff": weight_diff,
        "trained_weights_max_abs": update,
        "tensors": len(pa),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--n-train", type=int, default=256)
    r.add_argument("--epochs", type=int, default=2)
    r.add_argument("--n-heldout", type=int, default=64)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.tag, args.n_train, args.epochs, args.n_heldout)
    else:
        compare(args.old, args.new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
