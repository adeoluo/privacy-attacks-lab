"""Canary exposure, extraction and membership inference on a fine-tuned language model.

Usage:
    uv run --no-sync python benchmarks/lm_canaries.py [--models base ft dp8] [--device cpu]
        [--out results.json] [--save weights-dir]

Pythia-70M (deduplicated) is fine-tuned on a seeded half of the WikiText-2 training
paragraphs plus 24 fictional canaries inserted 1, 2, 4, 8, 16 or 32 times (four
canaries per level). The other half of the paragraphs never trains the model and
serves as the non-members. For each model the script measures

- canary exposure, exactly, by enumerating all 10^5 five-digit secrets, and the
  skew-normal estimate from 5,000 sampled candidates (Carlini et al. 2019);
- extraction: whether greedy decoding after the canary prefix emits the secret;
- sequence membership inference on 4,000 members and 4,000 non-members: loss,
  Min-K% (k = 20%), zlib ratio and reference-calibrated loss (fine-tuned minus
  base model loss, the reference attack of Carlini et al. 2021 / Mireshghallah et al. 2022);
- training-data extraction from 1,000 unconditional samples (96 tokens, top-k 40):
  the samples ranked by each membership signal and checked for verbatim 32-token
  spans of member and of non-member paragraphs (the non-member count controls
  for text the base model already knew from pre-training);
- utility: perplexity on the WikiText-2 validation paragraphs.

Results go to benchmarks/results/lm_canaries.json (merged per model, so models
can be run one at a time). Fine-tuning saves its state after every epoch, finished
canaries are kept in a side file, and finished weights are reused, so an
interrupted run continues where it stopped.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from privacy_attacks_lab import canary, lm, metrics
from privacy_attacks_lab.accountants import calibrate_noise
from privacy_attacks_lab.data import DEFAULT_DATA_DIR
from privacy_attacks_lab.dpsgd import DPConfig
from privacy_attacks_lab.training import get_device

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "lm_canaries.json"
SAVE = DEFAULT_DATA_DIR / "lm"
MAX_LEN = 128
EPOCHS = 4
BATCH = 16
DP_BATCH = 256
DELTA = 1e-5
# Candidates per forward pass in the exact exposure enumeration (bounds peak memory; 512
# ran out of the 2.1 GB MPS cap right after fine-tuning).
ENUM_BATCH = 256
# Sequences per forward pass when scoring (4 x 128 tokens x 50k vocabulary logits = 0.1 GB).
SCORE_BATCH = 4
# Examples per forward/backward pass in fine-tuning (gradients are accumulated per batch).
TRAIN_CHUNK = 4
# Five-digit secrets (10^5 candidates): exact enumeration takes about 30 s per canary on
# the shared M1; six digits would take about 5 minutes per canary.
SECRET_DIGITS = 5

MODELS: dict[str, dict[str, Any]] = {
    "base": {"train": False},
    "ft": {"train": True, "eps": None, "lr": 2e-4},
    "dp8": {"train": True, "eps": 8.0, "lr": 1e-3},
    "dp1": {"train": True, "eps": 1.0, "lr": 1e-3},
}


def _free(model: Any) -> None:
    """Return cached MPS blocks to the system so peak memory stays near the working set."""
    if next(model.parameters()).device.type == "mps":
        torch.mps.empty_cache()


def _mps_mb() -> str:
    if not torch.backends.mps.is_available():
        return "cpu"
    alloc = torch.mps.current_allocated_memory() / 2**20
    driver = torch.mps.driver_allocated_memory() / 2**20
    return f"MPS {alloc:.0f} MB allocated, {driver:.0f} MB driver"


def build_data(tok: Any) -> dict[str, Any]:
    paras = lm.load_wikitext(DEFAULT_DATA_DIR / lm.WIKITEXT_FILE)
    train = paras["train"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train))
    half = len(train) // 2
    members = [train[i] for i in np.sort(perm[:half])]
    nonmembers = [train[i] for i in np.sort(perm[half:])]
    canaries = canary.make_canaries(length=SECRET_DIGITS)
    docs, which = canary.insert(members, canaries, seed=0)
    return {
        "canaries": canaries,
        "train": lm.encode(tok, docs, MAX_LEN),
        "train_which": which,
        "members": members,
        "nonmembers": nonmembers,
        "valid": paras["valid"],
    }


def mia_block(scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    out = {}
    for name, s in scores.items():
        ci = metrics.bootstrap(s, labels, reps=500)
        out[name] = {k: iv.to_dict() for k, iv in ci.items()}
    return out


def evaluate(
    name: str, model: Any, tok: Any, d: dict[str, Any], base_cache: dict[str, Any]
) -> dict[str, Any]:
    res: dict[str, Any] = {}
    # Canary exposure and extraction.
    rows = []
    t0 = time.perf_counter()
    # Finished canaries are kept in a side file so an interrupted run resumes.
    SAVE.mkdir(parents=True, exist_ok=True)
    partial = SAVE / f"{name}_canaries.jsonl"
    done = (
        {json.loads(x)["name"]: json.loads(x) for x in partial.read_text().splitlines() if x}
        if partial.exists()
        else {}
    )
    for k, c in enumerate(d["canaries"]):
        if c.name in done:
            rows.append(done[c.name])
            continue
        rng = np.random.default_rng([1, k])
        lp = lm.candidate_log_probs(
            model, tok, c.prefix.rstrip(" "), length=len(c.digits), batch_size=ENUM_BATCH
        )
        _free(model)
        rank = canary.rank_of(lp, c.index)
        sample_idx = rng.choice(len(lp), 5000, replace=False)
        est = canary.exposure_from_sample(lp[c.index], lp[sample_idx], len(lp))
        greedy = lm.greedy_digits(model, tok, c.prefix.rstrip(" "), len(c.digits))
        rows.append(
            {
                "name": c.name,
                "repeats": c.repeats,
                "rank": rank,
                "exposure": canary.exposure(rank, len(lp)),
                "exposure_skewnorm": est["skewnorm"],
                "exposure_sampled": est["empirical"],
                "log_prob": float(lp[c.index]),
                "greedy": greedy.strip(),
                "extracted": greedy.strip() == c.secret,
            }
        )
        with partial.open("a") as fh:
            fh.write(json.dumps(rows[-1]) + "\n")
        print(
            f"  {name} {c.name:<18} r={c.repeats:<2} rank={rank:<7} exp={rows[-1]['exposure']:.2f} "
            f"greedy={greedy.strip()!r}",
            flush=True,
        )
    res["canaries"] = rows
    res["canary_seconds"] = time.perf_counter() - t0
    gc.collect()
    _free(model)
    print(f"  {name} canaries done, {_mps_mb()}", flush=True)

    # Sequence membership inference.
    rng = np.random.default_rng(2)
    mem = [d["members"][i] for i in rng.choice(len(d["members"]), 4000, replace=False)]
    non = [d["nonmembers"][i] for i in rng.choice(len(d["nonmembers"]), 4000, replace=False)]
    texts = mem + non
    labels = np.r_[np.ones(4000, bool), np.zeros(4000, bool)]
    enc = lm.encode(tok, texts, MAX_LEN)
    sc = lm.sequence_scores(model, enc, batch_size=SCORE_BATCH)
    if "mia" not in base_cache:
        base_cache["mia"] = sc
    base = base_cache["mia"]
    ntok = enc.lengths.numpy().astype(np.float64)
    truncated = [tok.decode(enc.y[i, : int(enc.lengths[i])].tolist()) for i in range(len(texts))]
    zl = lm.zlib_bits(truncated)
    scores = {
        "loss": -sc["loss"],
        "min_k": -sc["min_k"],
        "zlib": -(sc["loss"] * ntok) / zl,
        "reference": -(sc["loss"] - base["loss"]),
    }
    res["mia"] = mia_block(scores, labels)
    for k, v in res["mia"].items():
        print(
            f"  {name} mia {k:<10} auc={v['auc']['estimate']:.4f} "
            f"tpr@1%={v['tpr@0.01']['estimate']:.4f}",
            flush=True,
        )

    # Utility.
    val = lm.encode(tok, d["valid"], MAX_LEN)
    vs = lm.sequence_scores(model, val, batch_size=SCORE_BATCH)
    vt = val.lengths.numpy()
    res["valid_ppl"] = float(math.exp((vs["loss"] * vt).sum() / vt.sum()))
    print(f"  {name} valid ppl={res['valid_ppl']:.2f}", flush=True)
    return res


def extraction(name: str, model: Any, base: Any, tok: Any, d: dict[str, Any]) -> dict[str, Any]:
    width = 32
    t0 = time.perf_counter()
    samples = lm.sample(model, tok, 1000, length=96, seed=3)
    mem_ids = tok([*d["members"]], add_special_tokens=False)["input_ids"]
    non_ids = tok([*d["nonmembers"]], add_special_tokens=False)["input_ids"]
    hm = lm.verbatim_hits(samples, lm.window_hashes(mem_ids, width), width)
    hn = lm.verbatim_hits(samples, lm.window_hashes(non_ids, width), width)
    eos = int(tok.eos_token_id)
    x = torch.from_numpy(np.c_[np.full(len(samples), eos), samples[:, :-1]])
    data = lm.Tokenized(
        x, torch.from_numpy(samples.astype(np.int64)), torch.full((len(samples),), 96)
    )
    ft = lm.sequence_scores(model, data, batch_size=SCORE_BATCH)["loss"]
    ref = lm.sequence_scores(base, data, batch_size=SCORE_BATCH)["loss"]
    zl = lm.zlib_bits([tok.decode(s.tolist()) for s in samples])
    rank_by = {"perplexity": ft, "reference_ratio": ft / ref, "zlib_ratio": ft * 96 / zl}
    out: dict[str, Any] = {
        "samples": len(samples),
        "width": width,
        "member_hits": int(hm.sum()),
        "nonmember_hits": int(hn.sum()),
        "seconds": time.perf_counter() - t0,
    }
    for k, v in rank_by.items():
        top = np.argsort(v)[:100]
        out[f"top100_{k}"] = {
            "member_hits": int(hm[top].sum()),
            "nonmember_hits": int(hn[top].sum()),
        }
    print(f"  {name} extraction {json.dumps(out)}", flush=True)
    return out


def main() -> int:
    global OUT, SAVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--device", default="auto", help="auto (MPS when available), mps or cpu")
    ap.add_argument("--out", type=Path, default=OUT, help="results file, merged per model")
    ap.add_argument(
        "--save", type=Path, default=SAVE, help="fine-tuned weights and per-canary partial results"
    )
    args = ap.parse_args()
    OUT, SAVE = args.out, args.save
    dev = get_device(args.device)
    base, tok = lm.load_model(dev)
    d = build_data(tok)
    print(
        f"train examples {len(d['train'])}, members {len(d['members'])}, "
        f"non-members {len(d['nonmembers'])}",
        flush=True,
    )
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["setup"] = {
        "model": lm.MODEL_ID,
        "max_len": MAX_LEN,
        "epochs": EPOCHS,
        "batch": BATCH,
        "dp_expected_batch": DP_BATCH,
        "delta": DELTA,
        "train_examples": len(d["train"]),
        "members": len(d["members"]),
        "nonmembers": len(d["nonmembers"]),
        "canaries": len(d["canaries"]),
        "canary_space": 10**SECRET_DIGITS,
        "trainable_params": None,
        "machine": f"{platform.machine()} {dev.type}",
    }
    cache: dict[str, Any] = {}
    # The base model's sequence scores are the reference for the calibrated attack.
    base_res = evaluate("base", base, tok, d, cache) if "base" in args.models else None
    if base_res is not None:
        results["base"] = base_res
        OUT.write_text(json.dumps(results, indent=1) + "\n")
        base_res["extraction"] = extraction("base", base, base, tok, d)
        results["base"] = base_res
        OUT.write_text(json.dumps(results, indent=1) + "\n")
    elif "mia" not in cache:
        rng = np.random.default_rng(2)
        mem = [d["members"][i] for i in rng.choice(len(d["members"]), 4000, replace=False)]
        non = [d["nonmembers"][i] for i in rng.choice(len(d["nonmembers"]), 4000, replace=False)]
        cache["mia"] = lm.sequence_scores(
            base, lm.encode(tok, mem + non, MAX_LEN), batch_size=SCORE_BATCH
        )
    for name in args.models:
        spec = MODELS[name]
        if not spec["train"]:
            continue
        model, _ = lm.load_model(dev)
        results["setup"]["trainable_params"] = lm.freeze_for_finetune(model)
        dp = None
        sigma = None
        if spec["eps"] is not None:
            q = DP_BATCH / len(d["train"])
            steps = EPOCHS * round(1 / q)
            sigma = calibrate_noise(spec["eps"], DELTA, q, steps)
            dp = DPConfig(sigma, max_grad_norm=1.0, delta=DELTA)
        cfg = lm.LMTrainConfig(
            epochs=EPOCHS,
            batch_size=DP_BATCH if dp else BATCH,
            lr=spec["lr"],
            dp=dp,
            chunk=TRAIN_CHUNK,
            checkpoint=SAVE / f"{name}_train.ckpt",
        )
        SAVE.mkdir(parents=True, exist_ok=True)
        weights, info = SAVE / f"{name}.pt", SAVE / f"{name}_train.json"
        if weights.exists() and info.exists():
            # Resume: reuse the fine-tuned weights of an interrupted run.
            lm.load_trainable(model, torch.load(weights, map_location=dev))
            train_info = json.loads(info.read_text())
            print(f"loaded fine-tuned {name} from {weights}", flush=True)
        else:
            print(f"fine-tuning {name} (sigma={sigma})", flush=True)
            tr = lm.finetune(model, d["train"], cfg)
            print(
                f"  {name} trained in {tr.seconds:.0f}s, loss {tr.history}, eps={tr.epsilon}",
                flush=True,
            )
            trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(trained, weights)
            train_info = {
                "seconds": tr.seconds,
                "steps": tr.steps,
                "history": tr.history,
                "epsilon": None if math.isinf(tr.epsilon) else tr.epsilon,
                "sigma": sigma,
                "lr": spec["lr"],
            }
            info.write_text(json.dumps(train_info) + "\n")
            (SAVE / f"{name}_train.ckpt").unlink(missing_ok=True)
            gc.collect()
            _free(model)
        r = evaluate(name, model, tok, d, cache)
        r["train"] = train_info
        r["extraction"] = extraction(name, model, base, tok, d)
        results[name] = r
        OUT.write_text(json.dumps(results, indent=1) + "\n")
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
