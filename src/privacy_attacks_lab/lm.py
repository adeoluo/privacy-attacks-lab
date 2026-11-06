"""A small causal language model fine-tuned on text with inserted canaries.

The experiments follow two papers:

- *The Secret Sharer* (Carlini et al. 2019): insert fictional secrets into the
  fine-tuning text and measure their exposure (:mod:`privacy_attacks_lab.canary`)
  with and without DP-SGD.
- *Extracting Training Data from Large Language Models* (Carlini et al. 2021):
  sample from the fine-tuned model, rank samples by membership signals (the
  model's perplexity, its ratio to a reference model's, and the zlib ratio) and
  count samples that reproduce fine-tuning text verbatim.

The model is ``EleutherAI/pythia-70m-deduped`` (70M parameters, no token needed).
Fine-tuning updates the 24 linear layers of its six transformer blocks (18.9M
parameters) and keeps the embeddings, unembedding and layer norms frozen, so the
same setting runs with and without DP-SGD (ghost clipping handles the linear
layers on sequences; see :func:`privacy_attacks_lab.dpsgd.clipped_sum_ghost`).

Every example is one paragraph of WikiText-2 (Merity et al. 2016), truncated to
``max_len`` tokens and preceded by the end-of-text token, so the first word is
predicted too. Canaries are examples of their own.
"""

from __future__ import annotations

import math
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from privacy_attacks_lab.accountants import Accountant
from privacy_attacks_lab.dpsgd import DPConfig, noisy_gradient, poisson_batches
from privacy_attacks_lab.training import seed_everything

MODEL_ID = "EleutherAI/pythia-70m-deduped"
WIKITEXT_FILE = "wikitext-2-raw-v1.zip"
IGNORE = -100


def load_wikitext(path: Path) -> dict[str, list[str]]:
    """Paragraphs (non-empty, non-heading lines) of each WikiText-2 split."""
    out: dict[str, list[str]] = {}
    with zipfile.ZipFile(path) as zf:
        for split in ("train", "valid", "test"):
            raw = zf.read(f"wikitext-2-raw/wiki.{split}.raw").decode("utf-8")
            lines = (ln.strip() for ln in raw.splitlines())
            out[split] = [ln for ln in lines if ln and not ln.startswith("=")]
    return out


@dataclass
class Tokenized:
    """Right-padded examples: ``x`` inputs and ``y`` next-token targets (``-100`` = padding)."""

    x: torch.Tensor
    y: torch.Tensor
    lengths: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, idx: np.ndarray | torch.Tensor) -> Tokenized:
        i = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return Tokenized(self.x[i], self.y[i], self.lengths[i])

    def to(self, device: torch.device) -> Tokenized:
        return Tokenized(self.x.to(device), self.y.to(device), self.lengths.to(device))


def encode(tokenizer: Any, texts: list[str], max_len: int = 128) -> Tokenized:
    """``[eos] + tokens`` truncated to ``max_len + 1`` tokens and split into inputs and targets."""
    eos = int(tokenizer.eos_token_id)
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    n = len(ids)
    x = torch.full((n, max_len), eos, dtype=torch.long)
    y = torch.full((n, max_len), IGNORE, dtype=torch.long)
    lengths = torch.zeros(n, dtype=torch.long)
    for i, seq in enumerate(ids):
        full = [eos, *seq[:max_len]]
        k = len(full) - 1
        x[i, : k + 1 if k < max_len else max_len] = torch.tensor(full[:max_len])
        y[i, :k] = torch.tensor(full[1 : k + 1])
        lengths[i] = k
    return Tokenized(x, y, lengths)


def token_log_probs(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``log p(y_t | x_<=t)`` per position; 0 where ``y`` is padding."""
    logits = model(input_ids=x).logits.float()
    lp = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.clamp(min=0).reshape(-1),
        reduction="none",
    ).view(y.shape)
    return lp * (y != IGNORE)


def lm_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example mean next-token cross-entropy (the DP-SGD unit is one example)."""
    lp = token_log_probs(model, x, y)
    count = (y != IGNORE).sum(dim=1).clamp(min=1)
    return -lp.sum(dim=1) / count


def load_model(device: torch.device, model_id: str = MODEL_ID) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return model.to(device).eval(), tok  # type: ignore[arg-type]


def freeze_for_finetune(model: nn.Module) -> int:
    """Train only the linear layers inside the transformer blocks; return their size."""
    for p in model.parameters():
        p.requires_grad_(False)
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in f".{name}":
            for p in mod.parameters():
                p.requires_grad_(True)
                count += p.numel()
    return count


@dataclass
class LMTrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 0
    chunk: int = 8
    dp: DPConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LMTrainResult:
    seconds: float
    steps: int
    history: list[float]
    epsilon: float = math.inf
    sample_rate: float = 0.0


def finetune(model: nn.Module, data: Tokenized, cfg: LMTrainConfig) -> LMTrainResult:
    """Fine-tune in place with AdamW, without DP (shuffled batches) or with DP-SGD
    (Poisson batches of expected size ``batch_size``, per-example clipping and noise)."""
    dev = next(model.parameters()).device
    gen = seed_everything(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(data)
    model.train()
    history: list[float] = []
    start = time.perf_counter()
    steps = 0
    if cfg.dp is None:
        per_epoch = math.ceil(n / cfg.batch_size)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / (cfg.epochs * per_epoch))
        )
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen)
            total, seen = 0.0, 0
            for s in range(0, n, cfg.batch_size):
                idx = perm[s : s + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                for c in range(0, len(idx), cfg.chunk):
                    sub = idx[c : c + cfg.chunk]
                    loss = lm_loss(model, data.x[sub].to(dev), data.y[sub].to(dev))
                    (loss.sum() / len(idx)).backward()
                    total += float(loss.detach().sum())
                opt.step()
                sched.step()
                seen += len(idx)
                steps += 1
            history.append(total / seen)
        _sync(dev)
        return LMTrainResult(time.perf_counter() - start, steps, history)

    dp = cfg.dp
    q = min(1.0, cfg.batch_size / n)
    per_epoch = max(1, round(1 / q))
    acc = Accountant()
    total_steps = cfg.epochs * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total_steps)
    )
    for _ in range(cfg.epochs):
        total, seen = 0.0, 0
        for idx in poisson_batches(n, q, per_epoch, gen):
            acc.step(dp.noise_multiplier, q)
            grads, loss_sum = noisy_gradient(
                model,
                data.x[idx].to(dev),
                data.y[idx].to(dev),
                dp,
                q * n,
                gen,
                cfg.chunk,
                "ghost",
                lm_loss,
            )
            opt.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.grad = grads[name]
            opt.step()
            sched.step()
            total += loss_sum
            seen += len(idx)
            steps += 1
        history.append(total / max(seen, 1))
    _sync(dev)
    eps = acc.epsilon(dp.delta) if dp.noise_multiplier > 0 else math.inf
    return LMTrainResult(time.perf_counter() - start, steps, history, eps, q)


def _sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def sequence_scores(
    model: nn.Module, data: Tokenized, batch_size: int = 32, min_k: float = 0.2
) -> dict[str, np.ndarray]:
    """Per-example membership signals of a language model.

    ``loss``: mean token NLL. ``min_k``: mean NLL of the ``min_k`` fraction of
    tokens with the lowest probability (Min-K% Prob, Shi et al. 2024).
    """
    model.eval()
    dev = next(model.parameters()).device
    losses, mink = [], []
    for s in range(0, len(data), batch_size):
        x = data.x[s : s + batch_size].to(dev)
        y = data.y[s : s + batch_size].to(dev)
        lp = token_log_probs(model, x, y).cpu().numpy()
        mask = (y != IGNORE).cpu().numpy()
        for row, m in zip(lp, mask, strict=True):
            v = row[m]
            losses.append(-v.mean() if len(v) else 0.0)
            k = max(1, math.ceil(min_k * len(v)))
            mink.append(-np.sort(v)[:k].mean() if len(v) else 0.0)
    return {"loss": np.asarray(losses), "min_k": np.asarray(mink)}


def zlib_bits(texts: list[str]) -> np.ndarray:
    """Compressed length in bits, the zlib entropy of Carlini et al. (2021)."""
    return np.array([8 * len(zlib.compress(t.encode("utf-8"))) for t in texts], dtype=np.float64)


def digit_token_ids(tokenizer: Any) -> list[int]:
    """Token ids of ``" 0"`` to ``" 9"``; each must be a single token."""
    ids = []
    for d in range(10):
        enc = tokenizer.encode(f" {d}", add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError("the tokenizer splits a spaced digit into several tokens")
        ids.append(int(enc[0]))
    return ids


@torch.no_grad()
def candidate_log_probs(
    model: nn.Module, tokenizer: Any, prefix: str, length: int = 6, batch_size: int = 4096
) -> np.ndarray:
    """``log p(secret | prefix)`` for every one of the ``10^length`` digit strings.

    Exact enumeration: the model is run on all ``10^d`` partial secrets at every
    depth ``d < length`` and the log-probabilities of the next digit are added,
    so entry ``i`` is the log-probability of the digits of ``i`` (zero-padded)
    under the full vocabulary softmax.
    """
    model.eval()
    dev = next(model.parameters()).device
    eos = int(tokenizer.eos_token_id)
    pre = [eos, *tokenizer.encode(prefix, add_special_tokens=False)]
    digits = torch.tensor(digit_token_ids(tokenizer), device=dev)
    pre_t = torch.tensor(pre, device=dev)
    # Accumulated in float64 on the CPU (MPS has no float64).
    scores = torch.zeros(1, dtype=torch.float64)
    for depth in range(length):
        count = 10**depth
        nxt = torch.empty(count, 10, dtype=torch.float64)
        for s in range(0, count, batch_size):
            e = min(count, s + batch_size)
            codes = torch.arange(s, e, device=dev)
            cols = []
            for p in range(depth - 1, -1, -1):
                cols.append(digits[(codes // 10**p) % 10])
            tail = (
                torch.stack(cols, dim=1)
                if cols
                else torch.zeros(e - s, 0, dtype=torch.long, device=dev)
            )
            x = torch.cat([pre_t.expand(e - s, -1), tail], dim=1)
            logits = model(input_ids=x, logits_to_keep=1).logits[:, -1].float()
            nxt[s:e] = torch.log_softmax(logits, dim=-1)[:, digits].cpu().double()
        scores = (scores[:, None] + nxt).reshape(-1)
    return scores.numpy()


@torch.no_grad()
def greedy_digits(model: nn.Module, tokenizer: Any, prefix: str, length: int = 6) -> str:
    """Greedy (unconstrained) continuation of ``prefix`` for ``length`` tokens, decoded."""
    model.eval()
    dev = next(model.parameters()).device
    ids = [int(tokenizer.eos_token_id), *tokenizer.encode(prefix, add_special_tokens=False)]
    x = torch.tensor([ids], device=dev)
    out = model.generate(  # type: ignore[operator]
        x,
        max_new_tokens=length,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(x),
    )
    return str(tokenizer.decode(out[0, len(ids) :]))


@torch.no_grad()
def sample(
    model: nn.Module,
    tokenizer: Any,
    n: int,
    length: int = 96,
    top_k: int = 40,
    batch_size: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """``n`` unconditional samples of ``length`` tokens (top-k sampling from end-of-text)."""
    model.eval()
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    eos = int(tokenizer.eos_token_id)
    rows = []
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        x = torch.full((b, 1), eos, device=dev)
        out = model.generate(  # type: ignore[operator]
            x,
            max_new_tokens=length,
            min_new_tokens=length,
            do_sample=True,
            top_k=top_k,
            pad_token_id=eos,
            attention_mask=torch.ones_like(x),
        )
        rows.append(out[:, 1:].cpu().numpy())
    return np.concatenate(rows)


def window_hashes(tokens: list[list[int]] | np.ndarray, width: int) -> np.ndarray:
    """Sorted unique 64-bit polynomial hashes of every ``width``-token window."""
    base = np.uint64(1_000_003)
    out = []
    with np.errstate(over="ignore"):
        powers = base ** np.arange(width - 1, -1, -1, dtype=np.uint64)
        for seq in tokens:
            a = np.asarray(seq, dtype=np.uint64)
            if len(a) < width:
                continue
            win = np.lib.stride_tricks.sliding_window_view(a, width)
            out.append((win * powers).sum(axis=1))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.uint64)


def verbatim_hits(samples: np.ndarray, corpus_hashes: np.ndarray, width: int) -> np.ndarray:
    """For each sample, whether any ``width``-token window occurs in the corpus."""
    hits = np.zeros(len(samples), dtype=bool)
    for i, seq in enumerate(samples):
        h = window_hashes([seq], width)
        hits[i] = bool(np.isin(h, corpus_hashes, assume_unique=True).any())
    return hits
