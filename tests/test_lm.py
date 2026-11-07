"""Language model pieces on a tiny randomly initialised GPT-NeoX (no download)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from privacy_attacks_lab import dpsgd, lm  # noqa: E402

WORDS = ["<eos>", "The", " vault", " code", " of", " is", "."] + [f" {d}" for d in range(10)]
WORDS += [f" w{i}" for i in range(20)]


class FakeTokenizer:
    """Whitespace tokenizer over a fixed vocabulary; ``" 7"`` is one token."""

    eos_token_id = 0
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.vocab = {w: i for i, w in enumerate(WORDS)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for k, word in enumerate(text.replace(".", " .").split()):
            key = word if (k == 0 and word == "The") or word == "." else f" {word}"
            out.append(self.vocab[key])
        return out

    def __call__(self, texts: list[str], add_special_tokens: bool = False) -> dict[str, Any]:
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids: Any) -> str:
        return "".join(WORDS[int(i)] for i in ids)


def tiny_model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return transformers.GPTNeoXForCausalLM(cfg).eval()


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


def test_load_wikitext(tmp_path: Path) -> None:
    path = tmp_path / "w.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for split in ("train", "valid", "test"):
            zf.writestr(
                f"wikitext-2-raw/wiki.{split}.raw",
                f" = Title = \n\n First {split} paragraph . \n = = Sub = = \n Second . \n",
            )
    out = lm.load_wikitext(path)
    assert out["train"] == ["First train paragraph .", "Second ."]
    assert set(out) == {"train", "valid", "test"}


def test_encode_shifts_and_pads(tok: FakeTokenizer) -> None:
    data = lm.encode(tok, ["The vault code", "w1 w2 w3 w4 w5 w6"], max_len=4)
    assert data.x.tolist()[0] == [0, 1, 2, 3]
    assert data.y.tolist()[0] == [1, 2, 3, -100]
    # Truncated: inputs are eos + 3 tokens, targets the next 4.
    assert data.lengths.tolist() == [3, 4]
    assert data.y[1].tolist() == [tok.vocab[f" w{i}"] for i in (1, 2, 3, 4)]
    assert len(data.subset(np.array([1]))) == 1
    assert data.to(torch.device("cpu")).x.shape == (2, 4)


def test_lm_loss_matches_manual(tok: FakeTokenizer) -> None:
    model = tiny_model()
    data = lm.encode(tok, ["The vault code of", "w1 w2"], max_len=6)
    losses = lm.lm_loss(model, data.x, data.y)
    logits = model(input_ids=data.x).logits
    for i in range(2):
        k = int(data.lengths[i])
        ref = torch.nn.functional.cross_entropy(logits[i, :k], data.y[i, :k])
        torch.testing.assert_close(losses[i], ref)


def test_freeze_for_finetune_counts_block_linears() -> None:
    model = tiny_model()
    count = lm.freeze_for_finetune(model)
    # per block: qkv 32*96+96, dense 32*32+32, up 32*64+64, down 64*32+32
    per_block = 32 * 96 + 96 + 32 * 32 + 32 + 32 * 64 + 64 + 64 * 32 + 32
    assert count == 2 * per_block
    assert not model.gpt_neox.embed_in.weight.requires_grad
    assert not model.embed_out.weight.requires_grad


@pytest.mark.parametrize("seq_len", [5, 12, 40])
def test_ghost_clipping_on_sequences_matches_per_example_loop(
    tok: FakeTokenizer, seq_len: int
) -> None:
    model = tiny_model()
    lm.freeze_for_finetune(model)
    texts = ["The vault code of 1 2 3", "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11", "w3", "The 4 4"]
    data = lm.encode(tok, texts, max_len=seq_len)
    ghost, norms, _ = dpsgd.clipped_sum_ghost(model, data.x, data.y, 0.05, lm.lm_loss)
    hooks, losses = dpsgd.per_sample_grads(model, data.x, data.y, "hooks", lm.lm_loss)
    ref, ref_norms = dpsgd.clip_and_sum(hooks, 0.05)
    torch.testing.assert_close(norms, ref_norms, rtol=1e-4, atol=1e-6)
    for k in ref:
        torch.testing.assert_close(ghost[k], ref[k], rtol=1e-4, atol=1e-6)
    # And the hook gradients equal one backward pass per example.
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for i in range(len(texts)):
        model.zero_grad()
        lm.lm_loss(model, data.x[i : i + 1], data.y[i : i + 1]).sum().backward()
        for n, p in params.items():
            torch.testing.assert_close(hooks[n][i], p.grad, rtol=1e-4, atol=1e-6)
    assert losses.shape == (4,)


def test_ghost_falls_back_for_long_sequences() -> None:
    lin = torch.nn.Linear(2, 2)
    assert dpsgd._use_ghost(lin, torch.zeros(3, 2))
    assert not dpsgd._use_ghost(lin, torch.zeros(3, 5, 2))
    assert not dpsgd._use_ghost(torch.nn.Conv2d(1, 1, 1), torch.zeros(1, 1, 2, 2))


def _corpus(tok: FakeTokenizer) -> lm.Tokenized:
    rng = np.random.default_rng(0)
    texts = [" ".join(f"w{i}" for i in rng.integers(0, 20, size=8)) for _ in range(40)]
    texts += ["The vault code of is 3 1 4 1 5 9"] * 8
    return lm.encode(tok, texts, max_len=12)


def test_finetune_without_and_with_dp(tok: FakeTokenizer) -> None:
    data = _corpus(tok)
    model = tiny_model()
    lm.freeze_for_finetune(model)
    res = lm.finetune(model, data, lm.LMTrainConfig(epochs=3, batch_size=8, lr=3e-3, chunk=4))
    assert res.steps == 18
    assert res.history[-1] < res.history[0]
    assert res.epsilon == float("inf")
    model2 = tiny_model()
    lm.freeze_for_finetune(model2)
    dp = dpsgd.DPConfig(noise_multiplier=1.0, max_grad_norm=1.0)
    res2 = lm.finetune(
        model2, data, lm.LMTrainConfig(epochs=2, batch_size=12, lr=1e-3, chunk=4, dp=dp)
    )
    assert res2.steps == 8
    assert 0 < res2.epsilon < 100
    assert res2.sample_rate == pytest.approx(0.25)


class _InterruptError(Exception):
    pass


@pytest.mark.parametrize("private", [False, True])
def test_finetune_resumes_from_epoch_checkpoint(
    tok: FakeTokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private: bool
) -> None:
    data = _corpus(tok)
    dp = dpsgd.DPConfig(noise_multiplier=1.0, max_grad_norm=1.0) if private else None

    def cfg(path: Path | None) -> lm.LMTrainConfig:
        return lm.LMTrainConfig(
            epochs=3, batch_size=8, lr=3e-3, chunk=4, dp=dp, checkpoint=path, seed=5
        )

    full = tiny_model()
    lm.freeze_for_finetune(full)
    ref = lm.finetune(full, data, cfg(None))

    save = lm._checkpoint

    def save_then_stop(*args: Any) -> None:
        save(*args)
        if args[1] == 0:
            raise _InterruptError

    ckpt = tmp_path / "train.ckpt"
    first = tiny_model()
    lm.freeze_for_finetune(first)
    monkeypatch.setattr(lm, "_checkpoint", save_then_stop)
    with pytest.raises(_InterruptError):
        lm.finetune(first, data, cfg(ckpt))
    monkeypatch.setattr(lm, "_checkpoint", save)
    assert ckpt.exists()

    resumed = tiny_model()
    lm.freeze_for_finetune(resumed)
    res = lm.finetune(resumed, data, cfg(ckpt))
    assert res.steps == ref.steps
    np.testing.assert_allclose(res.history, ref.history, rtol=1e-5)
    if private:
        assert res.epsilon == pytest.approx(ref.epsilon)
    for (name, a), (_, b) in zip(full.named_parameters(), resumed.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)


def test_sequence_scores_and_zlib(tok: FakeTokenizer) -> None:
    model = tiny_model()
    data = _corpus(tok)
    sc = lm.sequence_scores(model, data, batch_size=7)
    assert sc["loss"].shape == (48,)
    np.testing.assert_allclose(
        sc["loss"], lm.lm_loss(model, data.x, data.y).detach().numpy(), rtol=1e-5
    )
    assert (sc["min_k"] >= sc["loss"] - 1e-6).all()
    z = lm.zlib_bits(["aaaaaaaaaaaaaaaa", "a quick brown fox jumps"])
    assert z[0] < z[1]


def test_candidate_log_probs_are_exact(tok: FakeTokenizer) -> None:
    model = tiny_model()
    prefix = "The vault code of is"
    lp = lm.candidate_log_probs(model, tok, prefix, length=3, batch_size=37)
    assert lp.shape == (1000,)
    # Brute force: score each candidate with one forward pass.
    pre = [0, *tok.encode(prefix)]
    for code in (0, 7, 123, 999):
        digits = [int(c) for c in f"{code:03d}"]
        ids = pre + [tok.vocab[f" {d}"] for d in digits]
        x = torch.tensor([ids[:-1]])
        logits = torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
        ref = sum(float(logits[len(pre) - 1 + j, ids[len(pre) + j]]) for j in range(3))
        assert lp[code] == pytest.approx(ref, abs=1e-4)
    # Probabilities of all candidates sum to the probability of "three digits".
    assert np.exp(lp).sum() <= 1.0 + 1e-6


def test_digit_tokens_must_be_single() -> None:
    class Splitter(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2]

    with pytest.raises(ValueError, match="splits"):
        lm.digit_token_ids(Splitter())


def test_generation_helpers(tok: FakeTokenizer) -> None:
    model = tiny_model()
    out = lm.greedy_digits(model, tok, "The vault code of is", length=3)
    assert isinstance(out, str)
    s = lm.sample(model, tok, n=5, length=6, batch_size=2, seed=1)
    assert s.shape == (5, 6)


def test_window_hashes_and_hits() -> None:
    corpus = [[1, 2, 3, 4, 5, 6], [7, 8, 9]]
    h = lm.window_hashes(corpus, 3)
    assert len(h) == 4 + 1
    samples = np.array([[0, 3, 4, 5, 0], [9, 9, 9, 9, 9], [6, 7, 8, 9, 1]])
    np.testing.assert_array_equal(lm.verbatim_hits(samples, h, 3), [True, False, True])
    assert len(lm.window_hashes([[1, 2]], 3)) == 0
