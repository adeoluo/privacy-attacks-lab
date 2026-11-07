"""Canaries and exposure (Carlini, Liu, Erlingsson, Kos and Song, USENIX Security 2019).

A canary is a fictional secret, such as ``"the vault code of Orrin Vale is 4 8 1 9 0 3"``,
inserted ``r`` times into a model's training text. Its *exposure* measures how much
more likely the model finds the inserted secret than the other candidates of the
same format. With ``R`` the candidate space (``10^n`` strings of ``n`` digits) and
``rank`` the canary's position when all candidates are sorted by the model's
log-perplexity (1 = most likely),

    exposure = log2 |R| - log2 rank.

Exposure is 0 for the least likely candidate, about 1 for a typical one (a random
secret ranks in the middle) and ``log2 |R|`` (16.6 for 10^5 candidates) when the
canary is the model's single most likely completion, at which point greedy
decoding extracts it.

Ranks are computed exactly by enumerating every candidate
(:func:`privacy_attacks_lab.lm.candidate_log_probs`). The paper's cheaper
estimate fits a skew-normal distribution to the log-perplexities of a random
sample of candidates and reads the canary's tail probability from it;
:func:`exposure_from_sample` implements it so the two can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: Fictional names; any resemblance to real people is unintended.
NAMES: tuple[str, ...] = (
    "Orrin Vale",
    "Maelis Quarry",
    "Teodor Brask",
    "Ilse Varnum",
    "Caspian Mott",
    "Wren Halloway",
    "Dov Esterly",
    "Pella Gorse",
    "Anouk Treadwell",
    "Sefton Pyle",
    "Liesl Marrow",
    "Corvin Ashdale",
    "Juno Penhallow",
    "Ravi Thistlewood",
    "Odile Stannard",
    "Bastien Crewe",
    "Hesper Lindqvist",
    "Tobiah Fenwick",
    "Marisol Kettering",
    "Edda Brightwater",
    "Quill Harrowgate",
    "Sabine Orme",
    "Fenn Calloway",
    "Ysolde Marchbank",
)

TEMPLATE = "The vault code of {name} is {secret}."


@dataclass(frozen=True)
class Canary:
    """One inserted secret: ``prefix + secret`` is the text, ``repeats`` the insertion count."""

    name: str
    digits: tuple[int, ...]
    repeats: int

    @property
    def secret(self) -> str:
        """Digits separated by spaces, so every digit is a single token."""
        return " ".join(str(d) for d in self.digits)

    @property
    def prefix(self) -> str:
        return TEMPLATE.split("{secret}")[0].format(name=self.name)

    @property
    def text(self) -> str:
        return TEMPLATE.format(name=self.name, secret=self.secret)

    @property
    def index(self) -> int:
        """Position of the secret in the lexicographic enumeration of candidates."""
        out = 0
        for d in self.digits:
            out = out * 10 + d
        return out


def make_canaries(
    repeats: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    per_level: int = 4,
    length: int = 6,
    seed: int = 0,
) -> list[Canary]:
    """``per_level`` canaries for every repetition count, each with its own name and secret."""
    need = len(repeats) * per_level
    if need > len(NAMES):
        raise ValueError(f"at most {len(NAMES)} canaries (one per fictional name)")
    rng = np.random.default_rng(seed)
    out = []
    for k, r in enumerate(np.repeat(repeats, per_level)):
        digits = tuple(int(d) for d in rng.integers(0, 10, size=length))
        out.append(Canary(NAMES[k], digits, int(r)))
    return out


def insert(
    documents: list[str], canaries: list[Canary], seed: int = 0
) -> tuple[list[str], np.ndarray]:
    """Insert each canary ``repeats`` times as its own document at random positions.

    Returns the new document list and, for each document, the index of the canary
    it holds (``-1`` for original text).
    """
    rng = np.random.default_rng(seed)
    extra = [(c.text, k) for k, c in enumerate(canaries) for _ in range(c.repeats)]
    docs = [(d, -1) for d in documents] + extra
    order = rng.permutation(len(docs))
    return [docs[i][0] for i in order], np.array([docs[i][1] for i in order], dtype=np.int64)


def rank_of(scores: np.ndarray, index: int) -> int:
    """1-based rank of ``scores[index]`` among ``scores`` (higher score = more likely).

    Ties count against the canary (a tied candidate is ranked ahead of it), so the
    reported exposure is never optimistic.
    """
    s = np.asarray(scores)
    return int((s >= s[index]).sum())


def exposure(rank: int, space: int) -> float:
    """``log2(space) - log2(rank)``."""
    if not 1 <= rank <= space:
        raise ValueError("rank must be in [1, space]")
    return float(np.log2(space) - np.log2(rank))


def exposure_from_sample(
    canary_log_prob: float, sample_log_probs: np.ndarray, space: int
) -> dict[str, float]:
    """Estimate exposure from a uniform sample of other candidates.

    ``empirical``: the canary's rank within the sample scaled to the full space
    (capped at ``log2(sample size + 1)``, the most a sample can resolve).
    ``skewnorm``: the extrapolation of Carlini et al. (2019, Section 4.3), a
    skew-normal fit to the sample's log-perplexities; exposure is
    ``-log2 P(X <= canary log-perplexity)``, capped at ``log2(space)``.
    """
    x = -np.asarray(sample_log_probs, dtype=np.float64)
    c = -float(canary_log_prob)
    m = len(x)
    below = int((x <= c).sum())
    empirical = float(np.log2(m + 1) - np.log2(below + 1))
    a, loc, scale = stats.skewnorm.fit(x)
    p = float(stats.skewnorm.cdf(c, a, loc, scale))
    skew = float(min(np.log2(space), -np.log2(max(p, 1.0 / space))))
    return {"empirical": empirical, "skewnorm": skew}
