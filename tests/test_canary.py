"""Canaries and exposure."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from privacy_attacks_lab import canary


def test_canary_text_and_index() -> None:
    c = canary.Canary("Orrin Vale", (0, 4, 8, 1, 9, 3), 2)
    assert c.secret == "0 4 8 1 9 3"
    assert c.prefix == "The vault code of Orrin Vale is "
    assert c.text == "The vault code of Orrin Vale is 0 4 8 1 9 3."
    assert c.index == 48193


def test_make_canaries_levels_and_uniqueness() -> None:
    cs = canary.make_canaries((1, 4), per_level=3)
    assert [c.repeats for c in cs] == [1, 1, 1, 4, 4, 4]
    assert len({c.name for c in cs}) == 6
    assert all(len(c.digits) == 6 for c in cs)
    assert canary.make_canaries(seed=1) != canary.make_canaries(seed=2)
    with pytest.raises(ValueError, match="at most"):
        canary.make_canaries((1, 2, 3), per_level=10)


def test_insert_counts_and_labels() -> None:
    docs = [f"doc {i}" for i in range(20)]
    cs = canary.make_canaries((1, 3), per_level=1)
    out, which = canary.insert(docs, cs, seed=0)
    assert len(out) == 24
    assert (which == -1).sum() == 20
    assert (which == 1).sum() == 3
    for text, k in zip(out, which, strict=True):
        if k >= 0:
            assert text == cs[k].text
    assert sorted(t for t, k in zip(out, which, strict=True) if k < 0) == sorted(docs)


def test_rank_and_exposure() -> None:
    scores = np.array([0.1, 0.5, 0.3, 0.5])
    assert canary.rank_of(scores, 1) == 2  # tie counts against the canary
    assert canary.rank_of(scores, 0) == 4
    assert canary.exposure(1, 1024) == pytest.approx(10.0)
    assert canary.exposure(1024, 1024) == 0.0
    with pytest.raises(ValueError):
        canary.exposure(0, 10)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_random_canary_has_exposure_about_one(seed: int) -> None:
    # A canary drawn like every other candidate has E[2^-exposure] = 1/2.
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=4096)
    ranks = [canary.rank_of(scores, i) for i in rng.integers(0, 4096, size=200)]
    vals = [canary.exposure(r, 4096) for r in ranks]
    assert 0.6 < np.mean(vals) < 2.0


def test_exposure_from_sample_matches_exact_on_skewnormal() -> None:
    from scipy import stats

    rng = np.random.default_rng(0)
    space = 1_000_000
    pop = -stats.skewnorm.rvs(3.0, loc=20, scale=4, size=space, random_state=rng)
    idx = int(np.argsort(pop)[-50])  # the 50th most likely candidate
    exact = canary.exposure(canary.rank_of(pop, idx), space)
    est = canary.exposure_from_sample(pop[idx], rng.choice(pop, 20_000), space)
    assert est["skewnorm"] == pytest.approx(exact, abs=0.5)
    assert est["empirical"] <= np.log2(20_001) + 1e-9
    worst = canary.exposure_from_sample(pop.min(), rng.choice(pop, 1000), space)
    assert worst["skewnorm"] < 0.1
