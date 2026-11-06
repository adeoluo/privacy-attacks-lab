"""Per-example signals: stability and agreement with direct formulas."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from privacy_attacks_lab import signals

logit_arrays = arrays(
    np.float64, (5, 4), elements=st.floats(-30, 30, allow_nan=False, allow_infinity=False)
)


@settings(max_examples=50, deadline=None)
@given(z=logit_arrays, y=st.integers(0, 3))
def test_phi_matches_definition(z: np.ndarray, y: int) -> None:
    p = signals.softmax(z)[:, y]
    phi = signals.logit_scaled_confidence(z, np.full(5, y))
    # Away from 0 and 1, where the reference formula itself loses precision.
    ok = (p > 1e-9) & (p < 1 - 1e-9)
    np.testing.assert_allclose(phi[ok], np.log(p[ok]) - np.log1p(-p[ok]), rtol=1e-5, atol=1e-6)


@settings(max_examples=50, deadline=None)
@given(z=logit_arrays, y=st.integers(0, 3))
def test_loss_matches_definition(z: np.ndarray, y: int) -> None:
    p = signals.softmax(z)[:, y]
    np.testing.assert_allclose(
        signals.loss(z, np.full(5, y)), -np.log(np.maximum(p, 1e-300)), rtol=1e-6
    )


def test_phi_is_finite_for_confident_predictions() -> None:
    z = np.array([[1000.0, 0.0, 0.0]])
    assert signals.logit_scaled_confidence(z, np.array([0]))[0] == pytest.approx(
        1000.0 - np.log(2.0)
    )
    assert signals.logit_scaled_confidence(z, np.array([1]))[0] == pytest.approx(-1000.0)


def test_topk_and_correct() -> None:
    z = np.array([[0.0, 2.0, 1.0], [5.0, 0.0, 0.0]])
    top = signals.top_k_probs(z, k=2)
    assert top.shape == (2, 2) and np.all(np.diff(top, axis=1) <= 0)
    np.testing.assert_array_equal(signals.correct(z, np.array([1, 1])), [True, False])
    assert signals.top_k_probs(z, k=10).shape == (2, 3)


def test_broadcast_labels_over_queries() -> None:
    z = np.random.default_rng(0).normal(size=(6, 2, 4))
    s = signals.summarize(z, np.arange(6)[:, None] % 4)
    assert s["phi"].shape == (6, 2) and s["topk"].shape == (6, 2, 3)
    assert s["phi"].dtype == np.float32


@pytest.mark.parametrize(
    ("z", "y"),
    [(np.zeros((2, 1)), np.array([0, 0])), (np.zeros((2, 3)), np.array([0, 3]))],
)
def test_signal_validation(z: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(ValueError):
        signals.loss(z, y)
