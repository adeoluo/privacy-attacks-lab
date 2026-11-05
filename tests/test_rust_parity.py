"""The Rust core and the Python reference must agree."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from privacy_attacks_lab import _backend
from privacy_attacks_lab.accountants import DEFAULT_ORDERS, compute_rdp, rdp_epsilon
from privacy_attacks_lab.accountants import rdp as ref

pytestmark = pytest.mark.skipif(not _backend.rust_available(), reason="Rust core not built")


@settings(max_examples=60, deadline=None)
@given(q=st.floats(1e-5, 1.0), sigma=st.floats(0.3, 20.0), steps=st.integers(0, 100_000))
def test_compute_rdp_parity(q: float, sigma: float, steps: int) -> None:
    a = compute_rdp(q, sigma, steps, backend="rust")
    b = compute_rdp(q, sigma, steps, backend="python")
    # Per-step values near 1e-7 carry absolute rounding of about 1e-15 (log A is a
    # difference of numbers near 1), far below anything that moves epsilon.
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=2e-14 * max(steps, 1))


@settings(max_examples=40, deadline=None)
@given(
    q=st.floats(1e-4, 0.5),
    sigma=st.floats(0.4, 10.0),
    steps=st.integers(1, 50_000),
    delta=st.sampled_from([1e-3, 1e-5, 1e-7]),
)
def test_epsilon_parity(q: float, sigma: float, steps: int, delta: float) -> None:
    er, orr = rdp_epsilon(q, sigma, steps, delta, backend="rust")
    ep, orp = rdp_epsilon(q, sigma, steps, delta, backend="python")
    assert er == pytest.approx(ep, rel=1e-10)
    assert orr == orp


def test_rdp_to_epsilon_parity_and_errors() -> None:
    rs = _backend.rust()
    orders = list(DEFAULT_ORDERS)
    values = list(ref.compute_rdp(0.01, 1.0, 1000))
    assert rs.rdp_to_epsilon(orders, values, 1e-5) == pytest.approx(
        ref.rdp_to_epsilon(orders, values, 1e-5)
    )
    with pytest.raises(ValueError):
        rs.rdp_to_epsilon(orders, values[:-1], 1e-5)
    with pytest.raises(ValueError):
        rs.rdp_to_epsilon(orders, values, 2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda rs: rs.compute_rdp(1.5, 1.0, 1, [2.0]),
        lambda rs: rs.compute_rdp(0.1, -1.0, 1, [2.0]),
        lambda rs: rs.compute_rdp(0.1, 1.0, 1, [1.0]),
        lambda rs: rs.epsilon(0.1, 1.0, 1, 0.0, [2.0]),
        lambda rs: rs.calibrate_sigma(0.0, 1e-5, 0.1, 10, [2.0], 1e-3),
    ],
)
def test_rust_validates_inputs(call: object) -> None:
    with pytest.raises(ValueError):
        call(_backend.rust())  # type: ignore[operator]


def test_backend_resolution() -> None:
    assert _backend.resolve("python") == "python"
    assert _backend.resolve("auto") == "rust"
    assert _backend.resolve("rust") == "rust"
    with pytest.raises(ValueError):
        _backend.resolve("gpu")  # type: ignore[arg-type]
    assert _backend.rust().__version__
