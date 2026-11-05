"""Unit and property tests of the RDP, PRV and GDP accountants."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from privacy_attacks_lab.accountants import (
    DEFAULT_ORDERS,
    Accountant,
    calibrate_noise,
    compute_rdp,
    epsilon,
    gdp,
    max_tpr,
    prv,
    rdp,
    rdp_epsilon,
)

Q, SIGMA, STEPS, DELTA = 256 / 60000, 1.1, 14062, 1e-5


def test_default_orders_match_opacus_grid() -> None:
    assert len(DEFAULT_ORDERS) == 99 + 52
    assert DEFAULT_ORDERS[0] == pytest.approx(1.1)
    assert DEFAULT_ORDERS[98] == pytest.approx(10.9)
    assert DEFAULT_ORDERS[-1] == 63.0


def test_known_epsilon_mnist_setting() -> None:
    # Abadi et al. 2016 style MNIST setting; value cross-checked against Opacus.
    assert epsilon(Q, SIGMA, STEPS, DELTA) == pytest.approx(2.5965558689607, rel=1e-10)


def test_prv_is_tighter_than_rdp_and_gdp_close() -> None:
    e_rdp = epsilon(Q, SIGMA, STEPS, DELTA, accountant="rdp")
    e_prv = epsilon(Q, SIGMA, STEPS, DELTA, accountant="prv")
    e_gdp = epsilon(Q, SIGMA, STEPS, DELTA, accountant="gdp")
    assert e_prv < e_rdp
    assert abs(e_gdp - e_prv) < 0.2


def test_zero_steps_and_zero_rate_are_free() -> None:
    assert epsilon(0.1, 1.0, 0, DELTA) == 0.0
    assert epsilon(0.0, 1.0, 100, DELTA) == 0.0


def test_zero_noise_is_infinite() -> None:
    assert math.isinf(epsilon(0.1, 0.0, 10, DELTA))


@pytest.mark.parametrize(
    ("q", "sigma", "steps", "delta"),
    [
        (-0.1, 1.0, 1, 1e-5),
        (1.1, 1.0, 1, 1e-5),
        (0.1, -1.0, 1, 1e-5),
        (0.1, 1.0, -1, 1e-5),
        (0.1, 1.0, 1, 0.0),
        (0.1, 1.0, 1, 1.0),
    ],
)
def test_invalid_arguments_raise(q: float, sigma: float, steps: int, delta: float) -> None:
    with pytest.raises(ValueError):
        epsilon(q, sigma, steps, delta)


def test_unknown_accountant() -> None:
    with pytest.raises(ValueError, match="unknown accountant"):
        epsilon(0.1, 1.0, 10, DELTA, accountant="xyz")  # type: ignore[arg-type]


def test_full_batch_is_gaussian_mechanism() -> None:
    # q = 1: RDP of the Gaussian mechanism is alpha / (2 sigma^2) per step.
    values = rdp.compute_rdp(1.0, 2.0, 3, [2.0, 5.5, 10.0])
    np.testing.assert_allclose(values, 3 * np.array([2.0, 5.5, 10.0]) / 8.0)


def test_rdp_single_edge_cases() -> None:
    assert rdp.rdp_single(0.0, 1.0, 2.0) == 0.0
    assert math.isinf(rdp.rdp_single(0.5, 0.0, 2.0))
    assert math.isinf(rdp.rdp_single(0.5, 1.0, math.inf))
    for bad in [(-0.1, 1.0, 2.0), (0.1, -1.0, 2.0), (0.1, 1.0, 1.0)]:
        with pytest.raises(ValueError):
            rdp.rdp_single(*bad)


def test_compute_rdp_negative_steps() -> None:
    with pytest.raises(ValueError):
        rdp.compute_rdp(0.1, 1.0, -1)


def test_integer_and_fractional_orders_are_continuous() -> None:
    # The two series agree in the limit; check nearby orders bracket the integer one.
    below = rdp.rdp_single(0.01, 1.0, 4.999999)
    at = rdp.rdp_single(0.01, 1.0, 5.0)
    above = rdp.rdp_single(0.01, 1.0, 5.000001)
    assert below <= at * (1 + 1e-5) and at <= above * (1 + 1e-5)
    assert below == pytest.approx(at, rel=1e-4)


def test_log_helpers() -> None:
    assert rdp.log_add(-math.inf, 1.0) == 1.0
    assert rdp.log_add(math.log(2), math.log(3)) == pytest.approx(math.log(5))
    assert rdp.log_sub(math.log(5), math.log(3)) == pytest.approx(math.log(2))
    assert rdp.log_sub(1.0, -math.inf) == 1.0
    assert rdp.log_sub(1.0, 1.0) == -math.inf
    with pytest.raises(ValueError):
        rdp.log_sub(0.0, 1.0)
    assert rdp.log_sub(1000.0, 0.0) == pytest.approx(1000.0)
    assert rdp.log_erfc(0.5) == pytest.approx(math.log(math.erfc(0.5)))
    assert np.isfinite(rdp.log_erfc(40.0))


def test_rdp_to_epsilon_errors_and_nan() -> None:
    with pytest.raises(ValueError):
        rdp.rdp_to_epsilon([2.0, 3.0], [1.0], 1e-5)
    with pytest.raises(ValueError):
        rdp.rdp_to_epsilon([2.0], [1.0], 0.0)
    eps, order = rdp.rdp_to_epsilon([2.0], [math.nan], 1e-5)
    assert math.isinf(eps) and math.isnan(order)


@settings(max_examples=40, deadline=None)
@given(
    q=st.floats(1e-4, 0.5),
    sigma=st.floats(0.5, 5.0),
    steps=st.integers(1, 5000),
)
def test_epsilon_monotone_in_steps_and_sigma(q: float, sigma: float, steps: int) -> None:
    e = epsilon(q, sigma, steps, DELTA)
    assert e >= 0.0
    assert epsilon(q, sigma, steps + 100, DELTA) >= e - 1e-12
    assert epsilon(q, sigma * 1.5, steps, DELTA) <= e + 1e-12


@settings(max_examples=40, deadline=None)
@given(q=st.floats(1e-4, 0.9), sigma=st.floats(0.3, 10.0), alpha=st.sampled_from(DEFAULT_ORDERS))
def test_rdp_bounded_by_gaussian(q: float, sigma: float, alpha: float) -> None:
    # Subsampling never hurts: RDP(q) <= RDP(1) = alpha / (2 sigma^2).
    assert rdp.rdp_single(q, sigma, alpha) <= alpha / (2 * sigma * sigma) * (1 + 1e-9)


@settings(max_examples=30, deadline=None)
@given(q=st.floats(1e-3, 0.3), sigma=st.floats(0.5, 4.0), k=st.integers(1, 100))
def test_rdp_composes_linearly(q: float, sigma: float, k: int) -> None:
    one = compute_rdp(q, sigma, 1)
    np.testing.assert_allclose(compute_rdp(q, sigma, k), k * one, rtol=1e-12)


@pytest.mark.parametrize("accountant", ["rdp", "prv", "gdp"])
def test_calibrate_noise_meets_target(accountant: str) -> None:
    q, steps = 0.01, 1000
    for target in (1.0, 4.0):
        sigma = calibrate_noise(target, DELTA, q, steps, accountant=accountant)  # type: ignore[arg-type]
        assert epsilon(q, sigma, steps, DELTA, accountant=accountant) <= target + 1e-9  # type: ignore[arg-type]
        slightly_less = sigma * (1 - 2e-3)
        assert epsilon(q, slightly_less, steps, DELTA, accountant=accountant) > target - 0.05  # type: ignore[arg-type]


def test_calibrate_noise_python_backend_matches_rust() -> None:
    a = calibrate_noise(2.0, DELTA, 0.01, 500, backend="python")
    b = calibrate_noise(2.0, DELTA, 0.01, 500, backend="auto")
    assert a == pytest.approx(b, rel=2e-4)


def test_calibrate_noise_rejects_bad_targets() -> None:
    with pytest.raises(ValueError):
        calibrate_noise(0.0, DELTA, 0.01, 10)
    with pytest.raises(ValueError, match="too small"):
        calibrate_noise(1e-9, 1e-12, 1.0, 10_000_000, backend="python")
    with pytest.raises(ValueError, match="too small"):
        calibrate_noise(1e-9, 1e-12, 1.0, 10_000_000)


def test_rdp_epsilon_returns_order() -> None:
    eps, order = rdp_epsilon(Q, SIGMA, STEPS, DELTA)
    assert order in DEFAULT_ORDERS
    eps_py, order_py = rdp_epsilon(Q, SIGMA, STEPS, DELTA, backend="python")
    assert eps == pytest.approx(eps_py, rel=1e-12) and order == order_py


def test_accountant_merges_and_matches_closed_form() -> None:
    acc = Accountant()
    assert acc.epsilon(DELTA) == 0.0
    for _ in range(100):
        acc.step(1.0, 0.01)
    assert acc.history == [(1.0, 0.01, 100)] and acc.steps == 100
    assert acc.epsilon(DELTA) == pytest.approx(epsilon(0.01, 1.0, 100, DELTA))


@pytest.mark.parametrize("mechanism", ["rdp", "prv"])
def test_accountant_heterogeneous_history(mechanism: str) -> None:
    acc = Accountant(mechanism=mechanism)  # type: ignore[arg-type]
    acc.step(1.0, 0.01, 200)
    acc.step(2.0, 0.02, 300)
    assert len(acc.history) == 2
    mixed = acc.epsilon(DELTA)
    low = epsilon(0.01, 1.0, 200, DELTA, accountant=mechanism)  # type: ignore[arg-type]
    assert mixed > low


def test_gdp_accountant_requires_constant_parameters() -> None:
    acc = Accountant(mechanism="gdp")
    acc.step(1.0, 0.01, 100)
    assert acc.epsilon(DELTA) == pytest.approx(epsilon(0.01, 1.0, 100, DELTA, accountant="gdp"))
    acc.step(2.0, 0.01, 100)
    with pytest.raises(ValueError):
        acc.epsilon(DELTA)


def test_gdp_helpers() -> None:
    mu = gdp.mu_poisson(0.01, 1.0, 10000)
    assert mu > 0
    assert gdp.epsilon_from_mu(1e-6, 1e-5) == 0.0
    e = gdp.epsilon_from_mu(mu, DELTA)
    assert gdp.delta_from_mu(e, mu) == pytest.approx(DELTA, rel=1e-6)


def test_prv_bounds_are_ordered() -> None:
    r = prv.prv_epsilon([(SIGMA, Q, STEPS)], DELTA)
    assert r.lower < r.estimate < r.upper
    assert r.upper - r.lower < 0.05


def test_prv_delta_roundtrip() -> None:
    d = prv.compose([(1.0, 0.01, 1000)], eps_error=0.01, delta_error=1e-8)
    eps = d.epsilon(DELTA, 1e-8, 0.01).estimate
    assert d.delta(eps) == pytest.approx(DELTA, rel=0.05)
    assert math.isinf(d.epsilon(0.0, 1e-8, 0.01).upper)


def test_prv_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        prv.SubsampledGaussianPRV(0.0, 1.0)
    with pytest.raises(ValueError):
        prv.SubsampledGaussianPRV(0.1, 0.0)
    with pytest.raises(ValueError):
        prv.compose([], 0.01, 1e-8)
    odd = prv.DiscretePRV(np.ones(3) / 3, prv.Grid(-1.0, 1.0, 3))
    with pytest.raises(ValueError):
        prv.self_compose(odd, 2)


def test_prv_cdf_is_a_distribution() -> None:
    y = prv.SubsampledGaussianPRV(0.1, 1.0)
    ts = np.linspace(-1, 10, 500)
    c = y.cdf(ts)
    assert np.all(np.diff(c) >= -1e-15)
    assert c[0] == 0.0 and c[-1] == pytest.approx(1.0, abs=1e-12)


def test_prv_grid_alignment() -> None:
    g = prv.Grid.aligned(-1.03, 1.01, 0.1)
    assert g.size % 2 == 0
    assert g.dt == pytest.approx(0.1)
    assert g.shifted(0.5).shift == 0.5


@pytest.mark.parametrize("eps", [0.5, 1.0, 8.0])
def test_max_tpr_bound(eps: float) -> None:
    f = np.array([0.0, 1e-3, 0.01, 0.5, 1.0])
    t = max_tpr(f, eps, DELTA)
    assert np.all(t >= f - 1e-12)
    assert np.all(np.diff(t) >= 0)
    assert t[-1] == pytest.approx(1.0)
    a = math.exp(eps) * 1e-3 + DELTA
    b = 1 - math.exp(-eps) * (1 - DELTA - 1e-3)
    assert t[1] == pytest.approx(min(a, b, 1.0), rel=1e-9)


def test_max_tpr_infinite_epsilon() -> None:
    assert np.all(max_tpr(np.array([0.0, 0.5]), math.inf, DELTA) == 1.0)
