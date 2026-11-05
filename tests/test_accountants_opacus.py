"""Validation of the accountants against Opacus (opt-in: the validation group)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from privacy_attacks_lab.accountants import DEFAULT_ORDERS, calibrate_noise, compute_rdp, epsilon
from privacy_attacks_lab.accountants import prv as prv_mod

from .conftest import needs_opacus

pytestmark = [needs_opacus, pytest.mark.validation]

GRID = list(itertools.product([1e-3, 0.004, 0.02, 0.1], [0.6, 1.0, 2.0, 5.0], [1, 100, 10_000]))


@pytest.mark.parametrize(("q", "sigma", "steps"), GRID)
def test_rdp_curve_matches_opacus(q: float, sigma: float, steps: int) -> None:
    from opacus.accountants.analysis import rdp as orpd

    ours = compute_rdp(q, sigma, steps)
    theirs = orpd.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=list(DEFAULT_ORDERS))
    # Agreement to 8 significant digits; the residue is rounding in the fractional-order series.
    np.testing.assert_allclose(ours, theirs, rtol=1e-8, atol=0)


@pytest.mark.parametrize(("q", "sigma", "steps"), GRID)
def test_rdp_epsilon_matches_opacus(q: float, sigma: float, steps: int) -> None:
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    assert epsilon(q, sigma, steps, 1e-5) == pytest.approx(acc.get_epsilon(1e-5), rel=1e-11)


@pytest.mark.parametrize(
    ("q", "sigma", "steps"),
    [(256 / 60000, 1.1, 14062), (0.01, 1.0, 1000), (0.02, 0.8, 500), (0.004, 2.0, 5000)],
)
def test_prv_epsilon_matches_opacus(q: float, sigma: float, steps: int) -> None:
    from opacus.accountants import PRVAccountant

    acc = PRVAccountant()
    acc.history = [(sigma, q, steps)]
    theirs = acc.get_epsilon(1e-5, eps_error=0.01)
    assert epsilon(q, sigma, steps, 1e-5, accountant="prv") == pytest.approx(theirs, abs=1e-6)


def test_prv_heterogeneous_history_matches_opacus() -> None:
    from opacus.accountants import PRVAccountant

    history = [(1.0, 0.01, 300), (2.0, 0.02, 200), (1.5, 0.01, 100)]
    acc = PRVAccountant()
    acc.history = list(history)
    ours = prv_mod.prv_epsilon(history, 1e-5).upper
    assert ours == pytest.approx(acc.get_epsilon(1e-5), abs=1e-6)


@pytest.mark.parametrize(("q", "sigma", "steps"), GRID[::3])
def test_gdp_matches_opacus(q: float, sigma: float, steps: int) -> None:
    from opacus.accountants.analysis import gdp

    theirs = gdp.compute_eps_poisson(steps=steps, noise_multiplier=sigma, sample_rate=q, delta=1e-5)
    assert epsilon(q, sigma, steps, 1e-5, accountant="gdp") == pytest.approx(theirs, rel=1e-8)


@pytest.mark.parametrize("target", [1.0, 3.0, 8.0])
def test_calibration_matches_opacus(target: float) -> None:
    from opacus.accountants.utils import get_noise_multiplier

    theirs = get_noise_multiplier(
        target_epsilon=target,
        target_delta=1e-5,
        sample_rate=0.004,
        steps=2500,
        accountant="rdp",
        epsilon_tolerance=0.001,
    )
    ours = calibrate_noise(target, 1e-5, 0.004, 2500)
    # Opacus stops within epsilon_tolerance of the target; both must meet the target.
    assert ours == pytest.approx(theirs, rel=2e-3)
    assert epsilon(0.004, ours, 2500, 1e-5) <= target
