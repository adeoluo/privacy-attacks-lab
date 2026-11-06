"""Membership inference attacks: definitions, edge cases and a synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from privacy_attacks_lab.attacks import (
    ATTACKS,
    ShadowAttack,
    calibrated_attack,
    fit_gaussians,
    lira_offline,
    lira_online,
    loss_attack,
    shadow_features,
)
from privacy_attacks_lab.attacks.shadow import LogisticRegression
from privacy_attacks_lab.metrics import summarize

from .conftest import needs_sklearn


def simulate(
    models: int = 32, n: int = 400, seed: int = 0, gap: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic phi in which each example has its own difficulty and spread.

    IN models see ``phi ~ N(d + gap * s, s^2)`` and OUT models ``N(d, s^2)``, where
    ``d`` and ``s`` vary per example: the setting in which per-example calibration
    (LiRA) beats a global threshold.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 4, size=n)
    s = rng.uniform(0.3, 3.0, size=n)
    order = rng.uniform(size=(models, n)).argsort(0).argsort(0)
    masks = order < models // 2
    phi = d + s * rng.normal(size=(models, n)) + gap * s * masks
    return phi, masks


def test_registry_lists_every_attack() -> None:
    assert ATTACKS == ("loss", "calibrated", "shadow", "lira-offline", "lira-online")


def test_loss_attack_negates_and_averages_queries() -> None:
    np.testing.assert_allclose(loss_attack(np.array([1.0, 2.0])), [-1.0, -2.0])
    np.testing.assert_allclose(loss_attack(np.array([[1.0, 3.0], [0.0, 2.0]])), [-2.0, -1.0])
    with pytest.raises(ValueError):
        loss_attack(np.zeros((2, 2, 2)))


def test_calibrated_subtracts_out_mean() -> None:
    ref = np.array([[1.0, 5.0], [3.0, 7.0], [100.0, 100.0]])
    masks = np.array([[False, False], [False, False], [True, True]])
    out = calibrated_attack(np.array([2.0, 2.0]), ref, masks)
    np.testing.assert_allclose(out, [-(2 - 2.0), -(2 - 6.0)])
    online = calibrated_attack(np.array([2.0, 2.0]), ref, masks, online=True)
    np.testing.assert_allclose(online, [-(2 - 104 / 3), -(2 - 112 / 3)])


def test_calibrated_falls_back_without_out_refs() -> None:
    ref = np.array([[1.0], [3.0]])
    out = calibrated_attack(np.array([0.0]), ref, np.array([[True], [True]]))
    np.testing.assert_allclose(out, [2.0])


def test_calibrated_queries_and_shapes() -> None:
    t = np.ones((3, 2))
    r = np.ones((4, 3, 2))
    m = np.zeros((4, 3), dtype=bool)
    np.testing.assert_allclose(calibrated_attack(t, r, m), 0.0)
    with pytest.raises(ValueError, match="disagree"):
        calibrated_attack(np.ones(3), np.ones((4, 2)), np.zeros((4, 2), dtype=bool))


def test_fit_gaussians_mean_matches_numpy() -> None:
    phi, masks = simulate(16, 50)
    fit = fit_gaussians(phi, masks, center="mean")
    for j in range(5):
        np.testing.assert_allclose(fit.mu_in[j, 0], phi[masks[:, j], j].mean())
        np.testing.assert_allclose(fit.sigma_out[j, 0], phi[~masks[:, j], j].std())
    assert (fit.count_in == 8).all()
    assert (fit.count_out == 8).all()
    med = fit_gaussians(phi, masks)
    np.testing.assert_allclose(med.mu_out[0, 0], np.median(phi[~masks[:, 0], 0]))


def test_fit_gaussians_pooled_variance() -> None:
    phi, masks = simulate(16, 50)
    fit = fit_gaussians(phi, masks, fix_variance=True, center="mean")
    assert np.unique(fit.sigma_in).size == 1
    dev = np.where(masks, phi - fit.mu_in[:, 0][None], 0.0)
    np.testing.assert_allclose(fit.sigma_in[0, 0], np.sqrt((dev**2).sum() / masks.sum()))


def test_fit_gaussians_validates() -> None:
    with pytest.raises(ValueError, match="center"):
        fit_gaussians(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool), center="mode")
    with pytest.raises(ValueError, match="refs"):
        fit_gaussians(np.zeros(3), np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="ref_masks"):
        fit_gaussians(np.zeros((2, 3)), np.zeros((3, 3), dtype=bool))


def test_lira_online_is_gaussian_log_likelihood_ratio() -> None:
    phi, masks = simulate(20, 30, seed=2)
    target = np.random.default_rng(5).normal(size=30)
    fit = fit_gaussians(phi, masks, center="mean")
    expected = stats.norm.logpdf(target, fit.mu_in[:, 0], fit.sigma_in[:, 0]) - stats.norm.logpdf(
        target, fit.mu_out[:, 0], fit.sigma_out[:, 0]
    )
    got = lira_online(target, phi, masks, center="mean")
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_lira_offline_one_sided_and_pdf() -> None:
    phi, masks = simulate(20, 30, seed=3)
    target = np.zeros(30)
    fit = fit_gaussians(phi, masks, center="mean")
    np.testing.assert_allclose(
        lira_offline(target, phi, masks, center="mean"),
        (target - fit.mu_out[:, 0]) / fit.sigma_out[:, 0],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        lira_offline(target, phi, masks, center="mean", one_sided=False),
        -stats.norm.logpdf(target, fit.mu_out[:, 0], fit.sigma_out[:, 0]),
        rtol=1e-9,
    )


def test_lira_ignores_in_models_offline() -> None:
    phi, masks = simulate(20, 30)
    changed = np.where(masks, phi + 100.0, phi)
    t = np.zeros(30)
    np.testing.assert_allclose(lira_offline(t, phi, masks), lira_offline(t, changed, masks))


def test_lira_multiple_queries_average() -> None:
    phi, masks = simulate(20, 30)
    stacked = np.stack([phi, phi], axis=2)
    t = np.ones(30)
    one = lira_online(t, phi, masks)
    two = lira_online(np.stack([t, t], axis=1), stacked, masks)
    np.testing.assert_allclose(one, two)


def test_lira_shape_mismatch() -> None:
    phi, masks = simulate(4, 10)
    with pytest.raises(ValueError, match="target_phi"):
        lira_online(np.zeros(9), phi, masks)
    with pytest.raises(ValueError, match="target_phi"):
        lira_offline(np.zeros(9), phi, masks)


def test_lira_with_missing_in_or_out_models_is_finite() -> None:
    phi = np.random.default_rng(0).normal(size=(3, 4))
    masks = np.array(
        [[True, False, True, True], [True, False, False, True], [True, False, True, False]]
    )
    on = lira_online(np.zeros(4), phi, masks)
    off = lira_offline(np.zeros(4), phi, masks)
    assert np.isfinite(on).all()
    assert np.isfinite(off[1:]).all()


def test_lira_beats_global_threshold_on_heterogeneous_examples() -> None:
    phi, masks = simulate(65, 2000, seed=1)
    target, refs = 0, np.arange(1, 65)
    labels = masks[target]
    global_score = phi[target]
    on = summarize(lira_online(phi[target], phi[refs], masks[refs]), labels)
    off = summarize(lira_offline(phi[target], phi[refs], masks[refs]), labels)
    base = summarize(global_score, labels)
    assert on.tpr(0.01) > 3 * max(base.tpr(0.01), 0.01)
    assert off.tpr(0.01) > 3 * max(base.tpr(0.01), 0.01)
    assert on.auc > base.auc


def test_lira_online_approaches_neyman_pearson_optimum() -> None:
    # With exact Gaussians the optimal AUC is Phi(gap / sqrt(2)).
    gap = 1.5
    phi, masks = simulate(129, 3000, seed=4, gap=gap)
    refs = np.arange(1, 129)
    score = lira_online(phi[0], phi[refs], masks[refs], center="mean")
    auc = summarize(score, masks[0]).auc
    assert auc == pytest.approx(stats.norm.cdf(gap / np.sqrt(2)), abs=0.03)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), shift=st.floats(-5, 5))
def test_lira_scores_are_shift_invariant(seed: int, shift: float) -> None:
    phi, masks = simulate(10, 20, seed=seed)
    t = phi[0]
    a = lira_online(t, phi[1:], masks[1:])
    b = lira_online(t + shift, phi[1:] + shift, masks[1:])
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_shadow_features_shape() -> None:
    topk = np.full((4, 7, 2, 3), 0.3)
    phi = np.zeros((4, 7, 2))
    f = shadow_features(topk, phi)
    assert f.shape == (4, 7, 8)
    np.testing.assert_allclose(f[0, 0, :3], np.log(0.3))
    with pytest.raises(ValueError, match="shape"):
        shadow_features(topk, np.zeros((4, 7)))


def test_logistic_regression_separates() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = x[:, 0] + 0.1 * rng.normal(size=500) > 0
    lr = LogisticRegression(l2=1e-4).fit(x, y)
    assert ((lr.decision_function(x) > 0) == y).mean() > 0.95
    assert lr.coef[0] > abs(lr.coef[1])


@needs_sklearn
@pytest.mark.validation
def test_logistic_regression_matches_sklearn() -> None:
    from sklearn.linear_model import LogisticRegression as SkLR

    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 4))
    y = (x @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=800)) > 0
    l2 = 1e-2
    ours = LogisticRegression(l2=l2, max_iter=1000).fit(x, y)
    # sklearn minimises 0.5 ||w||^2 + C sum(loss); ours is mean(loss) + 0.5 l2 ||w||^2.
    ref = SkLR(C=1.0 / (l2 * len(y)), tol=1e-10, max_iter=10_000).fit(x, y)
    np.testing.assert_allclose(ours.coef, ref.coef_[0], rtol=1e-3, atol=1e-4)
    assert ours.bias == pytest.approx(ref.intercept_[0], abs=1e-4)


def _shadow_data(models: int = 8, n: int = 600, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n)
    masks = rng.uniform(size=(models, n)).argsort(0).argsort(0) < models // 2
    conf = np.where(masks, 0.95, 0.6) + 0.05 * rng.normal(size=(models, n))
    conf = np.clip(conf, 0.34, 0.999)
    topk = np.stack([conf, (1 - conf) * 0.7, (1 - conf) * 0.3], axis=-1)[:, :, None, :]
    phi = np.log(conf / (1 - conf))[:, :, None]
    return shadow_features(topk, phi), masks, labels


def test_shadow_attack_learns_membership() -> None:
    feats, masks, labels = _shadow_data()
    attack = ShadowAttack(min_rows_per_class=100).fit(feats[1:], masks[1:], labels)
    assert set(attack.models) == {0, 1, 2}
    s = summarize(attack.score(feats[0], labels), masks[0])
    assert s.auc > 0.95


def test_shadow_attack_global_and_subsampled() -> None:
    feats, masks, labels = _shadow_data()
    attack = ShadowAttack(per_class=False, max_rows=1000).fit(feats[1:], masks[1:], labels)
    assert attack.models == {}
    assert summarize(attack.score(feats[0], labels), masks[0]).auc > 0.9


def test_shadow_attack_errors() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ShadowAttack().score(np.zeros((2, 4)), np.zeros(2))
    with pytest.raises(ValueError, match="shadows"):
        ShadowAttack().fit(np.zeros((2, 4)), np.zeros((2, 2), dtype=bool), np.zeros(2))


# --------------------------------------------------------------------------- reference parity


def test_matches_authors_released_scoring() -> None:
    from privacy_attacks_lab.attacks import lira_reference as ref

    phi, m = simulate(32, 500, seed=7)
    rng = np.random.default_rng(0)
    refs = np.stack([phi, phi + 0.1 * rng.normal(size=phi.shape)], axis=2)
    target = rng.normal(0, 4, size=(500, 2))
    # Balanced masks: per-example variance versions agree to rounding.
    np.testing.assert_allclose(
        lira_online(target, refs, m), ref.reference_online(m, refs, target), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        lira_offline(target, refs, m, one_sided=False),
        ref.reference_offline(m, refs, target),
        rtol=1e-9,
        atol=1e-9,
    )
    # Offline with a pooled variance: the reference pools around the global mean,
    # so the two only agree in ranking when every example shares one centre.
    flat = np.where(m[:, :, None], 1.0, 0.0) + rng.normal(size=refs.shape)
    a = lira_offline(target, flat, m, fix_variance=True, one_sided=False)
    b = ref.reference_offline(m, flat, target, fix_variance=True)
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert ref.reference_online(m, refs, target, fix_variance=True).shape == (500,)
