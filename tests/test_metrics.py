"""ROC metrics: definitions, edge cases, Rust parity and agreement with scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from privacy_attacks_lab import _backend, metrics

from .conftest import needs_sklearn

BACKENDS = ["python", "rust"] if _backend.rust_available() else ["python"]


def _scores(n: int = 2000, shift: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.5
    scores = rng.normal(size=n) + shift * labels
    return scores, labels


@pytest.mark.parametrize("backend", BACKENDS)
def test_perfect_and_reversed_scores(backend: str) -> None:
    labels = np.array([1, 1, 0, 0])
    s = metrics.summarize(np.array([4.0, 3.0, 2.0, 1.0]), labels, backend=backend)  # type: ignore[arg-type]
    assert s.auc == 1.0
    assert s.accuracy == 1.0
    assert s.tprs == (1.0, 1.0, 1.0)
    r = metrics.summarize(np.array([1.0, 2.0, 3.0, 4.0]), labels, backend=backend)  # type: ignore[arg-type]
    assert r.auc == 0.0
    assert r.tprs == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_constant_scores_give_chance(backend: str) -> None:
    s = metrics.summarize(np.zeros(10), np.arange(10) % 2, backend=backend)  # type: ignore[arg-type]
    assert s.auc == pytest.approx(0.5)
    assert s.accuracy == pytest.approx(0.5)
    assert s.tpr(0.01) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_curve_starts_at_origin_and_ends_at_one(backend: str) -> None:
    s, lab = _scores(300)
    fpr, tpr, thr = metrics.roc_curve(s, lab, backend=backend)  # type: ignore[arg-type]
    assert (fpr[0], tpr[0], thr[0]) == (0.0, 0.0, np.inf)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert np.all(np.diff(thr[1:]) < 0)


def test_curve_ties_move_together() -> None:
    fpr, tpr, thr = metrics.roc_curve(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0]))
    np.testing.assert_allclose(fpr, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(thr, [np.inf, 1.0, 0.0])


def test_tpr_at_fpr_takes_points_at_or_below_target() -> None:
    fpr = np.array([0.0, 0.0005, 0.002, 0.5, 1.0])
    tpr = np.array([0.0, 0.1, 0.3, 0.9, 1.0])
    assert metrics.tpr_at_fpr(fpr, tpr, 0.001) == 0.1
    assert metrics.tpr_at_fpr(fpr, tpr, 0.002) == 0.3
    assert metrics.tpr_at_fpr(fpr, tpr, 1.0) == 1.0
    assert metrics.tpr_at_fpr(fpr[1:], tpr[1:], 1e-5) == 0.0


def test_auc_trapezoid() -> None:
    assert metrics.auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5
    assert metrics.auc(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_bad_inputs(backend: str) -> None:
    with pytest.raises(ValueError, match="same length"):
        metrics.summarize(np.zeros(3), np.zeros(2), backend=backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="0/1"):
        metrics.summarize(np.zeros(3), np.array([0, 2, 1]), backend=backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reps"):
        metrics.bootstrap(np.zeros(3), np.array([0, 1, 1]), reps=0)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        metrics.summarize(np.zeros(2), np.array([0, 1]), backend="gpu")  # type: ignore[arg-type]


def test_single_class_labels_do_not_crash() -> None:
    fpr, tpr, _ = metrics.roc_curve(np.arange(5.0), np.ones(5), backend="python")
    assert np.isnan(fpr[1:]).all()
    assert tpr[-1] == 1.0
    fpr, tpr, _ = metrics.roc_curve(np.arange(5.0), np.zeros(5), backend="python")
    assert np.isnan(tpr[1:]).all()


def test_nan_scores_rank_last() -> None:
    s = np.array([np.nan, 2.0, 1.0, np.nan])
    lab = np.array([1, 1, 0, 0])
    fpr, tpr, _ = metrics.roc_curve(s, lab, backend="python")
    # The two finite scores are ranked first; the NaNs form one final tied group.
    np.testing.assert_allclose(tpr, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(fpr, [0.0, 0.0, 0.5, 1.0])


def test_summary_to_dict_and_lookup() -> None:
    s, lab = _scores()
    r = metrics.summarize(s, lab)
    d = r.to_dict()
    assert set(d) == {"auc", "accuracy", "tpr@0.001", "tpr@0.01", "tpr@0.1"}
    assert r.tpr(0.1) == d["tpr@0.1"]


def test_stronger_signal_dominates() -> None:
    weak = metrics.summarize(*_scores(shift=0.5))
    strong = metrics.summarize(*_scores(shift=2.0))
    assert strong.auc > weak.auc
    assert strong.tpr(0.01) > weak.tpr(0.01)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bootstrap_interval_contains_estimate(backend: str) -> None:
    s, lab = _scores(1000)
    ci = metrics.bootstrap(s, lab, reps=200, seed=3, backend=backend)  # type: ignore[arg-type]
    for key, iv in ci.items():
        assert iv.low <= iv.estimate + 1e-9, key
        assert iv.estimate <= iv.high + 1e-9, key
    assert ci["auc"].high - ci["auc"].low < 0.1
    assert set(ci["auc"].to_dict()) == {"estimate", "low", "high"}


def test_bootstrap_is_deterministic() -> None:
    s, lab = _scores(500)
    a = metrics.bootstrap(s, lab, reps=50, seed=1)
    b = metrics.bootstrap(s, lab, reps=50, seed=1)
    assert a == b


def test_bootstrap_backends_agree_in_distribution() -> None:
    if not _backend.rust_available():
        pytest.skip("Rust core not built")
    s, lab = _scores(3000)
    a = metrics.bootstrap(s, lab, reps=400, seed=0, backend="rust")
    b = metrics.bootstrap(s, lab, reps=400, seed=0, backend="python")
    # Different random streams, same sampling distribution.
    assert a["auc"].low == pytest.approx(b["auc"].low, abs=0.01)
    assert a["auc"].high == pytest.approx(b["auc"].high, abs=0.01)


def test_log_roc_points_grid() -> None:
    s, lab = _scores()
    fpr, tpr, _ = metrics.roc_curve(s, lab)
    g, t = metrics.log_roc_points(fpr, tpr)
    assert len(g) == len(t) == 200
    assert g[0] == pytest.approx(1e-5)
    assert np.all(np.diff(t) >= 0)
    _, t2 = metrics.log_roc_points(fpr, tpr, np.array([0.5, 1.0]))
    assert t2[-1] == 1.0


@settings(max_examples=60, deadline=None)
@given(
    s=arrays(np.float64, st.integers(2, 80), elements=st.floats(-5, 5, allow_nan=False)),
    seed=st.integers(0, 1000),
)
def test_rust_and_python_curves_agree(s: np.ndarray, seed: int) -> None:
    if not _backend.rust_available():
        pytest.skip("Rust core not built")
    lab = np.random.default_rng(seed).random(len(s)) < 0.5
    lab[0], lab[1] = True, False
    a = metrics.roc_curve(s, lab, backend="rust")
    b = metrics.roc_curve(s, lab, backend="python")
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-12)
    sa = metrics.summarize(s, lab, backend="rust")
    sb = metrics.summarize(s, lab, backend="python")
    assert sa.auc == pytest.approx(sb.auc, abs=1e-12)
    assert sa.accuracy == pytest.approx(sb.accuracy, abs=1e-12)
    assert sa.tprs == pytest.approx(sb.tprs, abs=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    s=arrays(np.int64, st.integers(2, 60), elements=st.integers(-30, 30)),
    seed=st.integers(0, 1000),
)
def test_auc_is_rank_statistic(s: np.ndarray, seed: int) -> None:
    lab = np.random.default_rng(seed).random(len(s)) < 0.5
    lab[0], lab[1] = True, False
    s = s / 10.0
    pos, neg = s[lab], s[~lab]
    mann_whitney = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (
        len(pos) * len(neg)
    )
    assert metrics.summarize(s, lab).auc == pytest.approx(mann_whitney, abs=1e-12)
    # Increasing affine transforms leave every metric unchanged.
    a = metrics.summarize(s, lab)
    b = metrics.summarize(2.0 * s + 1.0, lab)
    assert a == b


@needs_sklearn
@pytest.mark.validation
@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn(seed: int) -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as sk_roc

    s, lab = _scores(5000, seed=seed)
    s = np.round(s, 2)  # create ties
    for backend in BACKENDS:
        fpr, tpr, thr = metrics.roc_curve(s, lab, backend=backend)  # type: ignore[arg-type]
        f2, t2, th2 = sk_roc(lab, s, drop_intermediate=False)
        np.testing.assert_allclose(fpr, f2, atol=1e-12)
        np.testing.assert_allclose(tpr, t2, atol=1e-12)
        np.testing.assert_allclose(thr[1:], th2[1:])
        assert metrics.summarize(s, lab, backend=backend).auc == pytest.approx(  # type: ignore[arg-type]
            roc_auc_score(lab, s), abs=1e-12
        )
