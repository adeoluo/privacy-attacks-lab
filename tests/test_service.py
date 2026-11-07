"""HTTP service: routes, validation and Prometheus metrics, over a real socket."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from privacy_attacks_lab import canary, metrics, service
from privacy_attacks_lab.accountants import calibrate_noise, epsilon, max_tpr


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(base: str, path: str, body: Any, raw: bytes | None = None) -> tuple[int, dict[str, Any]]:
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_ready(base: str) -> None:
    status, body = get(base, "/health")
    assert status == 200 and json.loads(body) == {"status": "ok"}
    status, body = get(base, "/ready")
    ready = json.loads(body)
    assert status == 200 and ready["status"] == "ready" and "rust_core" in ready


def test_unknown_routes_are_404(base: str) -> None:
    assert get(base, "/nope")[0] == 404
    assert post(base, "/nope", {})[0] == 404


def test_epsilon_matches_library(base: str) -> None:
    status, out = post(base, "/epsilon", {"q": 0.01, "sigma": 1.1, "steps": 1000})
    assert status == 200
    for mech in ("rdp", "prv", "gdp"):
        assert out[f"epsilon_{mech}"] == pytest.approx(
            epsilon(0.01, 1.1, 1000, 1e-5, accountant=mech), rel=1e-12
        )


def test_epsilon_subset_of_accountants(base: str) -> None:
    status, out = post(
        base, "/epsilon", {"q": 0.01, "sigma": 1.0, "steps": 10, "accountants": ["rdp"]}
    )
    assert status == 200 and "epsilon_rdp" in out and "epsilon_prv" not in out


def test_calibrate_round_trip(base: str) -> None:
    status, out = post(base, "/calibrate", {"target_epsilon": 3.0, "q": 0.02, "steps": 500})
    assert status == 200
    assert out["sigma"] == pytest.approx(calibrate_noise(3.0, 1e-5, 0.02, 500), rel=1e-9)
    assert epsilon(0.02, out["sigma"], 500, 1e-5) <= 3.0 + 1e-6


def test_max_tpr(base: str) -> None:
    status, out = post(base, "/max-tpr", {"epsilon": 1.0, "fpr": [0.001, 0.01]})
    assert status == 200
    np.testing.assert_allclose(out["max_tpr"], max_tpr(np.array([0.001, 0.01]), 1.0, 1e-5))


def test_roc_matches_metrics(base: str) -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    scores = rng.normal(size=400) + labels
    status, out = post(base, "/roc", {"scores": scores.tolist(), "labels": labels.tolist()})
    assert status == 200
    ref = metrics.summarize(scores, labels.astype(bool))
    assert out["auc"] == pytest.approx(ref.auc)
    assert out["tpr"] == pytest.approx(list(ref.tprs))
    assert out["members"] == int(labels.sum())


def test_exposure_from_rank_and_sample(base: str) -> None:
    status, out = post(base, "/exposure", {"rank": 1, "space": 1_000_000})
    assert status == 200 and out["exposure"] == pytest.approx(canary.exposure(1, 1_000_000))
    rng = np.random.default_rng(1)
    sample = rng.normal(-20, 2, 500).tolist()
    status, out = post(base, "/exposure", {"canary_log_prob": -5.0, "sample_log_probs": sample})
    assert status == 200 and out["skewnorm"] > 5 and out["samples"] == 500


@pytest.mark.parametrize(
    ("path", "body", "fragment"),
    [
        ("/epsilon", {"sigma": 1.0, "steps": 10}, "missing field 'q'"),
        ("/epsilon", {"q": "x", "sigma": 1.0, "steps": 10}, "must be a number"),
        ("/epsilon", {"q": True, "sigma": 1.0, "steps": 10}, "must be a number"),
        ("/epsilon", {"q": 0.1, "sigma": 1.0, "steps": 1.5}, "steps must be an integer"),
        ("/epsilon", {"q": 0.1, "sigma": 1.0, "steps": 0}, "steps must be an integer"),
        ("/epsilon", {"q": 0.1, "sigma": 1.0, "steps": 10, "accountants": ["x"]}, "accountants"),
        (
            "/calibrate",
            {"target_epsilon": 1, "q": 0.1, "steps": 5, "accountant": "x"},
            "accountant",
        ),
        ("/max-tpr", {"epsilon": 1.0, "fpr": []}, "fpr must be"),
        ("/max-tpr", {"epsilon": 1.0, "fpr": [2.0]}, "every fpr"),
        ("/roc", {"scores": [1, 2], "labels": [1]}, "same length"),
        ("/roc", {"scores": "a", "labels": [1]}, "must be lists"),
        ("/roc", {"scores": [1, 2], "labels": [0, 2]}, "labels must be 0 or 1"),
        ("/roc", {"scores": ["a", 2], "labels": [0, 1]}, "must be numbers"),
        ("/exposure", {"rank": 0, "space": 10}, "rank must be"),
        ("/exposure", {"rank": 1.5, "space": 10}, "rank must be an integer"),
        ("/exposure", {"rank": 1, "space": 1}, "space must be"),
        ("/exposure", {"canary_log_prob": 1.0, "sample_log_probs": [1.0]}, "at least 10"),
        ("/epsilon", [1, 2], "JSON object"),
    ],
)
def test_bad_requests_are_400(base: str, path: str, body: Any, fragment: str) -> None:
    status, out = post(base, path, body)
    assert status == 400
    assert fragment in out["error"]


def test_malformed_json_is_400(base: str) -> None:
    status, out = post(base, "/epsilon", None, raw=b"{not json")
    assert status == 400 and "error" in out


def test_metrics_counts_requests(base: str) -> None:
    post(base, "/max-tpr", {"epsilon": 2.0})
    status, body = get(base, "/metrics")
    text = body.decode()
    assert status == 200
    assert 'pal_requests_total{route="/max-tpr",status="200"}' in text
    assert "# TYPE pal_request_seconds_total counter" in text
    assert "pal_build_info{" in text
