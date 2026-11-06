"""HTTP service: privacy accounting, attack metrics and canary exposure over JSON.

Standard library only (``http.server``), so it runs with the base install.

GET  /health    liveness
GET  /ready     readiness (accountants imported, Rust core status)
GET  /metrics   Prometheus text: request counts and latency per route
POST /epsilon   {"q", "sigma", "steps", "delta"?, "accountants"?}  epsilon per accountant
POST /calibrate {"target_epsilon", "q", "steps", "delta"?, "accountant"?}  noise multiplier
POST /max-tpr   {"epsilon", "delta"?, "fpr"?}  TPR bound implied by (epsilon, delta)-DP
POST /roc       {"scores", "labels", "fpr"?}  AUC, balanced accuracy and TPR at low FPR
POST /exposure  {"rank", "space"} or {"canary_log_prob", "sample_log_probs", "space"}
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from privacy_attacks_lab import __version__, canary, metrics
from privacy_attacks_lab._backend import rust_available
from privacy_attacks_lab.accountants import MECHANISMS, calibrate_noise, epsilon, max_tpr

MAX_BODY = 8 * 1024 * 1024
MAX_STEPS = 10_000_000


class RequestError(ValueError):
    """A client error, answered with HTTP 400."""


def _num(body: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in body:
        if default is None:
            raise RequestError(f"missing field {key!r}")
        return default
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RequestError(f"field {key!r} must be a number")
    return float(value)


def _steps(body: dict[str, Any]) -> int:
    steps = _num(body, "steps")
    if steps != int(steps) or not 1 <= steps <= MAX_STEPS:
        raise RequestError(f"steps must be an integer in [1, {MAX_STEPS}]")
    return int(steps)


def _fprs(body: dict[str, Any]) -> list[float]:
    fprs = body.get("fpr", [0.001, 0.01, 0.1])
    if not isinstance(fprs, list) or not fprs or len(fprs) > 64:
        raise RequestError("fpr must be a non-empty list of at most 64 numbers")
    out = []
    for f in fprs:
        if isinstance(f, bool) or not isinstance(f, int | float) or not 0.0 <= f <= 1.0:
            raise RequestError("every fpr must be a number in [0, 1]")
        out.append(float(f))
    return out


def handle_epsilon(body: dict[str, Any]) -> dict[str, Any]:
    q, sigma, delta = _num(body, "q"), _num(body, "sigma"), _num(body, "delta", 1e-5)
    steps = _steps(body)
    mechs = body.get("accountants", list(MECHANISMS))
    if not isinstance(mechs, list) or any(m not in MECHANISMS for m in mechs):
        raise RequestError(f"accountants must be a list drawn from {list(MECHANISMS)}")
    out: dict[str, Any] = {"q": q, "sigma": sigma, "steps": steps, "delta": delta}
    for m in mechs:
        out[f"epsilon_{m}"] = epsilon(q, sigma, steps, delta, accountant=m)
    return out


def handle_calibrate(body: dict[str, Any]) -> dict[str, Any]:
    target, q, delta = _num(body, "target_epsilon"), _num(body, "q"), _num(body, "delta", 1e-5)
    steps = _steps(body)
    acc = body.get("accountant", "rdp")
    if acc not in MECHANISMS:
        raise RequestError(f"accountant must be one of {list(MECHANISMS)}")
    sigma = calibrate_noise(target, delta, q, steps, accountant=acc)
    return {"target_epsilon": target, "accountant": acc, "sigma": sigma, "q": q, "steps": steps}


def handle_max_tpr(body: dict[str, Any]) -> dict[str, Any]:
    eps, delta = _num(body, "epsilon"), _num(body, "delta", 1e-5)
    fprs = _fprs(body)
    bound = np.atleast_1d(max_tpr(np.asarray(fprs), eps, delta))
    return {"epsilon": eps, "delta": delta, "fpr": fprs, "max_tpr": [float(x) for x in bound]}


def handle_roc(body: dict[str, Any]) -> dict[str, Any]:
    scores, labels = body.get("scores"), body.get("labels")
    if not isinstance(scores, list) or not isinstance(labels, list):
        raise RequestError("scores and labels must be lists")
    if len(scores) != len(labels) or len(scores) < 2:
        raise RequestError("scores and labels must have the same length (at least 2)")
    try:
        s = np.asarray(scores, dtype=np.float64)
        lab = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RequestError("scores and labels must be numbers") from exc
    if not np.isin(lab, (0.0, 1.0)).all():
        raise RequestError("labels must be 0 or 1")
    summary = metrics.summarize(s, lab.astype(bool), _fprs(body))
    return {
        "n": len(s),
        "members": int(lab.sum()),
        "auc": summary.auc,
        "accuracy": summary.accuracy,
        "fpr": list(summary.fprs),
        "tpr": list(summary.tprs),
    }


def handle_exposure(body: dict[str, Any]) -> dict[str, Any]:
    space = _num(body, "space", 1e6)
    if space != int(space) or space < 2:
        raise RequestError("space must be an integer of at least 2")
    if "rank" in body:
        rank = _num(body, "rank")
        if rank != int(rank):
            raise RequestError("rank must be an integer")
        return {
            "rank": int(rank),
            "space": int(space),
            "exposure": canary.exposure(int(rank), int(space)),
        }
    sample = body.get("sample_log_probs")
    if not isinstance(sample, list) or len(sample) < 10:
        raise RequestError("give rank, or canary_log_prob with at least 10 sample_log_probs")
    est = canary.exposure_from_sample(
        _num(body, "canary_log_prob"), np.asarray(sample, dtype=np.float64), int(space)
    )
    return {"space": int(space), "samples": len(sample), **est}


ROUTES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/epsilon": handle_epsilon,
    "/calibrate": handle_calibrate,
    "/max-tpr": handle_max_tpr,
    "/roc": handle_roc,
    "/exposure": handle_exposure,
}


class Metrics:
    """Request counters and latency sums in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = {}
        self.seconds: dict[str, float] = {}

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(route, status)] = self.requests.get((route, status), 0) + 1
            self.seconds[route] = self.seconds.get(route, 0.0) + seconds

    def render(self) -> str:
        rust = str(rust_available()).lower()
        with self._lock:
            lines = [
                "# HELP pal_requests_total Requests by route and status.",
                "# TYPE pal_requests_total counter",
            ]
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f'pal_requests_total{{route="{route}",status="{status}"}} {n}')
            lines += [
                "# HELP pal_request_seconds_total Time spent serving requests by route.",
                "# TYPE pal_request_seconds_total counter",
            ]
            for route, s in sorted(self.seconds.items()):
                lines.append(f'pal_request_seconds_total{{route="{route}"}} {s:.6f}')
            lines += [
                "# HELP pal_build_info Package version and Rust core availability.",
                "# TYPE pal_build_info gauge",
                f'pal_build_info{{version="{__version__}",rust="{rust}"}} 1',
            ]
        return "\n".join(lines) + "\n"


def make_handler(stats: Metrics) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"privacy-attacks-lab/{__version__}"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _finish(self, route: str, status: int, payload: Any, t0: float, ctype: str) -> None:
            self._send(status, payload, ctype)
            stats.observe(route, status, time.perf_counter() - t0)

        def do_GET(self) -> None:
            t0 = time.perf_counter()
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._finish(path, 200, {"status": "ok"}, t0, "application/json")
            elif path == "/ready":
                body = {"status": "ready", "version": __version__, "rust_core": rust_available()}
                self._finish(path, 200, body, t0, "application/json")
            elif path == "/metrics":
                text = stats.render().encode()
                self._finish(path, 200, text, t0, "text/plain; version=0.0.4")
            else:
                self._finish("other", 404, {"error": "not found"}, t0, "application/json")

        def do_POST(self) -> None:
            t0 = time.perf_counter()
            path = self.path.split("?", 1)[0]
            handler = ROUTES.get(path)
            if handler is None:
                self._finish("other", 404, {"error": "not found"}, t0, "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise RequestError(f"body must be 1 to {MAX_BODY} bytes of JSON")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise RequestError("body must be a JSON object")
                status, payload = 200, handler(body)
            except (RequestError, ValueError) as exc:
                status, payload = 400, {"error": str(exc)}
            self._finish(path, status, payload, t0, "application/json")

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(Metrics()))


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    server = make_server(host, port)
    print(f"serving on http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
