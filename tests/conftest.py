"""Shared fixtures and markers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_OPACUS = importlib.util.find_spec("opacus") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch (uv sync --extra torch)")
needs_opacus = pytest.mark.skipif(not HAS_OPACUS, reason="needs opacus (validation group)")
needs_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="needs scikit-learn (validation group)")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
