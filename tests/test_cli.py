"""CLI integration tests."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from privacy_attacks_lab.cli import main


def test_epsilon_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["epsilon", "--q", "0.01", "--sigma", "1.0", "--steps", "1000", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["epsilon_prv"] < out["epsilon_rdp"]
    assert set(out) >= {"epsilon_rdp", "epsilon_prv", "epsilon_gdp"}


def test_epsilon_epochs_and_single_accountant(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["epsilon", "--q", "0.01", "--sigma", "1.0", "--epochs", "10", "--accountant", "rdp"])
        == 0
    )
    text = capsys.readouterr().out
    assert "steps" in text and "1000" in text and "epsilon_prv" not in text


def test_epsilon_needs_steps() -> None:
    with pytest.raises(SystemExit):
        main(["epsilon", "--q", "0.01", "--sigma", "1.0"])


def test_calibrate(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["calibrate", "--q", "0.01", "--steps", "1000", "--target-epsilon", "2", "--json"])
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert 0.5 < out["sigma"] < 3


def test_max_tpr_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["max-tpr", "--epsilon", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"epsilon", "delta", "max_tpr@0.001", "max_tpr@0.01", "max_tpr@0.1"}


def test_errors_return_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["epsilon", "--q", "2", "--sigma", "1.0", "--steps", "10"]) == 2
    assert "error" in capsys.readouterr().err


def test_module_entrypoint() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "privacy_attacks_lab", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "0.1.0" in res.stdout
