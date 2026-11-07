"""CLI integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import privacy_attacks_lab
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
    assert privacy_attacks_lab.__version__ in res.stdout


def _result_file(tmp_path: Path) -> Path:
    row = {"auc": 0.7, "accuracy": 0.65, "tpr@0.001": 0.03, "tpr@0.01": 0.1, "tpr@0.1": 0.3}
    data = {
        "preset": "toy",
        "num_models": 8,
        "targets": [0, 1],
        "table": {"loss": {**row, "tpr@0.001": 0.001}, "lira-online": row, "lira-online-ref": row},
    }
    path = tmp_path / "mia_toy.json"
    path.write_text(json.dumps(data))
    return path


def test_report_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(_result_file(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "toy: 8 models, 2 targets" in out
    lines = out.splitlines()
    assert lines[-1].startswith("lira-online ")
    assert "lira-online-ref" not in out
    assert "3.00%" in lines[-1]


def test_report_all_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = str(_result_file(tmp_path))
    assert main(["report", path, "--all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert set(rows) == {"loss", "lira-online", "lira-online-ref"}


def test_report_rejects_file_without_table(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("{}")
    assert main(["report", str(path)]) == 2
    assert main(["report", str(tmp_path / "missing.json")]) == 2
