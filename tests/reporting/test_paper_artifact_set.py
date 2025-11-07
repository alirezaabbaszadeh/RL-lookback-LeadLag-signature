import json
from pathlib import Path

import pytest

from leadlag.reporting.paper_outputs import (
    REQUIRED_PAPER_ARTIFACTS,
    list_paper_artifacts,
    validate_paper_artifact_set,
)
from leadlag.reporting.paper_outputs import main as paper_outputs_main


def test_validate_paper_artifact_set_success(tmp_path: Path) -> None:
    for name in REQUIRED_PAPER_ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    validated = validate_paper_artifact_set(tmp_path)
    assert len(validated) == len(REQUIRED_PAPER_ARTIFACTS)


def test_validate_paper_artifact_set_missing(tmp_path: Path) -> None:
    (tmp_path / "main_results.csv").write_text("ok", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        validate_paper_artifact_set(tmp_path)


def test_list_paper_artifacts_identifies_states(tmp_path: Path) -> None:
    (tmp_path / "main_results.csv").write_text("ok", encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("extra", encoding="utf-8")

    listing = list_paper_artifacts(tmp_path)

    assert any(path.name == "main_results.csv" for path in listing.present)
    assert any(path.name == "ablations.csv" for path in listing.missing)
    assert any(path.name == "unexpected.txt" for path in listing.unexpected)


def test_paper_outputs_validate_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in REQUIRED_PAPER_ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    exit_code = paper_outputs_main(
        ["--format", "json", "validate", "--root", str(tmp_path)]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    assert len(payload["data"]["artifacts"]) == len(REQUIRED_PAPER_ARTIFACTS)


def test_paper_outputs_ls_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "main_results.csv").write_text("ok", encoding="utf-8")
    (tmp_path / "bonus.png").write_text("x", encoding="utf-8")

    exit_code = paper_outputs_main([
        "--format",
        "json",
        "ls",
        "--root",
        str(tmp_path),
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert any(entry.endswith("ablations.csv") for entry in data["missing"])
    assert any(entry.endswith("bonus.png") for entry in data["unexpected"])
