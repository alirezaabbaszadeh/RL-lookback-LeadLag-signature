from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.reporting.paper_outputs import (
    REQUIRED_PAPER_ARTIFACTS,
    validate_paper_artifact_set,
)


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
