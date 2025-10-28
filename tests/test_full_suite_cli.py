from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadlag.pipelines import run_full_suite


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


@pytest.fixture
def full_suite_mocks(monkeypatch):
    monkeypatch.setattr(run_full_suite, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_suite, "dependency_preflight", lambda skip, logger: {})
    monkeypatch.setattr(run_full_suite, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_suite, "get_logger", lambda *args, **kwargs: DummyLogger())
    monkeypatch.setattr(
        run_full_suite,
        "inspect_scenario",
        lambda name: run_full_suite.ScenarioRequirements(
            name=name,
            requires_signature=False,
            requires_sb3=False,
            requires_sb3_contrib=False,
        ),
    )
    monkeypatch.setattr(
        run_full_suite.hydra_main,
        "_load_scenario_cfg",
        lambda name: {"runner": "scenario", "analysis": {"method": "ccf"}},
    )
    monkeypatch.setattr(run_full_suite.hydra_main, "validate_scenario_cfg", lambda cfg: None)


def test_full_suite_json_summary(tmp_path, full_suite_mocks, capsys):
    output_root = tmp_path / "suite"
    exit_code = run_full_suite.main(
        [
            "--output-root",
            str(output_root),
            "--skip-baseline",
            "--skip-ablation",
            "--skip-report",
            "--skip-meta-offline",
            "--skip-audit",
            "--skip-schema-check",
            "--format",
            "json",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["output_root"] == str(output_root.resolve())
    assert data["dry_run"] is True
