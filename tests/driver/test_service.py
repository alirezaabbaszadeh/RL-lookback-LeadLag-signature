from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadlag.driver import service


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.records.append(("info", message, kwargs.get("context")))

    def warning(self, message: str, *args, **kwargs) -> None:
        self.records.append(("warning", message, kwargs.get("context")))

    def exception(self, message: str, *args, **kwargs) -> None:
        self.records.append(("exception", message, kwargs.get("context")))


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "alpha.yaml"
    path.write_text("run:\n  run_name: alpha\n", encoding="utf-8")
    return path


def test_execute_scenarios_success_runs_and_aggregates(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    runner_calls: list[tuple[str, Path, Path]] = []
    aggregate_calls: list[Path] = []

    def fake_merge(path: Path) -> dict:
        assert path == scenario_file
        return {"run": {"run_name": "alpha"}, "analysis": {}, "data": {}, "dynamic": {}}

    monkeypatch.setattr(service, "_merge_extends", fake_merge)
    monkeypatch.setattr(service, "_validate_scenario_schema", lambda *_args, **_kwargs: None)

    def fake_execute(runner: str, sc_path: Path, root: Path) -> Path:
        runner_calls.append((runner, sc_path, root))
        out_dir = root / "alpha_20240101_000000"
        out_dir.mkdir(parents=True)
        (out_dir / "summary.csv").write_text("metric,mean\n", encoding="utf-8")
        return out_dir

    monkeypatch.setattr(service, "_execute_runner", fake_execute)

    def fake_aggregate(root_str: str) -> Path:
        root_path = Path(root_str)
        aggregate_calls.append(root_path)
        return root_path / "aggregate"

    monkeypatch.setattr(service, "aggregate", fake_aggregate)

    logger = DummyLogger()
    options = service.ExecutionOptions(results_root=results_root)
    result = service.execute_scenarios([scenario_file], options, logger=logger)

    assert runner_calls == [("dynamic", scenario_file, results_root)]
    assert aggregate_calls == [results_root]
    assert result.aggregate == results_root / "aggregate"
    assert result.summary == [
        {
            "scenario": "alpha",
            "status": "success",
            "output": str(results_root / "alpha_20240101_000000"),
            "runner": "dynamic",
        }
    ]
    assert result.errors == []
    assert result.exit_code == 0
    assert result.aborted is False
    assert result.dry_run is False


def test_execute_scenarios_failure_stops_when_requested(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"

    def failing_merge(_path: Path) -> dict:
        raise ValueError("bad config")

    monkeypatch.setattr(service, "_merge_extends", failing_merge)
    monkeypatch.setattr(service, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "aggregate", lambda *_args, **_kwargs: pytest.fail("aggregate"))

    logger = DummyLogger()
    options = service.ExecutionOptions(
        results_root=results_root,
        stop_on_error=True,
    )
    result = service.execute_scenarios([scenario_file], options, logger=logger)

    assert result.exit_code == 1
    assert result.aborted is True
    assert result.summary == [
        {
            "scenario": "alpha",
            "status": "load_failed",
            "runner": None,
            "error": "bad config",
        }
    ]
    assert result.errors[0]["code"] == "scenario_load_failed"


def test_execute_scenarios_dry_run_logs_entries(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service,
        "_merge_extends",
        lambda *_args, **_kwargs: pytest.fail("_merge_extends should not run during dry-run"),
    )
    monkeypatch.setattr(
        service,
        "_execute_runner",
        lambda *_args, **_kwargs: pytest.fail("runner should not run during dry-run"),
    )

    logger = DummyLogger()
    options = service.ExecutionOptions(results_root=scenario_file.parent / "results", dry_run=True)
    result = service.execute_scenarios([scenario_file], options, logger=logger)

    assert result.dry_run is True
    assert result.summary == []
    assert result.exit_code == 0
    assert [entry["name"] for entry in result.dry_run_entries] == ["alpha"]
    dry_messages = [msg for level, msg, _ in logger.records if level == "info"]
    assert any("[dry-run]" in msg for msg in dry_messages)


def test_collect_status_reports_runs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    aggregate_dir = results_root / "aggregate"
    aggregate_dir.mkdir(parents=True)
    (aggregate_dir / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    success = results_root / "alpha_20240101_000000"
    success.mkdir()
    (success / "run_metadata.json").write_text(
        json.dumps({"config_path": "configs/scenarios/alpha.yaml"}), encoding="utf-8"
    )
    (success / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    empty = results_root / "beta_20240101_000010"
    empty.mkdir()

    runs = service.collect_status(results_root)

    statuses = {entry["run_dir"]: entry["status"] for entry in runs}
    assert str(aggregate_dir) in statuses and statuses[str(aggregate_dir)] == "aggregate"
    assert statuses[str(success)] == "success"
    assert statuses[str(empty)] == "empty"


def test_execute_scenarios_aggregation_failure_recorded(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"

    monkeypatch.setattr(
        service, "_merge_extends", lambda *_args, **_kwargs: {"run": {"run_name": "alpha"}}
    )
    monkeypatch.setattr(service, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_execute_runner",
        lambda *_args, **_kwargs: (results_root / "alpha_run").mkdir(parents=True) or results_root / "alpha_run",
    )

    def failing_aggregate(_root: str) -> Path:
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(service, "aggregate", failing_aggregate)

    logger = DummyLogger()
    options = service.ExecutionOptions(results_root=results_root)
    result = service.execute_scenarios([scenario_file], options, logger=logger)

    assert result.aggregate is None
    assert result.errors and result.errors[0]["code"] == "aggregation_failed"
    assert result.exit_code == 0
    assert result.aborted is False
