from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from leadlag.driver import dto, execution, execution_setup, selection


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


def test_load_scenario_context_skips_existing(tmp_path: Path, scenario_file: Path) -> None:
    results_root = tmp_path / "results"
    previous = results_root / "alpha_20240101_000000"
    previous.mkdir(parents=True)
    (previous / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root, skip_existing=True)
    context, result, error = execution.load_scenario_context(
        scenario_file, options, results_root, logger
    )

    assert context is None
    assert error is None
    assert result == dto.ScenarioResult(
        scenario="alpha", status="skipped", runner=None, reason="existing_results"
    )


def test_load_scenario_context_failure(tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_root = tmp_path / "results"

    def failing_merge(_path: Path) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(execution, "_merge_extends", failing_merge)
    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root)
    context, result, error = execution.load_scenario_context(
        scenario_file, options, results_root, logger
    )

    assert context is None
    assert result is not None and result.status == "load_failed"
    assert error == {
        "code": "scenario_load_failed",
        "message": "Scenario load failed",
        "details": {"scenario": "alpha", "error": "boom"},
    }


def test_run_scenario_with_context_success(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    options = execution_setup.ExecutionOptions(results_root=results_root)

    def fake_merge(path: Path) -> dict[str, Any]:
        assert path == scenario_file
        return {"run": {"run_name": "alpha"}, "dynamic": {}}

    monkeypatch.setattr(execution, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    context, result, error = execution.load_scenario_context(
        scenario_file, options, results_root, DummyLogger()
    )

    assert context is not None and result is None and error is None

    run_dir = results_root / "alpha_20240101_000000"
    run_dir.mkdir(parents=True)

    def fake_execute(_runner: str, _sc_path: Path, _root: Path) -> Path:
        return run_dir

    monkeypatch.setattr(execution, "_execute_runner", fake_execute)

    scenario_result, execution_error = execution.run_scenario_with_context(
        context, DummyLogger()
    )

    assert execution_error is None
    assert scenario_result.status == "success"
    assert scenario_result.output == str(run_dir)


def test_run_scenario_with_context_failure(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    context = dto.ScenarioExecutionContext(
        scenario="alpha",
        path=scenario_file,
        results_root=results_root,
        config={"run": {"run_name": "alpha"}},
        runner="dynamic",
    )

    def failing_execute(*_args, **_kwargs) -> Path:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(execution, "_execute_runner", failing_execute)

    result, error = execution.run_scenario_with_context(context, DummyLogger())

    assert result.status == "error"
    assert error == {
        "code": "scenario_execution_failed",
        "message": "Scenario execution failed",
        "details": {"scenario": "alpha", "error": "kaboom"},
    }


def test_record_outcome_updates_collections() -> None:
    summary: list[dto.ScenarioResult] = []
    errors: list[dict[str, object]] = []
    result = dto.ScenarioResult(scenario="alpha", status="success", runner="dynamic")

    had_error = execution.record_outcome(summary, errors, result, None)
    assert summary == [result]
    assert errors == []
    assert had_error is False

    err_result = dto.ScenarioResult(scenario="beta", status="error", runner="dynamic")
    error_entry = {"code": "boom"}
    had_error = execution.record_outcome(summary, errors, err_result, error_entry)
    assert had_error is True
    assert errors == [error_entry]


def test_scenario_executor_dry_run_logs_entries(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution,
        "_merge_extends",
        lambda *_args, **_kwargs: pytest.fail("_merge_extends should not run during dry-run"),
    )
    monkeypatch.setattr(
        execution,
        "_execute_runner",
        lambda *_args, **_kwargs: pytest.fail("runner should not run during dry-run"),
    )

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(
        results_root=scenario_file.parent / "results", dry_run=True
    )
    executor = execution.ScenarioExecutor(options, logger=logger)
    result = executor.run([scenario_file])

    assert result.dry_run is True
    assert result.summary == []
    assert result.exit_code == 0
    assert [entry.name for entry in result.dry_run_entries] == ["alpha"]
    dry_messages = [msg for level, msg, _ in logger.records if level == "info"]
    assert any("[dry-run]" in msg for msg in dry_messages)


def test_scenario_executor_stop_on_error_aborts(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"

    def failing_merge(_path: Path) -> dict:
        raise ValueError("bad config")

    monkeypatch.setattr(execution, "_merge_extends", failing_merge)
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(
        results_root=results_root,
        stop_on_error=True,
    )
    executor = execution.ScenarioExecutor(options, logger=logger)
    result = executor.run([scenario_file])

    assert result.exit_code == 1
    assert result.aborted is True
    assert result.summary == [
        dto.ScenarioResult(
            scenario="alpha",
            status="load_failed",
            runner=None,
            error="bad config",
        )
    ]
    assert result.errors[0]["code"] == "scenario_load_failed"


def test_scenario_executor_aggregation_uses_injected_callable(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    runner_calls: list[tuple[str, Path, Path]] = []
    aggregate_calls: list[str] = []

    def fake_merge(path: Path) -> dict:
        assert path == scenario_file
        return {"run": {"run_name": "alpha"}, "analysis": {}, "data": {}, "dynamic": {}}

    monkeypatch.setattr(execution, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)

    def fake_execute(runner: str, sc_path: Path, root: Path) -> Path:
        runner_calls.append((runner, sc_path, root))
        out_dir = root / "alpha_20240101_000000"
        out_dir.mkdir(parents=True)
        (out_dir / "summary.csv").write_text("metric,mean\n", encoding="utf-8")
        return out_dir

    monkeypatch.setattr(execution, "_execute_runner", fake_execute)

    def fake_aggregate(root_str: str) -> Path:
        aggregate_calls.append(root_str)
        return Path(root_str) / "aggregate"

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root)
    executor = execution.ScenarioExecutor(options, logger=logger, aggregator=fake_aggregate)
    result = executor.run([scenario_file])

    assert runner_calls == [("dynamic", scenario_file, results_root)]
    assert aggregate_calls == [str(results_root)]
    assert result.aggregate == results_root / "aggregate"
    assert result.errors == []
    assert result.exit_code == 0
    assert result.aborted is False


def test_trigger_aggregation_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    summary = [
        dto.ScenarioResult(scenario="alpha", status="success", runner="dynamic"),
        dto.ScenarioResult(scenario="beta", status="skipped", runner=None),
    ]

    aggregate_calls: list[Path] = []

    def fake_aggregate(root: str) -> Path:
        aggregate_calls.append(Path(root))
        return Path(root) / "aggregate"

    monkeypatch.setattr(execution, "aggregate", fake_aggregate)

    aggregate_path, errors, exit_code, aborted = execution.trigger_aggregation(
        summary,
        execution_setup.ExecutionOptions(results_root=results_root),
        results_root,
        DummyLogger(),
    )

    assert aggregate_path == results_root / "aggregate"
    assert errors == []
    assert exit_code == 0
    assert aborted is False
    assert aggregate_calls == [results_root]


def test_trigger_aggregation_failure_stop_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    summary = [
        dto.ScenarioResult(scenario="alpha", status="success", runner="dynamic"),
    ]

    def failing_aggregate(*_args, **_kwargs) -> Path:
        raise RuntimeError("agg")

    monkeypatch.setattr(execution, "aggregate", failing_aggregate)

    aggregate_path, errors, exit_code, aborted = execution.trigger_aggregation(
        summary,
        execution_setup.ExecutionOptions(results_root=results_root, stop_on_error=True),
        results_root,
        DummyLogger(),
    )

    assert aggregate_path is None
    assert errors and errors[0]["code"] == "aggregation_failed"
    assert exit_code == 1
    assert aborted is True


def test_execute_scenarios_success_runs_and_aggregates(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    runner_calls: list[tuple[str, Path, Path]] = []
    aggregate_calls: list[Path] = []

    def fake_merge(path: Path) -> dict:
        assert path == scenario_file
        return {"run": {"run_name": "alpha"}, "analysis": {}, "data": {}, "dynamic": {}}

    monkeypatch.setattr(execution, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)

    def fake_execute(runner: str, sc_path: Path, root: Path) -> Path:
        runner_calls.append((runner, sc_path, root))
        out_dir = root / "alpha_20240101_000000"
        out_dir.mkdir(parents=True)
        (out_dir / "summary.csv").write_text("metric,mean\n", encoding="utf-8")
        return out_dir

    monkeypatch.setattr(execution, "_execute_runner", fake_execute)

    def fake_aggregate(root_str: str) -> Path:
        root_path = Path(root_str)
        aggregate_calls.append(root_path)
        return root_path / "aggregate"

    monkeypatch.setattr(execution, "aggregate", fake_aggregate)

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root)
    result = execution.execute_scenarios([scenario_file], options, logger=logger)

    assert runner_calls == [("dynamic", scenario_file, results_root)]
    assert aggregate_calls == [results_root]
    assert result.aggregate == results_root / "aggregate"
    assert result.summary == [
        dto.ScenarioResult(
            scenario="alpha",
            status="success",
            runner="dynamic",
            output=str(results_root / "alpha_20240101_000000"),
        )
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

    monkeypatch.setattr(execution, "_merge_extends", failing_merge)
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(execution, "aggregate", lambda *_args, **_kwargs: pytest.fail("aggregate"))

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(
        results_root=results_root,
        stop_on_error=True,
    )
    result = execution.execute_scenarios([scenario_file], options, logger=logger)

    assert result.exit_code == 1
    assert result.aborted is True
    assert result.summary == [
        dto.ScenarioResult(
            scenario="alpha",
            status="load_failed",
            runner=None,
            error="bad config",
        )
    ]
    assert result.errors[0]["code"] == "scenario_load_failed"


def test_execute_scenarios_dry_run_logs_entries(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution,
        "_merge_extends",
        lambda *_args, **_kwargs: pytest.fail("_merge_extends should not run during dry-run"),
    )
    monkeypatch.setattr(
        execution,
        "_execute_runner",
        lambda *_args, **_kwargs: pytest.fail("runner should not run during dry-run"),
    )

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=scenario_file.parent / "results", dry_run=True)
    result = execution.execute_scenarios([scenario_file], options, logger=logger)

    assert result.dry_run is True
    assert result.summary == []
    assert result.exit_code == 0
    assert [entry.name for entry in result.dry_run_entries] == ["alpha"]
    dry_messages = [msg for level, msg, _ in logger.records if level == "info"]
    assert any("[dry-run]" in msg for msg in dry_messages)


def test_execute_scenarios_aggregation_failure_recorded(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"

    monkeypatch.setattr(
        execution, "_merge_extends", lambda *_args, **_kwargs: {"run": {"run_name": "alpha"}}
    )
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        execution,
        "_execute_runner",
        lambda *_args, **_kwargs: (results_root / "alpha_run").mkdir(parents=True) or results_root / "alpha_run",
    )

    def failing_aggregate(_root: str) -> Path:
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(execution, "aggregate", failing_aggregate)

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root)
    result = execution.execute_scenarios([scenario_file], options, logger=logger)

    assert result.aggregate is None
    assert result.errors and result.errors[0]["code"] == "aggregation_failed"
    assert result.exit_code == 0
    assert result.aborted is False


def test_structured_models_serialize_to_expected_payloads(
    tmp_path: Path, scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"

    monkeypatch.setattr(
        execution,
        "_merge_extends",
        lambda *_args, **_kwargs: {
            "run": {"run_name": "alpha"},
            "dynamic": {},
        },
    )
    monkeypatch.setattr(execution, "_validate_scenario_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        execution,
        "_execute_runner",
        lambda *_args, **_kwargs: (results_root / "alpha_run").mkdir(parents=True)
        or results_root / "alpha_run",
    )
    monkeypatch.setattr(execution, "aggregate", lambda root: Path(root) / "aggregate")

    logger = DummyLogger()
    options = execution_setup.ExecutionOptions(results_root=results_root)
    result = execution.execute_scenarios([scenario_file], options, logger=logger)

    summary_payloads = [entry.to_payload() for entry in result.summary]
    assert summary_payloads == [
        {
            "scenario": "alpha",
            "status": "success",
            "runner": "dynamic",
            "output": str(results_root / "alpha_run"),
        }
    ]
    assert list(summary_payloads[0].keys()) == [
        "scenario",
        "status",
        "runner",
        "output",
    ]

    driver_summary = dto.DriverSummary(
        selected=["alpha"],
        results_root=str(results_root),
        summary=result.summary,
        aggregate=str(results_root / "aggregate"),
        dry_run=False,
    )
    driver_payload = driver_summary.to_payload()
    assert list(driver_payload.keys()) == [
        "selected",
        "results_root",
        "summary",
        "aggregate",
        "dry_run",
    ]
    assert driver_payload["summary"] == summary_payloads

    status_entries = selection.collect_status(results_root)
    status_payloads = [entry.to_payload() for entry in status_entries]
    assert all("run_dir" in payload and "status" in payload for payload in status_payloads)
