from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.driver import dto
from leadlag.driver.logging import (
    build_driver_summary,
    configure_driver_logger,
    render_dry_run_summary,
    render_execution_summary,
    render_status_summary,
)


class DummyLogger:
    def __init__(self, name: str, *, context: dict[str, object]):
        self.name = name
        self.context = context


@pytest.fixture
def patch_logging(monkeypatch):
    calls = {}

    def fake_setup(path, level, context):
        calls["setup"] = {
            "path": Path(path),
            "level": level,
            "context": context,
        }

    def fake_get_logger(name, context=None):
        calls["get_logger"] = {
            "name": name,
            "context": context,
        }
        return DummyLogger(name, context=context or {})

    monkeypatch.setattr("leadlag.driver.logging.setup_logging", fake_setup)
    monkeypatch.setattr("leadlag.driver.logging.get_logger", fake_get_logger)
    return calls


def test_configure_driver_logger_default_path(tmp_path, patch_logging):
    results_root = tmp_path / "results"
    results_root.mkdir()

    logger = configure_driver_logger(results_root, log_level="info")

    setup_call = patch_logging["setup"]
    assert setup_call["path"] == results_root / "main.log"
    assert setup_call["level"] == "INFO"
    assert setup_call["context"] == {"module": "driver"}

    logger_call = patch_logging["get_logger"]
    assert logger_call["name"] == "leadlag.main"
    assert logger_call["context"] == {"results_root": str(results_root)}

    assert isinstance(logger, DummyLogger)
    assert logger.context == {"results_root": str(results_root)}


def test_configure_driver_logger_custom_path(tmp_path, patch_logging):
    results_root = tmp_path / "root"
    log_path = tmp_path / "custom.log"

    logger = configure_driver_logger(
        results_root,
        log_level="WARNING",
        log_path=log_path,
        logger_name="leadlag.tests",
        base_context={"foo": "bar"},
    )

    setup_call = patch_logging["setup"]
    assert setup_call["path"] == log_path
    assert setup_call["level"] == "WARNING"

    logger_call = patch_logging["get_logger"]
    assert logger_call["name"] == "leadlag.tests"
    assert logger_call["context"] == {
        "results_root": str(results_root),
        "foo": "bar",
    }

    assert isinstance(logger, DummyLogger)
    assert logger.context == {
        "results_root": str(results_root),
        "foo": "bar",
    }


def test_render_status_summary_no_runs(tmp_path):
    render = render_status_summary(tmp_path, [])

    assert render.text == f"No runs found under {tmp_path}"
    assert render.success is False
    assert render.errors == [{"code": "no_runs", "message": "No runs found."}]
    assert render.data == {"results_root": str(tmp_path), "runs": []}


def test_render_status_summary_with_runs(tmp_path):
    runs = [
        dto.RunStatusEntry(
            run_dir="results/run_a",
            status="success",
            scenario="alpha",
            summary_path="results/run_a/summary.csv",
        ),
        dto.RunStatusEntry(
            run_dir="results/aggregate",
            status="aggregate",
            path="results/aggregate",
        ),
    ]

    render = render_status_summary(tmp_path, runs)

    assert render.success is True
    assert render.errors is None
    assert render.data["runs"] == [entry.to_payload() for entry in runs]
    assert render.text.splitlines() == [
        "   success  alpha  results/run_a",
        " aggregate  <unknown>  results/aggregate",
    ]


def test_render_dry_run_summary(tmp_path):
    entries = [
        dto.ScenarioSelection(
            name="alpha",
            display="configs/alpha.yaml",
            path="/abs/configs/alpha.yaml",
        ),
        dto.ScenarioSelection(
            name="beta",
            display="beta.yaml",
            path="/abs/configs/beta.yaml",
        ),
    ]
    execution = dto.ExecutionResult(dry_run=True, dry_run_entries=entries)
    summary = build_driver_summary(
        [entry.name for entry in entries], tmp_path, execution
    )

    render = render_dry_run_summary(summary)

    assert render.data == summary.to_payload()
    assert render.text.splitlines() == [
        "Selected scenarios:",
        "  - alpha",
        "  - beta",
    ]


def test_render_execution_summary_success(tmp_path):
    aggregate_path = tmp_path / "agg.json"
    summary = [
        dto.ScenarioResult(
            scenario="alpha",
            status="success",
            runner="auto",
            output="done",
        )
    ]

    execution = dto.ExecutionResult(
        summary=summary,
        errors=[],
        aggregate=aggregate_path,
        exit_code=0,
        aborted=False,
        dry_run=False,
    )

    render = render_execution_summary(
        tmp_path,
        execution=execution,
        selected=["alpha"],
    )

    assert render.success is True
    assert render.message == "LeadLag scenarios completed."
    assert render.artifacts == {"aggregate": str(aggregate_path)}
    assert render.errors is None
    assert render.data == build_driver_summary(
        ["alpha"], tmp_path, execution
    ).to_payload()
    assert render.text.splitlines() == [
        f"Results root: {tmp_path}",
        "Scenario outcomes:",
        "  - alpha: success (done)",
        f"Aggregate: {aggregate_path}",
    ]


def test_render_execution_summary_with_errors(tmp_path):
    errors = [{"code": "failed", "message": "boom"}]
    summary = [
        dto.ScenarioResult(
            scenario="beta",
            status="failed",
            runner="auto",
            error="boom",
        )
    ]

    execution = dto.ExecutionResult(
        summary=summary,
        errors=errors,
        aggregate=None,
        exit_code=2,
        aborted=False,
        dry_run=False,
    )

    render = render_execution_summary(
        tmp_path,
        execution=execution,
        selected=["beta"],
    )

    assert render.success is False
    assert render.message == "LeadLag scenarios completed with errors."
    assert render.errors == errors
    assert render.artifacts is None
    assert render.data == build_driver_summary(
        ["beta"], tmp_path, execution
    ).to_payload()
    assert render.text.splitlines() == [
        f"Results root: {tmp_path}",
        "Scenario outcomes:",
        "  - beta: failed (boom)",
    ]
