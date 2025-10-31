from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from leadlag import main as cli_main
from leadlag.cli import commands as cli_commands
from leadlag.cli.dependencies import build_driver_service
from leadlag.cli.responders import DryRunResponder, ExecutionResponder
from leadlag.driver.logging import (
    build_driver_summary,
    render_dry_run_summary,
    render_execution_summary,
)


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, tuple]] = []
        self.warnings: list[tuple[str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.infos.append((message, args))

    def warning(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.warnings.append((message, kwargs.get("context")))


def _base_args(tmp_path: Path, **overrides) -> Namespace:
    base = dict(
        include=None,
        exclude=None,
        max_scenarios=None,
        runner="auto",
        skip_existing=False,
        stop_on_error=False,
        dry_run=False,
        list=False,
        status=False,
        scenarios=None,
        validate=None,
        log_level="INFO",
        log_path=None,
        results_root=str(tmp_path),
        format="text",
        json=False,
    )
    base.update(overrides)
    args = Namespace(**base)
    setattr(args, "_leadlag_command", "leadlag")
    return args


SUMMARY_ENVELOPE_KEYS = {"selected", "results_root", "summary", "aggregate", "dry_run"}


def test_cli_dispatch_converts_response(tmp_path):
    args = _base_args(tmp_path)
    cli = cli_main.LeadLagCLI(args)

    response = cli_commands.CommandResponse(
        exit_code=0,
        message="ok",
        data={"value": 1},
    )

    spec = cli_commands.CommandSpec(
        "dummy",
        lambda _args: True,
        lambda context: response,
    )

    result = cli.dispatch([spec])

    assert result.exit_code == 0
    assert result.payload["message"] == "ok"
    assert result.payload["data"] == {"value": 1}
    assert result.payload["command"] == "leadlag"


def test_validate_success(tmp_path):
    args = _base_args(tmp_path, validate="demo")
    scenario_path = tmp_path / "demo.yaml"

    validated: dict[str, object] = {}

    def _validate(config, scenario):
        validated["scenario"] = scenario

    command = cli_commands.ValidateCommand(
        resolve_scenario_reference=lambda value: scenario_path,
        merge_extends=lambda path: {"config": True},
        validate_scenario_schema=_validate,
    )
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.data["valid"] is True
    assert validated["scenario"] == "demo"


def test_validate_failure(tmp_path):
    args = _base_args(tmp_path, validate="broken")

    def _raise(_value):
        raise RuntimeError("boom")

    command = cli_commands.ValidateCommand(
        resolve_scenario_reference=_raise,
        merge_extends=cli_main._merge_extends,
        validate_scenario_schema=cli_main._validate_scenario_schema,
    )
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 1
    assert result.emitter == "error"
    assert result.code == "scenario_validation_failed"


def test_status_summary(tmp_path):
    args = _base_args(tmp_path)

    class DummyStatus:
        def __init__(self) -> None:
            self.data = {"runs": 1}
            self.text = "ok"
            self.errors = None
            self.success = True

    command = cli_commands.StatusCommand(
        collect_status=lambda root: ["status"],
        render_status_summary=lambda root, runs: DummyStatus(),
    )
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.data == {"runs": 1}


def test_list_discovers_scenarios(monkeypatch, tmp_path):
    args = _base_args(tmp_path)

    discovered = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    scenario_manager = cli_commands.ScenarioManager(lambda: discovered)
    command = cli_commands.ListCommand(scenarios=scenario_manager)
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.data == {"scenarios": ["alpha", "beta"]}
    assert "alpha\nbeta" == result.text


def test_execute_dry_run(tmp_path):
    args = _base_args(tmp_path, dry_run=True)

    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    scenario_manager = cli_commands.ScenarioManager(lambda: scenarios)
    service = build_driver_service()

    def _prepare(_args):
        prepared_root = tmp_path / "results"
        options = service.ExecutionOptions(
            results_root=prepared_root,
            runner_preference=_args.runner,
            skip_existing=_args.skip_existing,
            stop_on_error=_args.stop_on_error,
            dry_run=_args.dry_run,
        )
        return service.ExecutionSetup(
            results_root=prepared_root,
            logger=DummyLogger(),
            options=options,
            command="leadlag --dry-run",
        )

    dry_run_entries = [
        service.ScenarioSelection(
            name="alpha",
            display=str(scenarios[0].name),
            path=str(scenarios[0]),
        ),
        service.ScenarioSelection(
            name="beta",
            display=str(scenarios[1].name),
            path=str(scenarios[1]),
        ),
    ]
    execution_result = service.ExecutionResult(
        dry_run=True,
        dry_run_entries=dry_run_entries,
        exit_code=0,
    )

    service = replace(
        service,
        filter_scenarios=lambda discovered, include, exclude: list(discovered),
        prepare_execution=_prepare,
        execute_scenarios=lambda selected, options, logger=None: execution_result,
    )

    discovery = cli_commands.ScenarioDiscovery(scenario_manager)
    setup = cli_commands.ExecutionSetup(
        driver_service=service,
        scenario_selector=cli_commands.ScenarioSelectionService(driver_service=service),
    )
    responder = cli_commands.ExecutionResponseHandler(
        dry_run_responder=DryRunResponder(
            build_driver_summary=build_driver_summary,
            render_dry_run_summary=render_dry_run_summary,
        ),
        execution_responder=ExecutionResponder(
            render_execution_summary=render_execution_summary,
        ),
    )
    command = cli_commands.ExecuteCommand(
        driver_service=service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 0
    assert result.emitter == "output"
    resolved_root = (tmp_path / "results").resolve()
    expected_summary = build_driver_summary(
        ["alpha", "beta"], resolved_root, execution_result
    ).to_payload()

    assert result.message == "Dry-run completed."
    assert result.data == expected_summary
    assert SUMMARY_ENVELOPE_KEYS <= result.data.keys()
    assert result.data["dry_run"] is True
    assert result.command == "leadlag --dry-run"
    assert result.results_root == (tmp_path / "results").resolve()


def test_execute_full_run(tmp_path):
    args = _base_args(tmp_path)

    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    scenario_manager = cli_commands.ScenarioManager(lambda: scenarios)
    service = build_driver_service()

    def _prepare(_args):
        prepared_root = tmp_path / "results"
        options = service.ExecutionOptions(
            results_root=prepared_root,
            runner_preference=_args.runner,
            skip_existing=_args.skip_existing,
            stop_on_error=_args.stop_on_error,
            dry_run=_args.dry_run,
        )
        return service.ExecutionSetup(
            results_root=prepared_root,
            logger=DummyLogger(),
            options=options,
            command="leadlag",
        )

    execution_result = service.ExecutionResult(
        dry_run=False,
        dry_run_entries=[],
        summary=[
            service.ScenarioResult(
                scenario="alpha",
                status="success",
                runner="auto",
                output="done",
            )
        ],
        errors=[],
        aggregate=tmp_path / "agg.json",
        exit_code=0,
        aborted=False,
    )

    service = replace(
        service,
        filter_scenarios=lambda discovered, include, exclude: [discovered[0]],
        prepare_execution=_prepare,
        execute_scenarios=lambda selected, options, logger=None: execution_result,
    )

    discovery = cli_commands.ScenarioDiscovery(scenario_manager)
    setup = cli_commands.ExecutionSetup(
        driver_service=service,
        scenario_selector=cli_commands.ScenarioSelectionService(driver_service=service),
    )
    responder = cli_commands.ExecutionResponseHandler(
        dry_run_responder=DryRunResponder(
            build_driver_summary=build_driver_summary,
            render_dry_run_summary=render_dry_run_summary,
        ),
        execution_responder=ExecutionResponder(
            render_execution_summary=render_execution_summary,
        ),
    )
    command = cli_commands.ExecuteCommand(
        driver_service=service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )
    context = cli_commands.CommandContext(
        args=args,
        results_root=Path(args.results_root),
        command="leadlag",
    )
    result = command(context)

    assert result.exit_code == 0
    assert result.emitter == "output"
    resolved_root = (tmp_path / "results").resolve()
    expected_summary = build_driver_summary(
        ["alpha"], resolved_root, execution_result
    ).to_payload()

    assert result.message == "LeadLag scenarios completed."
    assert result.data == expected_summary
    assert SUMMARY_ENVELOPE_KEYS <= result.data.keys()
    assert result.success is True
    assert result.data["dry_run"] is False
    assert result.command == "leadlag"
