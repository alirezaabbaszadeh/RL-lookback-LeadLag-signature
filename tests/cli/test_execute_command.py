from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from leadlag.cli.commands import (
    CommandContext,
    CommandResponse,
    ExecuteCommand,
    ExecutionPlan,
    ExecutionResponseHandler,
    ExecutionSetup,
    ScenarioDiscovery,
    ScenarioManager,
)
from leadlag.cli.responders import DryRunResponder, ExecutionResponder
from leadlag.cli.selection import SelectionResult, SelectionStatus


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, tuple]] = []
        self.warnings: list[tuple[str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - logging helper
        self.infos.append((message, args))

    def warning(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - logging helper
        self.warnings.append((message, kwargs.get("context")))


class StubSelector:
    def __init__(self, result: SelectionResult):
        self._result = result
        self.calls: list[dict[str, object]] = []

    def resolve(self, args, discovered, *, command, results_root):
        self.calls.append(
            {
                "args": args,
                "discovered": list(discovered),
                "command": command,
                "results_root": results_root,
            }
        )
        return self._result


class StubDryResponder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, execution, *, selected, command, results_root):
        payload = {
            "exit_code": execution.exit_code,
            "message": "dry",
            "text": "dry details",
            "data": {"phase": "dry", "selected": list(selected)},
            "command": command,
            "results_root": results_root,
        }
        self.calls.append({"execution": execution, "selected": list(selected), "payload": payload})
        return payload


class StubExecutionResponder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        execution,
        *,
        selected,
        command,
        results_root,
        exit_code,
    ) -> dict[str, object]:
        payload = {
            "exit_code": exit_code,
            "message": "executed",
            "text": "execution details",
            "data": {"phase": "run", "selected": list(selected)},
            "errors": execution.errors,
            "success": True,
            "command": command,
            "results_root": results_root,
        }
        self.calls.append(
            {
                "execution": execution,
                "selected": list(selected),
                "command": command,
                "results_root": results_root,
                "exit_code": exit_code,
                "payload": payload,
            }
        )
        return payload


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
    return Namespace(**base)


def _context(args: Namespace) -> CommandContext:
    return CommandContext(args=args, results_root=Path(args.results_root), command="leadlag")


def test_execute_command_dry_run_uses_builder(tmp_path):
    args = _base_args(tmp_path, dry_run=True)
    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    scenario_manager = ScenarioManager(lambda: scenarios)
    selector = StubSelector(
        SelectionResult(paths=[scenarios[0]], errors=(), status=SelectionStatus.OK)
    )
    dry_builder = StubDryResponder()
    exec_builder = StubExecutionResponder()
    execution = SimpleNamespace(
        dry_run=True,
        exit_code=0,
        summary=[],
        errors=None,
        aggregate=None,
        aborted=False,
    )
    executed: list[list[Path]] = []

    def prepare_execution(_args):
        return SimpleNamespace(
            results_root=tmp_path / "prepared",
            command="leadlag --dry-run",
            logger=DummyLogger(),
            options=SimpleNamespace(),
        )

    def execute_scenarios(selected, options, logger=None):
        executed.append(list(selected))
        return execution

    driver_service = SimpleNamespace(
        prepare_execution=prepare_execution,
        execute_scenarios=execute_scenarios,
    )

    discovery = ScenarioDiscovery(scenario_manager)
    setup = ExecutionSetup(
        driver_service=driver_service,
        scenario_selector=selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    command = ExecuteCommand(
        driver_service=driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    response = command(_context(args))

    assert isinstance(response, CommandResponse)
    assert response.exit_code == 0
    assert response.message == "dry"
    assert response.data == {"phase": "dry", "selected": ["alpha"]}
    assert dry_builder.calls and dry_builder.calls[0]["payload"]["command"] == "leadlag --dry-run"
    assert executed == [[scenarios[0]]]
    assert exec_builder.calls == []


def test_execute_command_success_uses_execution_responder(tmp_path):
    args = _base_args(tmp_path)
    scenarios = [tmp_path / "alpha.yaml"]
    scenario_manager = ScenarioManager(lambda: scenarios)
    selector = StubSelector(
        SelectionResult(paths=[scenarios[0]], errors=(), status=SelectionStatus.OK)
    )
    dry_builder = StubDryResponder()
    exec_builder = StubExecutionResponder()
    execution = SimpleNamespace(
        dry_run=False,
        exit_code=0,
        summary=[SimpleNamespace(status="success")],
        errors=None,
        aggregate=None,
        aborted=False,
    )

    def prepare_execution(_args):
        return SimpleNamespace(
            results_root=tmp_path / "prepared",
            command="leadlag",
            logger=DummyLogger(),
            options=SimpleNamespace(),
        )

    def execute_scenarios(selected, options, logger=None):
        return execution

    driver_service = SimpleNamespace(
        prepare_execution=prepare_execution,
        execute_scenarios=execute_scenarios,
    )

    discovery = ScenarioDiscovery(scenario_manager)
    setup = ExecutionSetup(
        driver_service=driver_service,
        scenario_selector=selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    command = ExecuteCommand(
        driver_service=driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    response = command(_context(args))

    assert response.exit_code == 0
    assert response.message == "executed"
    assert response.data == {"phase": "run", "selected": ["alpha"]}
    assert exec_builder.calls and exec_builder.calls[0]["exit_code"] == 0
    assert dry_builder.calls == []


def test_execute_command_returns_invalid_selection_failure(tmp_path):
    args = _base_args(tmp_path, scenarios=["missing"])
    scenarios = [tmp_path / "alpha.yaml"]
    scenario_manager = ScenarioManager(lambda: scenarios)
    selector = StubSelector(
        SelectionResult(
            paths=(),
            errors=("missing scenario",),
            status=SelectionStatus.INVALID,
        )
    )
    dry_builder = StubDryResponder()
    exec_builder = StubExecutionResponder()

    driver_service = SimpleNamespace(
        prepare_execution=lambda _args: SimpleNamespace(
            results_root=tmp_path / "prepared",
            command="leadlag",
            logger=DummyLogger(),
            options=SimpleNamespace(),
        ),
        execute_scenarios=lambda *args, **kwargs: SimpleNamespace(),
    )

    discovery = ScenarioDiscovery(scenario_manager)
    setup = ExecutionSetup(
        driver_service=driver_service,
        scenario_selector=selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    command = ExecuteCommand(
        driver_service=driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    response = command(_context(args))

    assert response.exit_code == 1
    assert response.code == "invalid_scenarios"
    assert response.details == {
        "errors": ["missing scenario"],
        "requested": ["missing"],
        "results_root": str((tmp_path / "prepared").resolve()),
    }
    assert response.command == "leadlag"
    assert response.results_root == (tmp_path / "prepared").resolve()
    assert exec_builder.calls == []
    assert dry_builder.calls == []


def test_execute_command_returns_no_match_failure(tmp_path):
    args = _base_args(tmp_path, include=["foo"], exclude=["bar"])
    scenarios = [tmp_path / "alpha.yaml"]
    scenario_manager = ScenarioManager(lambda: scenarios)
    selector = StubSelector(
        SelectionResult(paths=(), errors=(), status=SelectionStatus.NO_MATCHES)
    )
    dry_builder = StubDryResponder()
    exec_builder = StubExecutionResponder()

    driver_service = SimpleNamespace(
        prepare_execution=lambda _args: SimpleNamespace(
            results_root=tmp_path / "prepared",
            command="leadlag",
            logger=DummyLogger(),
            options=SimpleNamespace(),
        ),
        execute_scenarios=lambda *args, **kwargs: SimpleNamespace(),
    )

    discovery = ScenarioDiscovery(scenario_manager)
    setup = ExecutionSetup(
        driver_service=driver_service,
        scenario_selector=selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    command = ExecuteCommand(
        driver_service=driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    response = command(_context(args))

    assert response.exit_code == 1
    assert response.code == "no_scenarios_matched"
    assert response.details == {
        "include": ["foo"],
        "exclude": ["bar"],
        "results_root": str((tmp_path / "prepared").resolve()),
    }
    assert response.command == "leadlag"
    assert response.results_root == (tmp_path / "prepared").resolve()
    assert exec_builder.calls == []
    assert dry_builder.calls == []


def test_execute_command_returns_shared_response_when_no_scenarios(tmp_path):
    args = _base_args(tmp_path)
    scenario_manager = ScenarioManager(lambda: [])
    selector = StubSelector(
        SelectionResult(paths=[], errors=(), status=SelectionStatus.OK)
    )
    dry_builder = StubDryResponder()
    exec_builder = StubExecutionResponder()

    def _prepare_execution(_args):
        raise AssertionError("prepare_execution should not be called when no scenarios")

    driver_service = SimpleNamespace(
        prepare_execution=_prepare_execution,
        execute_scenarios=lambda *args, **kwargs: SimpleNamespace(),
    )

    discovery = ScenarioDiscovery(scenario_manager)
    setup = ExecutionSetup(
        driver_service=driver_service,
        scenario_selector=selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    command = ExecuteCommand(
        driver_service=driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    response = command(_context(args))

    assert response.exit_code == 1
    assert response.code == "no_scenarios_available"
    assert response.message.startswith("No scenarios found")
    assert response.details == {"results_root": str(tmp_path)}
    assert response.command == "leadlag"
    assert response.results_root == Path(args.results_root)
    assert selector.calls == []
    assert dry_builder.calls == []
    assert exec_builder.calls == []


def test_execution_response_handler_returns_dry_run_payload(tmp_path):
    logger = DummyLogger()

    def build_driver_summary(selected, results_root, execution):
        return SimpleNamespace(text="dry text", data={"selected": list(selected)})

    def render_dry_run_summary(summary):
        return summary

    def render_execution_summary(*args, **kwargs):
        raise AssertionError("Execution responder should not be called")

    handler = ExecutionResponseHandler(
        dry_run_responder=DryRunResponder(
            build_driver_summary=build_driver_summary,
            render_dry_run_summary=render_dry_run_summary,
        ),
        execution_responder=ExecutionResponder(
            render_execution_summary=render_execution_summary,
        ),
    )

    plan = ExecutionPlan(
        command="leadlag --dry-run",
        results_root=tmp_path,
        logger=logger,
        options=SimpleNamespace(),
        selected=[tmp_path / "alpha.yaml"],
        selected_names=["alpha"],
    )
    execution = SimpleNamespace(dry_run=True, exit_code=0)
    args = Namespace(stop_on_error=False)

    response = handler(execution, plan=plan, args=args)

    assert response.exit_code == 0
    assert response.message == "Dry-run completed."
    assert response.data == {"selected": ["alpha"]}


def test_execution_response_handler_handles_success(tmp_path):
    logger = DummyLogger()

    def build_driver_summary(selected, results_root, execution):
        return SimpleNamespace(text="dry text", data={"selected": list(selected)})

    def render_dry_run_summary(summary):
        return summary

    def render_execution_summary(results_root, *, execution, selected):
        return SimpleNamespace(
            message="executed",
            text="executed text",
            data={"selected": selected},
            success=True,
        )

    handler = ExecutionResponseHandler(
        dry_run_responder=DryRunResponder(
            build_driver_summary=build_driver_summary,
            render_dry_run_summary=render_dry_run_summary,
        ),
        execution_responder=ExecutionResponder(
            render_execution_summary=render_execution_summary,
        ),
    )

    plan = ExecutionPlan(
        command="leadlag",
        results_root=tmp_path,
        logger=logger,
        options=SimpleNamespace(),
        selected=[tmp_path / "alpha.yaml"],
        selected_names=["alpha"],
    )
    execution = SimpleNamespace(
        dry_run=False,
        exit_code=0,
        summary=[SimpleNamespace(status="success")],
    )
    args = Namespace(stop_on_error=False)

    response = handler(execution, plan=plan, args=args)

    assert response.exit_code == 0
    assert response.message == "executed"
    assert response.data == {"selected": ["alpha"]}
    assert logger.warnings == []


def test_execution_response_handler_adjusts_exit_code_on_failures(tmp_path):
    logger = DummyLogger()

    def build_driver_summary(selected, results_root, execution):
        return SimpleNamespace(text="dry text", data={"selected": list(selected)})

    def render_dry_run_summary(summary):
        return summary

    def render_execution_summary(results_root, *, execution, selected):
        return SimpleNamespace(
            message="executed",
            text="executed text",
            data={"selected": selected},
            errors=["boom"],
        )

    handler = ExecutionResponseHandler(
        dry_run_responder=DryRunResponder(
            build_driver_summary=build_driver_summary,
            render_dry_run_summary=render_dry_run_summary,
        ),
        execution_responder=ExecutionResponder(
            render_execution_summary=render_execution_summary,
        ),
    )

    plan = ExecutionPlan(
        command="leadlag",
        results_root=tmp_path,
        logger=logger,
        options=SimpleNamespace(),
        selected=[tmp_path / "alpha.yaml"],
        selected_names=["alpha"],
    )
    execution = SimpleNamespace(
        dry_run=False,
        exit_code=0,
        summary=[SimpleNamespace(status="failed")],
    )
    args = Namespace(stop_on_error=True)

    response = handler(execution, plan=plan, args=args)

    assert response.exit_code == 1
    assert logger.warnings == [("Some scenarios did not complete successfully", {"failures": 1})]
