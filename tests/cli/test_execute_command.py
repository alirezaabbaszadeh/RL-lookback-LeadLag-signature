from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from leadlag.cli.commands import (
    CommandContext,
    CommandResponse,
    ExecuteCommand,
    ScenarioManager,
)


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, tuple]] = []
        self.warnings: list[tuple[str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - logging helper
        self.infos.append((message, args))

    def warning(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - logging helper
        self.warnings.append((message, kwargs.get("context")))


class StubSelector:
    def __init__(self, selected, failure=None):
        self._selected = selected
        self._failure = failure
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
        return self._selected, self._failure


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
    selector = StubSelector([scenarios[0]])
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

    command = ExecuteCommand(
        driver_service=driver_service,
        scenarios=scenario_manager,
        scenario_selector=selector,
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
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
    selector = StubSelector([scenarios[0]])
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

    command = ExecuteCommand(
        driver_service=driver_service,
        scenarios=scenario_manager,
        scenario_selector=selector,
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    response = command(_context(args))

    assert response.exit_code == 0
    assert response.message == "executed"
    assert response.data == {"phase": "run", "selected": ["alpha"]}
    assert exec_builder.calls and exec_builder.calls[0]["exit_code"] == 0
    assert dry_builder.calls == []


def test_execute_command_returns_selection_failure(tmp_path):
    args = _base_args(tmp_path)
    scenarios = [tmp_path / "alpha.yaml"]
    scenario_manager = ScenarioManager(lambda: scenarios)
    failure = CommandResponse(exit_code=1, code="nope", message="failure")
    selector = StubSelector(None, failure=failure)
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

    command = ExecuteCommand(
        driver_service=driver_service,
        scenarios=scenario_manager,
        scenario_selector=selector,
        dry_run_responder=dry_builder,
        execution_responder=exec_builder,
    )

    response = command(_context(args))

    assert response is failure
    assert exec_builder.calls == []
    assert dry_builder.calls == []
