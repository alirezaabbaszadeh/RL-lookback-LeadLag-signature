from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from leadlag.cli import responses
from leadlag.cli.dependencies import DriverService
from leadlag.cli.responders import ExecutionResponder, DryRunResponder
from leadlag.cli.selection import SelectionResult, SelectionStatus


class NoScenariosAvailable(RuntimeError):
    """Raised when packaged scenarios cannot be discovered."""


@dataclass
class CommandContext:
    """Runtime information shared with CLI command handlers."""

    args: argparse.Namespace
    results_root: Path
    command: str


@dataclass(frozen=True)
class CommandResponse:
    """Structured outcome returned by a command handler."""

    exit_code: int
    message: str | None = None
    text: str | None = None
    data: dict[str, object] | None = None
    errors: object | None = None
    success: bool | None = None
    artifacts: object | None = None
    code: str | None = None
    details: dict[str, object] | None = None
    pretty: bool = True
    command: str | None = None
    results_root: Path | None = None

    @property
    def emitter(self) -> str:
        return "error" if self.code else "output"


class CommandHandler(Protocol):
    def __call__(self, context: CommandContext) -> CommandResponse:
        """Execute a CLI command and return a structured response."""


@dataclass(frozen=True)
class CommandSpec:
    name: str
    predicate: Callable[[argparse.Namespace], bool]
    handler: CommandHandler


class ScenarioManager:
    """Lazy discovery and caching of packaged scenarios."""

    def __init__(self, discover: Callable[[], Iterable[Path]]):
        self._discover = discover
        self._cache: list[Path] | None = None

    def ensure(self) -> Sequence[Path]:
        if self._cache is not None:
            return self._cache
        scenarios = [Path(path) for path in self._discover()]
        if not scenarios:
            raise NoScenariosAvailable
        self._cache = scenarios
        return self._cache

    def ensure_or_response(
        self,
        *,
        command: str | None,
        results_root: Path,
    ) -> tuple[Sequence[Path] | None, CommandResponse | None]:
        try:
            return self.ensure(), None
        except NoScenariosAvailable:
            return None, responses.no_scenarios_available(
                command=command,
                results_root=results_root,
            )


class ScenarioSelectionService:
    """Resolve requested scenarios and handle invalid selections."""

    def __init__(self, *, driver_service: DriverService) -> None:
        self._driver = driver_service

    def resolve(
        self,
        args: argparse.Namespace,
        discovered: Sequence[Path],
        *,
        command: str,
        results_root: Path,
    ) -> SelectionResult:
        # ``command`` and ``results_root`` are accepted for future selection
        # strategies that may need contextual information. They are not
        # required for the current implementations.
        _ = (command, results_root)
        if args.scenarios:
            selected, errors = self._driver.resolve_scenario_references(args.scenarios)
            if errors:
                return SelectionResult(
                    paths=(),
                    errors=list(errors),
                    status=SelectionStatus.INVALID,
                )
            resolved = [Path(path) for path in selected]
            resolved = [path.resolve() for path in resolved]
        else:
            resolved = [
                Path(path)
                for path in self._driver.filter_scenarios(
                    discovered,
                    args.include,
                    args.exclude,
                )
            ]

        if args.max_scenarios is not None:
            resolved = resolved[: max(args.max_scenarios, 0)]

        if not resolved:
            return SelectionResult(
                paths=(),
                errors=(),
                status=SelectionStatus.NO_MATCHES,
            )

        return SelectionResult(paths=tuple(resolved), errors=(), status=SelectionStatus.OK)


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved execution inputs produced by :class:`ExecutionSetup`."""

    command: str
    results_root: Path
    logger: object
    options: object
    selected: Sequence[Path]
    selected_names: Sequence[str]


class ScenarioDiscovery:
    """Discover packaged scenarios and surface shared failure responses."""

    def __init__(self, manager: ScenarioManager) -> None:
        self._manager = manager

    def __call__(
        self, *, command: str | None, results_root: Path
    ) -> tuple[list[Path] | None, CommandResponse | None]:
        discovered, failure = self._manager.ensure_or_response(
            command=command,
            results_root=results_root,
        )
        if failure is not None:
            return None, failure
        return [Path(path) for path in discovered], None


class ExecutionSetup:
    """Prepare execution and resolve scenario selections."""

    def __init__(
        self,
        *,
        driver_service: DriverService,
        scenario_selector: ScenarioSelectionService,
    ) -> None:
        self._driver = driver_service
        self._scenario_selector = scenario_selector

    def __call__(
        self,
        args: argparse.Namespace,
        discovered: Sequence[Path],
    ) -> tuple[ExecutionPlan | None, CommandResponse | None]:
        setup = self._driver.prepare_execution(args)
        results_root = Path(setup.results_root).resolve()
        command_string = setup.command
        logger = setup.logger
        execution_options = setup.options

        selection = self._scenario_selector.resolve(
            args,
            discovered,
            command=command_string,
            results_root=results_root,
        )
        if selection.status is SelectionStatus.INVALID:
            failure = responses.invalid_scenarios(
                errors=selection.errors,
                requested=args.scenarios,
                command=command_string,
                results_root=results_root,
            )
            return None, failure
        if selection.status is SelectionStatus.NO_MATCHES:
            failure = responses.no_matches(
                include=args.include,
                exclude=args.exclude,
                command=command_string,
                results_root=results_root,
            )
            return None, failure

        logger.info(
            "Discovered %s scenario(s); %s selected after filtering.",
            len(discovered),
            len(selection.paths),
        )

        selected = [Path(path) for path in selection.paths]
        plan = ExecutionPlan(
            command=command_string,
            results_root=results_root,
            logger=logger,
            options=execution_options,
            selected=selected,
            selected_names=[path.stem for path in selected],
        )
        return plan, None


class ExecutionResponseHandler:
    """Construct :class:`CommandResponse` objects for execution results."""

    def __init__(
        self,
        *,
        dry_run_responder: DryRunResponder,
        execution_responder: ExecutionResponder,
    ) -> None:
        self._dry_run_responder = dry_run_responder
        self._execution_responder = execution_responder

    def __call__(
        self,
        execution: object,
        *,
        plan: ExecutionPlan,
        args: argparse.Namespace,
    ) -> CommandResponse:
        if getattr(execution, "dry_run", False):
            payload = self._dry_run_responder(
                execution,
                selected=plan.selected_names,
                command=plan.command,
                results_root=plan.results_root,
            )
            return CommandResponse(**payload)

        exit_code = getattr(execution, "exit_code", 0)
        summary = getattr(execution, "summary", [])
        failures = [
            row
            for row in summary
            if getattr(row, "status", None) not in {"success", "skipped"}
        ]
        if failures:
            plan.logger.warning(
                "Some scenarios did not complete successfully",
                context={"failures": len(failures)},
            )
            if getattr(args, "stop_on_error", False) and exit_code == 0:
                exit_code = 1

        payload = self._execution_responder(
            execution,
            selected=plan.selected_names,
            command=plan.command,
            results_root=plan.results_root,
            exit_code=exit_code,
        )
        return CommandResponse(**payload)


class ValidateCommand:
    def __init__(
        self,
        *,
        resolve_scenario_reference: Callable[[str], Path],
        merge_extends: Callable[[Path], dict],
        validate_scenario_schema: Callable[[dict, str], None],
    ) -> None:
        self._resolve = resolve_scenario_reference
        self._merge_extends = merge_extends
        self._validate_schema = validate_scenario_schema

    def __call__(self, context: CommandContext) -> CommandResponse:
        args = context.args
        try:
            scenario_path = self._resolve(args.validate)
            config = self._merge_extends(scenario_path)
            self._validate_schema(config, scenario=scenario_path.stem)
        except Exception as exc:  # pragma: no cover - exercised in tests
            return CommandResponse(
                exit_code=1,
                code="scenario_validation_failed",
                message=f"Validation failed for '{args.validate}'",
                details={
                    "scenario": args.validate,
                    "error": str(exc),
                    "valid": False,
                },
            )

        return CommandResponse(
            exit_code=0,
            message="Scenario validation succeeded.",
            text=f"Scenario '{scenario_path.stem}' is valid ({scenario_path})",
            data={
                "scenario": scenario_path.stem,
                "path": str(scenario_path),
                "valid": True,
            },
        )


class StatusCommand:
    def __init__(
        self,
        *,
        collect_status: Callable[[Path], Iterable[object]],
        render_status_summary: Callable[[Path, Iterable[object]], object],
    ) -> None:
        self._collect_status = collect_status
        self._render_summary = render_status_summary

    def __call__(self, context: CommandContext) -> CommandResponse:
        runs = self._collect_status(context.results_root)
        status_render = self._render_summary(context.results_root, runs)
        return CommandResponse(
            exit_code=0,
            message="Run status summary.",
            text=getattr(status_render, "text", None),
            data=getattr(status_render, "data", None),
            errors=getattr(status_render, "errors", None),
            success=getattr(status_render, "success", None),
        )


class ListCommand:
    def __init__(self, *, scenarios: ScenarioManager) -> None:
        self._scenarios = scenarios

    def __call__(self, context: CommandContext) -> CommandResponse:
        discovered, failure = self._scenarios.ensure_or_response(
            command=context.command,
            results_root=context.results_root,
        )
        if failure is not None:
            return failure

        scenario_names = [path.stem for path in discovered]
        return CommandResponse(
            exit_code=0,
            message="Available scenarios listed.",
            text="\n".join(scenario_names),
            data={"scenarios": scenario_names},
        )


class ExecuteCommand:
    def __init__(
        self,
        *,
        driver_service,
        discovery: ScenarioDiscovery,
        execution_setup: ExecutionSetup,
        responder: ExecutionResponseHandler,
    ) -> None:
        self._driver = driver_service
        self._discover = discovery
        self._execution_setup = execution_setup
        self._responder = responder

    def __call__(self, context: CommandContext) -> CommandResponse:
        discovered, failure = self._discover(
            command=context.command,
            results_root=context.results_root,
        )
        if failure is not None:
            return failure

        plan, failure = self._execution_setup(context.args, discovered)
        if failure is not None or plan is None:
            return failure  # type: ignore[return-value]

        execution = self._driver.execute_scenarios(
            plan.selected,
            plan.options,
            logger=plan.logger,
        )

        return self._responder(
            execution,
            plan=plan,
            args=context.args,
        )


@dataclass(frozen=True)
class CommandDependencies:
    driver_service: DriverService
    scenario_manager: ScenarioManager
    merge_extends: Callable[[Path], dict]
    validate_scenario_schema: Callable[[dict, str], None]
    render_status_summary: Callable[[Path, Iterable[object]], object]
    scenario_selector: ScenarioSelectionService
    dry_run_response_builder: DryRunResponder
    execution_response_builder: ExecutionResponder


def build_command_registry(dependencies: CommandDependencies) -> tuple[CommandSpec, ...]:
    validate = ValidateCommand(
        resolve_scenario_reference=dependencies.driver_service.resolve_scenario_reference,
        merge_extends=dependencies.merge_extends,
        validate_scenario_schema=dependencies.validate_scenario_schema,
    )
    status = StatusCommand(
        collect_status=dependencies.driver_service.collect_status,
        render_status_summary=dependencies.render_status_summary,
    )
    list_command = ListCommand(scenarios=dependencies.scenario_manager)
    discovery = ScenarioDiscovery(manager=dependencies.scenario_manager)
    setup = ExecutionSetup(
        driver_service=dependencies.driver_service,
        scenario_selector=dependencies.scenario_selector,
    )
    responder = ExecutionResponseHandler(
        dry_run_responder=dependencies.dry_run_response_builder,
        execution_responder=dependencies.execution_response_builder,
    )
    execute = ExecuteCommand(
        driver_service=dependencies.driver_service,
        discovery=discovery,
        execution_setup=setup,
        responder=responder,
    )

    return (
        CommandSpec("validate", lambda args: bool(args.validate), validate),
        CommandSpec("status", lambda args: bool(args.status), status),
        CommandSpec("list", lambda args: bool(args.list), list_command),
        CommandSpec("execute", lambda _args: True, execute),
    )

