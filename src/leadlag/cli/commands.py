from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from leadlag.cli.dependencies import DriverService


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
            return None, CommandResponse(
                exit_code=1,
                code="no_scenarios_available",
                message=(
                    "No scenarios found in packaged scenarios "
                    "(leadlag.configs.scenarios)"
                ),
                details={"results_root": str(results_root)},
                command=command,
                results_root=results_root,
            )


class ScenarioSelectionService:
    """Resolve requested scenarios and handle invalid selections."""

    def __init__(self, *, driver_service: DriverService) -> None:
        self._driver = driver_service

    def _failure_invalid_request(
        self,
        *,
        errors: Sequence[str],
        requested: Sequence[str] | None,
        command: str,
        results_root: Path,
    ) -> CommandResponse:
        return CommandResponse(
            exit_code=1,
            code="invalid_scenarios",
            message="One or more scenarios not found",
            details={"errors": list(errors), "requested": list(requested or [])},
            command=command,
            results_root=results_root,
        )

    def _failure_no_matches(
        self,
        *,
        include: Sequence[str] | None,
        exclude: Sequence[str] | None,
        command: str,
        results_root: Path,
    ) -> CommandResponse:
        return CommandResponse(
            exit_code=1,
            code="no_scenarios_matched",
            message="No scenarios match the provided filters.",
            details={
                "include": include,
                "exclude": exclude,
                "results_root": str(results_root),
            },
            command=command,
            results_root=results_root,
        )

    def resolve(
        self,
        args: argparse.Namespace,
        discovered: Sequence[Path],
        *,
        command: str,
        results_root: Path,
    ) -> tuple[list[Path] | None, CommandResponse | None]:
        if args.scenarios:
            selected, errors = self._driver.resolve_scenario_references(args.scenarios)
            if errors:
                return None, self._failure_invalid_request(
                    errors=errors,
                    requested=args.scenarios,
                    command=command,
                    results_root=results_root,
                )
            resolved = [path.resolve() for path in selected]
        else:
            resolved = list(
                self._driver.filter_scenarios(
                    discovered,
                    args.include,
                    args.exclude,
                )
            )

        if args.max_scenarios is not None:
            resolved = resolved[: max(args.max_scenarios, 0)]

        if not resolved:
            return None, self._failure_no_matches(
                include=args.include,
                exclude=args.exclude,
                command=command,
                results_root=results_root,
            )

        return resolved, None


class DryRunResponseBuilder:
    """Construct response payloads for dry-run executions."""

    def __init__(
        self,
        *,
        build_driver_summary: Callable[[Sequence[str], Path, object], object],
        render_dry_run_summary: Callable[[object], object],
    ) -> None:
        self._build_driver_summary = build_driver_summary
        self._render_dry_run_summary = render_dry_run_summary

    def __call__(
        self,
        execution: object,
        *,
        selected: Sequence[str],
        command: str,
        results_root: Path,
    ) -> dict[str, object]:
        summary_payload = self._build_driver_summary(selected, results_root, execution)
        dry_render = self._render_dry_run_summary(summary_payload)
        return {
            "exit_code": execution.exit_code,
            "message": "Dry-run completed.",
            "text": getattr(dry_render, "text", None),
            "data": getattr(dry_render, "data", None),
            "command": command,
            "results_root": results_root,
        }


class ExecutionResponseBuilder:
    """Construct response payloads for executed scenarios."""

    def __init__(self, *, render_execution_summary: Callable[..., object]) -> None:
        self._render_execution_summary = render_execution_summary

    def __call__(
        self,
        execution: object,
        *,
        selected: Sequence[str],
        command: str,
        results_root: Path,
        exit_code: int,
    ) -> dict[str, object]:
        execution_render = self._render_execution_summary(
            results_root,
            execution=execution,
            selected=list(selected),
        )
        return {
            "exit_code": exit_code,
            "message": getattr(execution_render, "message", None),
            "text": getattr(execution_render, "text", None),
            "data": getattr(execution_render, "data", None),
            "artifacts": getattr(execution_render, "artifacts", None),
            "errors": getattr(execution_render, "errors", None),
            "success": getattr(execution_render, "success", None),
            "command": command,
            "results_root": results_root,
        }


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
        scenarios: ScenarioManager,
        scenario_selector: ScenarioSelectionService,
        dry_run_responder: Callable[..., dict[str, object]],
        execution_responder: Callable[..., dict[str, object]],
    ) -> None:
        self._driver = driver_service
        self._scenarios = scenarios
        self._scenario_selector = scenario_selector
        self._dry_run_responder = dry_run_responder
        self._execution_responder = execution_responder

    def __call__(self, context: CommandContext) -> CommandResponse:
        discovered, failure = self._scenarios.ensure_or_response(
            command=context.command,
            results_root=context.results_root,
        )
        if failure is not None:
            return failure
        discovered = list(discovered)

        setup = self._driver.prepare_execution(context.args)
        results_root = Path(setup.results_root).resolve()
        command_string = setup.command
        logger = setup.logger
        execution_options = setup.options

        args = context.args

        selected, failure = self._scenario_selector.resolve(
            args,
            discovered,
            command=command_string,
            results_root=results_root,
        )
        if failure is not None:
            return failure

        logger.info(
            "Discovered %s scenario(s); %s selected after filtering.",
            len(discovered),
            len(selected),
        )

        selected_names = [sc.stem for sc in selected]
        execution = self._driver.execute_scenarios(
            selected,
            execution_options,
            logger=logger,
        )

        if execution.dry_run:
            payload = self._dry_run_responder(
                execution,
                selected=selected_names,
                command=command_string,
                results_root=results_root,
            )
            return CommandResponse(**payload)

        summary = execution.summary
        exit_code = execution.exit_code

        failures = [row for row in summary if row.status not in {"success", "skipped"}]
        if failures:
            logger.warning(
                "Some scenarios did not complete successfully",
                context={"failures": len(failures)},
            )
            if args.stop_on_error and exit_code == 0:
                exit_code = 1

        payload = self._execution_responder(
            execution,
            selected=selected_names,
            command=command_string,
            results_root=results_root,
            exit_code=exit_code,
        )
        return CommandResponse(**payload)


@dataclass(frozen=True)
class CommandDependencies:
    driver_service: DriverService
    scenario_manager: ScenarioManager
    merge_extends: Callable[[Path], dict]
    validate_scenario_schema: Callable[[dict, str], None]
    render_status_summary: Callable[[Path, Iterable[object]], object]
    scenario_selector: ScenarioSelectionService
    dry_run_response_builder: Callable[..., dict[str, object]]
    execution_response_builder: Callable[..., dict[str, object]]


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
    execute = ExecuteCommand(
        driver_service=dependencies.driver_service,
        scenarios=dependencies.scenario_manager,
        scenario_selector=dependencies.scenario_selector,
        dry_run_responder=dependencies.dry_run_response_builder,
        execution_responder=dependencies.execution_response_builder,
    )

    return (
        CommandSpec("validate", lambda args: bool(args.validate), validate),
        CommandSpec("status", lambda args: bool(args.status), status),
        CommandSpec("list", lambda args: bool(args.list), list_command),
        CommandSpec("execute", lambda _args: True, execute),
    )

