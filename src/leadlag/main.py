from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Callable, Iterable, Sequence

from leadlag.driver import service as driver_service
from leadlag.driver.logging import (
    render_dry_run_summary,
    render_execution_summary,
    render_status_summary,
)
from leadlag.cli.errors import emit_error
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.training.run_scenario import _merge_extends, _validate_scenario_schema


@dataclass(frozen=True)
class CLIResult:
    exit_code: int
    emitter: str
    payload: dict[str, object]

    @classmethod
    def output(cls, exit_code: int, **payload: object) -> "CLIResult":
        return cls(exit_code=exit_code, emitter="output", payload=dict(payload))

    @classmethod
    def error(cls, exit_code: int, **payload: object) -> "CLIResult":
        return cls(exit_code=exit_code, emitter="error", payload=dict(payload))


@dataclass(frozen=True)
class CommandSpec:
    name: str
    predicate: Callable[[argparse.Namespace], bool]
    handler: Callable[["LeadLagCLI"], CLIResult]


class LeadLagCLI:
    """Dispatcher coordinating LeadLag CLI interactions."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.command = getattr(args, "_leadlag_command", "leadlag")
        self.results_root = Path(args.results_root).resolve()
        self.discovered_scenarios: Sequence[Path] | None = None

    def dispatch(self, registry: Sequence[CommandSpec]) -> CLIResult:
        for spec in registry:
            if spec.predicate(self.args):
                return spec.handler(self)
        raise RuntimeError("No matching CLI command found")

    # region Helpers
    def _success(self, exit_code: int, **payload: object) -> CLIResult:
        if "command" not in payload:
            payload["command"] = self.command
        payload.setdefault("pretty", True)
        return CLIResult.output(exit_code, **payload)

    def _ensure_scenarios(self) -> CLIResult | None:
        if self.discovered_scenarios is not None:
            return None
        scenarios = driver_service.discover_scenarios()
        if not scenarios:
            return CLIResult.error(
                1,
                code="no_scenarios_available",
                message=(
                    "No scenarios found in packaged scenarios "
                    "(leadlag.configs.scenarios)"
                ),
                details={"results_root": str(self.results_root)},
            )
        self.discovered_scenarios = list(scenarios)
        return None

    # endregion

    def validate(self) -> CLIResult:
        try:
            scenario_path = driver_service.resolve_scenario_reference(self.args.validate)
            config = _merge_extends(scenario_path)
            _validate_scenario_schema(config, scenario=scenario_path.stem)
        except Exception as exc:  # pragma: no cover - exercised in tests
            return CLIResult.error(
                1,
                code="scenario_validation_failed",
                message=f"Validation failed for '{self.args.validate}'",
                details={
                    "scenario": self.args.validate,
                    "error": str(exc),
                    "valid": False,
                },
            )

        return self._success(
            0,
            data={
                "scenario": scenario_path.stem,
                "path": str(scenario_path),
                "valid": True,
            },
            text=f"Scenario '{scenario_path.stem}' is valid ({scenario_path})",
            message="Scenario validation succeeded.",
        )

    def status(self) -> CLIResult:
        runs = driver_service.collect_status(self.results_root)
        status_render = render_status_summary(self.results_root, runs)
        return self._success(
            0,
            data=status_render.data,
            text=status_render.text,
            message="Run status summary.",
            errors=status_render.errors,
            success=status_render.success,
        )

    def list(self) -> CLIResult:
        ensured = self._ensure_scenarios()
        if ensured is not None:
            return ensured
        assert self.discovered_scenarios is not None  # for type-checkers
        scenario_names = [path.stem for path in self.discovered_scenarios]
        return self._success(
            0,
            data={"scenarios": scenario_names},
            text="\n".join(scenario_names),
            message="Available scenarios listed.",
        )

    def execute(self) -> CLIResult:
        ensured = self._ensure_scenarios()
        if ensured is not None:
            return ensured
        assert self.discovered_scenarios is not None  # for type-checkers

        prepared_root, logger, execution_options, command_string = (
            driver_service.prepare_execution(self.args)
        )
        self.results_root = prepared_root
        self.command = command_string

        discovered_scenarios = list(self.discovered_scenarios)
        args = self.args

        if args.scenarios:
            selected, errors = driver_service.resolve_scenario_references(args.scenarios)
            if errors:
                return CLIResult.error(
                    1,
                    code="invalid_scenarios",
                    message="One or more scenarios not found",
                    details={"errors": errors, "requested": list(args.scenarios)},
                )
            selected = [path.resolve() for path in selected]
        else:
            selected = driver_service.filter_scenarios(
                discovered_scenarios, args.include, args.exclude
            )

        if args.max_scenarios is not None:
            selected = selected[: max(args.max_scenarios, 0)]

        if not selected:
            return CLIResult.error(
                1,
                code="no_scenarios_matched",
                message="No scenarios match the provided filters.",
                details={
                    "include": args.include,
                    "exclude": args.exclude,
                    "results_root": str(self.results_root),
                },
            )

        logger.info(
            "Discovered %s scenario(s); %s selected after filtering.",
            len(discovered_scenarios),
            len(selected),
        )

        selected_names = [sc.stem for sc in selected]
        execution = driver_service.execute_scenarios(
            selected,
            execution_options,
            logger=logger,
        )

        if execution.dry_run:
            summary_payload = driver_service.DriverSummary(
                selected=selected_names,
                results_root=str(self.results_root),
                summary=[],
                aggregate=None,
                dry_run=True,
                dry_run_entries=execution.dry_run_entries,
            )
            dry_render = render_dry_run_summary(summary_payload)
            return self._success(
                execution.exit_code,
                data=dry_render.data,
                text=dry_render.text,
                message="Dry-run completed.",
            )

        summary = execution.summary
        errors_list = execution.errors
        aggregate_path = execution.aggregate
        exit_code = execution.exit_code
        aborted = execution.aborted

        failures = [row for row in summary if row.status not in {"success", "skipped"}]
        if failures:
            logger.warning(
                "Some scenarios did not complete successfully",
                context={"failures": len(failures)},
            )
            if args.stop_on_error and exit_code == 0:
                exit_code = 1

        execution_render = render_execution_summary(
            self.results_root,
            summary=summary,
            aggregate=aggregate_path,
            selected=selected_names,
            errors=errors_list,
            exit_code=exit_code,
            aborted=aborted,
        )

        return self._success(
            exit_code,
            data=execution_render.data,
            text=execution_render.text,
            message=execution_render.message,
            artifacts=execution_render.artifacts,
            errors=execution_render.errors,
            success=execution_render.success,
        )


def build_parser_and_registry() -> tuple[argparse.ArgumentParser, Sequence[CommandSpec]]:
    parser = argparse.ArgumentParser(
        description="Run configured LeadLag scenarios and aggregate results.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Directory where scenario outputs and aggregates are stored "
            "(default: LEADLAG_RESULTS_ROOT or 'results')."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Only run scenarios whose filename contains any of the provided substrings.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Skip scenarios whose filename contains any of the provided substrings.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Maximum number of scenarios to execute after filtering.",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "scenario", "dynamic", "rl"],
        default="auto",
        help="Force a specific runner or let it auto-detect based on config blocks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected scenarios without executing them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately if a scenario fails to load or execute.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available packaged scenarios and exit.",
    )
    add_format_flags(parser, default="text")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Explicit scenario names or YAML paths to run.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report run status under the results root and exit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenarios that already have a successful run in the results root.",
    )
    parser.add_argument(
        "--validate",
        help="Validate a scenario configuration (name or path) and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log-path",
        help="Optional path for the driver log file (defaults to <results-root>/main.log).",
    )

    registry: tuple[CommandSpec, ...] = (
        CommandSpec("validate", lambda args: bool(args.validate), LeadLagCLI.validate),
        CommandSpec("status", lambda args: bool(args.status), LeadLagCLI.status),
        CommandSpec("list", lambda args: bool(args.list), LeadLagCLI.list),
        CommandSpec("execute", lambda _args: True, LeadLagCLI.execute),
    )
    return parser, registry


def parse_args(
    argv: Iterable[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    parser = parser or build_parser_and_registry()[0]
    raw_argv = list(argv) if argv is not None else None
    args = parser.parse_args(raw_argv if raw_argv is not None else None)
    finalize_format_args(args, remove_in="0.2.0")
    if args.results_root is None:
        args.results_root = os.environ.get("LEADLAG_RESULTS_ROOT", "results")
    command_string = "leadlag"
    if raw_argv:
        command_string = "leadlag " + " ".join(raw_argv)
    setattr(args, "_leadlag_command", command_string)
    return args


def _emit_result(args: argparse.Namespace, result: CLIResult) -> None:
    if result.emitter == "error":
        emit_error(args, **result.payload)
    elif result.emitter == "output":
        emit_formatted_output(args, **result.payload)
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown emitter '{result.emitter}'")


def main(argv: Sequence[str] | None = None) -> int:
    parser, registry = build_parser_and_registry()
    raw_argv = list(argv) if argv is not None else None
    args = parse_args(raw_argv, parser=parser)
    cli = LeadLagCLI(args)
    result = cli.dispatch(registry)
    _emit_result(args, result)
    return result.exit_code



if __name__ == "__main__":
    raise SystemExit(main())
