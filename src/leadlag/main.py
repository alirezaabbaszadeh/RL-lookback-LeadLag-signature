from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from types import SimpleNamespace

from leadlag.driver import dto, execution, execution_setup, scenario_registry, selection
from leadlag.driver.logging import (
    render_dry_run_summary,
    render_execution_summary,
    render_status_summary,
)
from leadlag.cli.errors import emit_error
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.cli import commands as cli_commands
from leadlag.training.run_scenario import run_scenario
from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema


driver_service = SimpleNamespace(
    DriverSummary=dto.DriverSummary,
    ExecutionOptions=execution_setup.ExecutionOptions,
    ExecutionResult=dto.ExecutionResult,
    ExecutionSetup=execution_setup.ExecutionSetup,
    RunStatusEntry=dto.RunStatusEntry,
    ScenarioExecutionContext=dto.ScenarioExecutionContext,
    ScenarioResult=dto.ScenarioResult,
    ScenarioSelection=dto.ScenarioSelection,
    aggregate=execution.aggregate,
    collect_status=selection.collect_status,
    discover_scenarios=scenario_registry.discover_scenarios,
    execute_scenarios=execution.execute_scenarios,
    filter_scenarios=selection.filter_scenarios,
    has_successful_run=selection.has_successful_run,
    load_scenario_context=execution.load_scenario_context,
    matches_filters=selection.matches_filters,
    prepare_execution=execution_setup.prepare_execution,
    record_outcome=execution.record_outcome,
    _execute_runner=execution._execute_runner,
    _merge_extends=_merge_extends,
    _validate_scenario_schema=_validate_scenario_schema,
    resolve_scenario_reference=scenario_registry.resolve_scenario_reference,
    resolve_scenario_references=scenario_registry.resolve_scenario_references,
    run_scenario=run_scenario,
    run_scenario_with_context=execution.run_scenario_with_context,
    trigger_aggregation=execution.trigger_aggregation,
)


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


class LeadLagCLI:
    """Dispatcher coordinating LeadLag CLI interactions."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.command = getattr(args, "_leadlag_command", "leadlag")
        self.results_root = Path(args.results_root).resolve()
        self._scenario_manager = cli_commands.ScenarioManager(
            driver_service.discover_scenarios
        )
        self._registry: Sequence[cli_commands.CommandSpec] | None = None

    def build_registry(self) -> Sequence[cli_commands.CommandSpec]:
        if self._registry is None:
            dependencies = cli_commands.CommandDependencies(
                driver_service=driver_service,
                scenario_manager=self._scenario_manager,
                merge_extends=_merge_extends,
                validate_scenario_schema=_validate_scenario_schema,
                render_status_summary=render_status_summary,
                render_execution_summary=render_execution_summary,
                render_dry_run_summary=render_dry_run_summary,
            )
            self._registry = cli_commands.build_command_registry(dependencies)
        return self._registry

    def dispatch(
        self, registry: Sequence[cli_commands.CommandSpec]
    ) -> CLIResult:
        context = cli_commands.CommandContext(
            args=self.args, results_root=self.results_root, command=self.command
        )
        for spec in registry:
            if spec.predicate(self.args):
                response = spec.handler(context)
                self._update_state(response)
                return self._from_response(response)
        raise RuntimeError("No matching CLI command found")

    def _update_state(self, response: cli_commands.CommandResponse) -> None:
        if response.results_root is not None:
            self.results_root = Path(response.results_root)
        if response.command is not None:
            self.command = response.command

    def _from_response(self, response: cli_commands.CommandResponse) -> CLIResult:
        payload: dict[str, object] = {}
        if response.data is not None:
            payload["data"] = response.data
        if response.text is not None:
            payload["text"] = response.text
        if response.message is not None:
            payload["message"] = response.message
        if response.errors is not None:
            payload["errors"] = response.errors
        if response.success is not None:
            payload["success"] = response.success
        if response.artifacts is not None:
            payload["artifacts"] = response.artifacts
        if response.details is not None:
            payload["details"] = response.details

        payload["pretty"] = response.pretty
        payload["command"] = response.command or self.command

        if response.code is not None:
            payload["code"] = response.code

        return CLIResult(
            exit_code=response.exit_code,
            emitter=response.emitter,
            payload=payload,
        )


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def parse_args(
    argv: Iterable[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    parser = parser or build_parser()
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
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else None
    args = parse_args(raw_argv, parser=parser)
    cli = LeadLagCLI(args)
    registry = cli.build_registry()
    result = cli.dispatch(registry)
    _emit_result(args, result)
    return result.exit_code



if __name__ == "__main__":
    raise SystemExit(main())
