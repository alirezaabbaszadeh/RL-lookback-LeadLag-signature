from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from leadlag.driver.logging import (
    build_driver_summary,
    render_dry_run_summary,
    render_execution_summary,
    render_status_summary,
)
from leadlag.cli.errors import ERROR_UNKNOWN, emit_error
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.cli import commands as cli_commands
from leadlag.cli.dependencies import DriverService, build_driver_service
from leadlag.cli.responders import DryRunResponder, ExecutionResponder
from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema


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

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        build_driver_service: Callable[[], DriverService] = build_driver_service,
    ) -> None:
        self.args = args
        self.command = getattr(args, "_leadlag_command", "leadlag")
        self.results_root = Path(args.results_root).resolve()
        self._driver_service = build_driver_service()
        self._scenario_manager = cli_commands.ScenarioManager(
            self._driver_service.discover_scenarios
        )
        self._registry: Sequence[cli_commands.CommandSpec] | None = None

    @property
    def driver_service(self) -> DriverService:
        """Expose the driver collaborators used by this CLI instance."""

        return self._driver_service

    def build_registry(self) -> Sequence[cli_commands.CommandSpec]:
        if self._registry is None:
            scenario_selector = cli_commands.ScenarioSelectionService(
                driver_service=self._driver_service
            )
            dry_run_builder = DryRunResponder(
                build_driver_summary=build_driver_summary,
                render_dry_run_summary=render_dry_run_summary,
            )
            execution_builder = ExecutionResponder(
                render_execution_summary=render_execution_summary,
            )
            dependencies = cli_commands.CommandDependencies(
                driver_service=self._driver_service,
                scenario_manager=self._scenario_manager,
                merge_extends=_merge_extends,
                validate_scenario_schema=_validate_scenario_schema,
                render_status_summary=render_status_summary,
                scenario_selector=scenario_selector,
                dry_run_response_builder=dry_run_builder,
                execution_response_builder=execution_builder,
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
        code = result.payload.get("code", ERROR_UNKNOWN)
        message = str(result.payload.get("message", "An unexpected error occurred."))
        details = result.payload.get("details")
        emit_error(args, code=code, message=message, details=details)
    elif result.emitter == "output":
        emit_formatted_output(args, **result.payload)
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown emitter '{result.emitter}'")


def main(
    argv: Sequence[str] | None = None,
    *,
    build_driver_service: Callable[[], DriverService] = build_driver_service,
) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else None
    args = parse_args(raw_argv, parser=parser)
    cli = LeadLagCLI(args, build_driver_service=build_driver_service)
    registry = cli.build_registry()
    result = cli.dispatch(registry)
    _emit_result(args, result)
    return result.exit_code



if __name__ == "__main__":
    raise SystemExit(main())
