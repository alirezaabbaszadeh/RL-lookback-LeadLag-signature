from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Iterable, Sequence

from leadlag.driver import service as driver_service
from leadlag.driver.logging import render_dry_run_summary, render_status_summary
from leadlag.cli.errors import emit_error
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.training.run_scenario import _merge_extends, _validate_scenario_schema


@dataclass
class _CLIContext:
    command: str
    results_root: Path
    discovered_scenarios: Sequence[Path] | None = None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
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


def _handle_validate(args: argparse.Namespace, context: _CLIContext) -> int:
    try:
        scenario_path = driver_service.resolve_scenario_reference(args.validate)
        config = _merge_extends(scenario_path)
        _validate_scenario_schema(config, scenario=scenario_path.stem)
    except Exception as exc:  # pragma: no cover - exercised in tests
        emit_error(
            args,
            code="scenario_validation_failed",
            message=f"Validation failed for '{args.validate}'",
            details={"scenario": args.validate, "error": str(exc), "valid": False},
        )
        return 1

    emit_formatted_output(
        args,
        data={
            "scenario": scenario_path.stem,
            "path": str(scenario_path),
            "valid": True,
        },
        text=f"Scenario '{scenario_path.stem}' is valid ({scenario_path})",
        message="Scenario validation succeeded.",
        pretty=True,
        command=context.command,
    )
    return 0


def _handle_status(args: argparse.Namespace, context: _CLIContext) -> int:
    runs = driver_service.collect_status(context.results_root)
    status_render = render_status_summary(context.results_root, runs)
    emit_formatted_output(
        args,
        data=status_render.data,
        text=status_render.text,
        message="Run status summary.",
        errors=status_render.errors,
        success=status_render.success,
        pretty=True,
        command=context.command,
    )
    return 0


def _handle_list(args: argparse.Namespace, context: _CLIContext) -> int:
    if context.discovered_scenarios is None:  # pragma: no cover - guard for misuse
        raise ValueError("No scenarios available in context")

    scenario_names = [path.stem for path in context.discovered_scenarios]
    emit_formatted_output(
        args,
        data={"scenarios": scenario_names},
        text="\n".join(scenario_names),
        message="Available scenarios listed.",
        pretty=True,
        command=context.command,
    )
    return 0


def _handle_execute(
    args: argparse.Namespace,
    context: _CLIContext,
    logger: Logger,
    execution_options: driver_service.ExecutionOptions,
) -> int:
    if context.discovered_scenarios is None:  # pragma: no cover - guard for misuse
        raise ValueError("No scenarios available in context")

    results_root = context.results_root
    discovered_scenarios = list(context.discovered_scenarios)

    if args.scenarios:
        selected, errors = driver_service.resolve_scenario_references(args.scenarios)
        if errors:
            emit_error(
                args,
                code="invalid_scenarios",
                message="One or more scenarios not found",
                details={"errors": errors, "requested": list(args.scenarios)},
            )
            return 1
        selected = [path.resolve() for path in selected]
    else:
        selected = driver_service.filter_scenarios(
            discovered_scenarios, args.include, args.exclude
        )
    if args.max_scenarios is not None:
        selected = selected[: max(args.max_scenarios, 0)]

    if not selected:
        emit_error(
            args,
            code="no_scenarios_matched",
            message="No scenarios match the provided filters.",
            details={
                "include": args.include,
                "exclude": args.exclude,
                "results_root": str(results_root),
            },
        )
        return 1

    logger.info(
        "Discovered %s scenario(s); %s selected after filtering.",
        len(discovered_scenarios),
        len(selected),
    )
    selected_names = [sc.stem for sc in selected]
    summary_payload_base = {
        "selected": selected_names,
        "results_root": str(results_root),
    }
    execution = driver_service.execute_scenarios(
        selected,
        execution_options,
        logger=logger,
    )

    if execution.dry_run:
        summary_payload = driver_service.DriverSummary(
            selected=selected_names,
            results_root=str(results_root),
            summary=[],
            aggregate=None,
            dry_run=True,
            dry_run_entries=execution.dry_run_entries,
        )
        dry_render = render_dry_run_summary(summary_payload)
        emit_formatted_output(
            args,
            data=dry_render.data,
            text=dry_render.text,
            message="Dry-run completed.",
            pretty=True,
            command=context.command,
        )
        return execution.exit_code

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

    artifacts = {"aggregate": str(aggregate_path)} if aggregate_path else None
    success = exit_code == 0 and not failures and not aborted
    message = (
        "LeadLag scenarios completed."
        if success
        else "LeadLag scenarios completed with errors."
    )

    text_lines = [f"Results root: {results_root}"]
    if summary:
        text_lines.append("Scenario outcomes:")
        for row in summary:
            details = row.output or row.error or row.reason or ""
            if details:
                text_lines.append(f"  - {row.scenario}: {row.status} ({details})")
            else:
                text_lines.append(f"  - {row.scenario}: {row.status}")
    if aggregate_path:
        text_lines.append(f"Aggregate: {aggregate_path}")

    final_payload = driver_service.DriverSummary(
        selected=summary_payload_base["selected"],
        results_root=summary_payload_base["results_root"],
        summary=summary,
        aggregate=str(aggregate_path) if aggregate_path else None,
        dry_run=False,
    ).to_payload()
    final_data = {**summary_payload_base, **{k: v for k, v in final_payload.items() if k not in summary_payload_base}}

    emit_formatted_output(
        args,
        data=final_data,
        text="\n".join(text_lines),
        message=message,
        artifacts=artifacts,
        errors=errors_list or None,
        success=success,
        pretty=True,
        command=context.command,
    )

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = getattr(args, "_leadlag_command", "leadlag")

    results_root = Path(args.results_root).expanduser().resolve()
    context = _CLIContext(command=command, results_root=results_root)

    if args.validate:
        return _handle_validate(args, context)

    if args.status:
        return _handle_status(args, context)

    scenarios = driver_service.discover_scenarios()
    if not scenarios:
        emit_error(
            args,
            code="no_scenarios_available",
            message="No scenarios found in packaged scenarios (leadlag.configs.scenarios)",
            details={"results_root": str(results_root)},
        )
        return 1

    context.discovered_scenarios = list(scenarios)

    if args.list:
        return _handle_list(args, context)

    (
        prepared_root,
        logger,
        execution_options,
        command_string,
    ) = driver_service.prepare_execution(args)
    context.results_root = prepared_root
    context.command = command_string

    return _handle_execute(args, context, logger, execution_options)



if __name__ == "__main__":
    raise SystemExit(main())
