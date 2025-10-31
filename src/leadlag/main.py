from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

from leadlag.driver import service as driver_service
from leadlag.cli.errors import emit_error
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.training.run_scenario import _merge_extends, _validate_scenario_schema


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
    args = parser.parse_args(list(argv) if argv is not None else None)
    finalize_format_args(args, remove_in="0.2.0")
    if args.results_root is None:
        args.results_root = os.environ.get("LEADLAG_RESULTS_ROOT", "results")
    return args




def _format_status_text(
    runs: Sequence[driver_service.RunStatusEntry], results_root: Path
) -> str:
    if not runs:
        return f"No runs found under {results_root}"
    lines = []
    for entry in runs:
        status = entry.status or "unknown"
        run_dir = entry.run_dir
        scenario = entry.scenario or "<unknown>"
        lines.append(f"{status:>10}  {scenario}  {run_dir}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = "leadlag"
    if argv:
        command = "leadlag " + " ".join(argv)

    results_root = Path(args.results_root).resolve()

    if args.validate:
        try:
            scenario_path = driver_service.resolve_scenario_reference(args.validate)
            config = _merge_extends(scenario_path)
            _validate_scenario_schema(config, scenario=scenario_path.stem)
        except Exception as exc:
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
            command=command,
        )
        return 0

    if args.status:
        runs = driver_service.collect_status(results_root)
        status_text = _format_status_text(runs, results_root)
        runs_payload = [entry.to_payload() for entry in runs]
        errors = None
        success = True
        if not runs:
            errors = [{"code": "no_runs", "message": "No runs found."}]
            success = False
        emit_formatted_output(
            args,
            data={"results_root": str(results_root), "runs": runs_payload},
            text=status_text,
            message="Run status summary.",
            errors=errors,
            success=success,
            pretty=True,
            command=command,
        )
        return 0

    scenarios = driver_service.discover_scenarios()
    if not scenarios:
        emit_error(
            args,
            code="no_scenarios_available",
            message="No scenarios found in packaged scenarios (leadlag.configs.scenarios)",
            details={"results_root": str(results_root)},
        )
        return 1
    discovered_scenarios = list(scenarios)

    scenario_names = [path.stem for path in discovered_scenarios]
    if args.list:
        emit_formatted_output(
            args,
            data={"scenarios": scenario_names},
            text="\n".join(scenario_names),
            message="Available scenarios listed.",
            pretty=True,
            command=command,
        )
        return 0

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

    results_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path).resolve() if args.log_path else results_root / "main.log"
    setup_logging(log_path, level=args.log_level.upper(), context={"module": "driver"})
    logger = get_logger("leadlag.main", context={"results_root": results_root})

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
        driver_service.ExecutionOptions(
            results_root=results_root,
            runner_preference=args.runner,
            skip_existing=args.skip_existing,
            stop_on_error=args.stop_on_error,
            dry_run=args.dry_run,
        ),
        logger=logger,
    )

    if execution.dry_run:
        dry_text = ["Selected scenarios:"] + [f"  - {name}" for name in selected_names]
        summary_payload = driver_service.DriverSummary(
            selected=selected_names,
            results_root=str(results_root),
            summary=[],
            aggregate=None,
            dry_run=True,
            dry_run_entries=execution.dry_run_entries,
        )
        emit_formatted_output(
            args,
            data={
                **summary_payload_base,
                **{
                    key: value
                    for key, value in summary_payload.to_payload().items()
                    if key not in summary_payload_base
                },
            },
            text="\n".join(dry_text),
            message="Dry-run completed.",
            pretty=True,
            command=command,
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
        command=command,
    )

    return exit_code



if __name__ == "__main__":
    raise SystemExit(main())
