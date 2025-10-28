from __future__ import annotations

import argparse
import json
import os
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from leadlag.evaluation.aggregate import aggregate
from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.training.run_scenario import _merge_extends, _validate_scenario_schema, run_scenario
from leadlag.utils.resources import resolve_path
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.cli.errors import emit_error


def discover_scenarios() -> list[Path]:
    scenarios: list[Path] = []
    local_dir = Path("configs") / "scenarios"
    if local_dir.exists():
        local_scenarios = sorted(local_dir.glob("*.yaml"))
        if local_scenarios:
            return [p.resolve() for p in local_scenarios]

    try:
        base = resources.files("leadlag.configs").joinpath("scenarios")
        for entry in base.iterdir():
            if entry.name.endswith(".yaml"):
                resolved = resolve_path("leadlag.configs", f"scenarios/{entry.name}")
                if resolved:
                    scenarios.append(resolved)
    except (ModuleNotFoundError, AttributeError):
        pass

    if not scenarios:
        fallback_dir = resolve_path("leadlag.configs", "scenarios")
        if fallback_dir and fallback_dir.is_dir():
            scenarios.extend(sorted(fallback_dir.glob("*.yaml")))

    return sorted({path.resolve() for path in scenarios})


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


def _match_filters(name: str, include: Iterable[str] | None, exclude: Iterable[str] | None) -> bool:
    if include:
        if not any(token.lower() in name.lower() for token in include):
            return False
    if exclude:
        if any(token.lower() in name.lower() for token in exclude):
            return False
    return True


def _pick_runner(preference: str, config: Dict[str, object]) -> str:
    if preference in {"scenario", "dynamic", "rl"}:
        return preference

    if "dynamic" in config:
        return "dynamic"
    if "rl" in config:
        return "rl"
    return "scenario"


def _execute_runner(runner: str, scenario_path: Path, results_root: Path) -> Path:
    if runner == "scenario":
        return run_scenario(str(scenario_path), str(results_root))

    if runner == "dynamic":
        try:
            from leadlag.training.run_dynamic_baselines import run_dynamic
        except ImportError as exc:  # pragma: no cover - optional dependency path
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(
                "Dynamic baseline runner unavailable. Install optional dependencies for dynamic "
                f"baselines (missing module: {missing})."
            ) from exc
        return run_dynamic(str(scenario_path), str(results_root))

    if runner == "rl":
        try:
            from leadlag.training.run_rl import run_rl
        except ImportError as exc:  # pragma: no cover - optional dependency path
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(
                "RL runner unavailable. Install the RL extras (pip install -r requirements-rl.txt) "
                f"(missing module: {missing})."
            ) from exc
        return run_rl(str(scenario_path), str(results_root))

    raise ValueError(f"Unknown runner '{runner}'")


def _resolve_scenario_arg(entry: str) -> Path:
    candidate = Path(entry)
    if candidate.exists():
        return candidate.resolve()

    name = candidate.name
    resource = name
    if not resource.endswith(".yaml"):
        resource = f"{resource}.yaml"
    resolved = resolve_path("leadlag.configs", f"scenarios/{resource}")
    if resolved is not None and resolved.exists():
        return resolved
    raise FileNotFoundError(
        f"Scenario '{entry}' not found in packaged resources or filesystem paths."
    )


def _has_successful_run(run_name: str, results_root: Path) -> bool:
    if not results_root.exists():
        return False
    prefix = f"{run_name}_"
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            if (child / "summary.csv").exists():
                return True
    return False


def _collect_status(results_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    if not results_root.exists():
        return runs

    for child in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        if child.name == "aggregate":
            runs.append(
                {
                    "run_dir": str(child),
                    "status": "aggregate",
                    "path": str(child),
                }
            )
            continue

        entry: dict[str, object] = {"run_dir": str(child)}
        metadata_path = child / "run_metadata.json"
        summary_path = child / "summary.csv"

        scenario_name: str | None = None
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                config_path = meta.get("config_path")
                if isinstance(config_path, str) and config_path:
                    scenario_name = Path(config_path).stem
                scenario_name = scenario_name or meta.get("scenario") or meta.get("run_name")
            except Exception:
                scenario_name = None
        if scenario_name:
            entry["scenario"] = scenario_name

        if summary_path.exists():
            entry["status"] = "success"
            entry["summary_path"] = str(summary_path)
        elif metadata_path.exists():
            entry["status"] = "incomplete"
            entry["metadata_path"] = str(metadata_path)
        else:
            entry["status"] = "empty"

        runs.append(entry)

    return runs


def _format_status_text(runs: list[dict[str, object]], results_root: Path) -> str:
    if not runs:
        return f"No runs found under {results_root}"
    lines = []
    for entry in runs:
        status = entry.get("status", "unknown")
        run_dir = entry.get("run_dir", "")
        scenario = entry.get("scenario", "<unknown>")
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
            scenario_path = _resolve_scenario_arg(args.validate)
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
        runs = _collect_status(results_root)
        status_text = _format_status_text(runs, results_root)
        errors = None
        success = True
        if not runs:
            errors = [{"code": "no_runs", "message": "No runs found."}]
            success = False
        emit_formatted_output(
            args,
            data={"results_root": str(results_root), "runs": runs},
            text=status_text,
            message="Run status summary.",
            errors=errors,
            success=success,
            pretty=True,
            command=command,
        )
        return 0

    scenarios = discover_scenarios()
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
        explicit: list[Path] = []
        errors: list[str] = []
        for entry in args.scenarios:
            try:
                explicit.append(_resolve_scenario_arg(entry))
            except FileNotFoundError as exc:
                errors.append(str(exc))
        if errors:
            emit_error(
                args,
                code="invalid_scenarios",
                message="One or more scenarios not found",
                details={"errors": errors, "requested": list(args.scenarios)},
            )
            return 1
        scenarios = [path.resolve() for path in explicit]
    else:
        scenarios = discovered_scenarios

    selected = (
        scenarios
        if args.scenarios
        else [sc for sc in scenarios if _match_filters(sc.stem, args.include, args.exclude)]
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

    if args.dry_run:
        for sc in selected:
            sc_path = Path(sc)
            try:
                display = sc_path.relative_to(Path.cwd())
            except ValueError:
                display = sc_path
            logger.info(f"[dry-run] {display}")
        dry_text = ["Selected scenarios:"] + [f"  - {name}" for name in selected_names]
        emit_formatted_output(
            args,
            data={**summary_payload_base, "summary": [], "aggregate": None, "dry_run": True},
            text="\n".join(dry_text),
            message="Dry-run completed.",
            pretty=True,
            command=command,
        )
        return 0

    summary: list[dict[str, object]] = []
    errors_list: list[dict[str, object]] = []
    aggregate_path: Path | None = None
    exit_code = 0
    aborted = False

    for sc in selected:
        name = sc.stem
        if args.skip_existing and _has_successful_run(name, results_root):
            logger.info(
                "Skipping scenario with existing successful run",
                context={"scenario": name},
            )
            summary.append(
                {
                    "scenario": name,
                    "status": "skipped",
                    "runner": None,
                    "reason": "existing_results",
                }
            )
            continue
        try:
            config = _merge_extends(sc)
            _validate_scenario_schema(config, scenario=name)
        except Exception as exc:
            logger.exception("Failed to load scenario config", context={"scenario": name})
            summary.append(
                {
                    "scenario": name,
                    "status": "load_failed",
                    "runner": None,
                    "error": str(exc),
                }
            )
            errors_list.append(
                {
                    "code": "scenario_load_failed",
                    "message": "Scenario load failed",
                    "details": {"scenario": name, "error": str(exc)},
                }
            )
            if args.stop_on_error:
                exit_code = 1
                aborted = True
                break
            continue

        runner = _pick_runner(args.runner, config)
        logger.info("Running scenario", context={"scenario": name, "runner": runner})
        try:
            out_dir = _execute_runner(runner, sc, results_root)
            summary.append(
                {
                    "scenario": name,
                    "status": "success",
                    "output": str(out_dir),
                    "runner": runner,
                }
            )
            logger.info("Scenario completed", context={"scenario": name, "output": out_dir})
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Scenario execution failed", context={"scenario": name})
            summary.append(
                {
                    "scenario": name,
                    "status": "error",
                    "runner": runner,
                    "error": str(exc),
                }
            )
            errors_list.append(
                {
                    "code": "scenario_execution_failed",
                    "message": "Scenario execution failed",
                    "details": {"scenario": name, "error": str(exc)},
                }
            )
            if args.stop_on_error:
                exit_code = 1
                aborted = True
                break

    if not aborted:
        successes = [row for row in summary if row.get("status") == "success"]
        if successes:
            try:
                aggregate_path = aggregate(str(results_root))
                logger.info("Aggregated comparison complete", context={"aggregate": aggregate_path})
            except Exception as exc:  # pragma: no cover - aggregate failures should not hide scenario results
                logger.exception("Aggregation failed", context={"results_root": results_root})
                errors_list.append(
                    {
                        "code": "aggregation_failed",
                        "message": "Aggregation failed",
                        "details": {"results_root": str(results_root), "error": str(exc)},
                    }
                )
                if args.stop_on_error:
                    exit_code = 1
                    aborted = True

    failures = [row for row in summary if row.get("status") not in {"success", "skipped"}]
    if failures:
        logger.warning(
            "Some scenarios did not complete successfully",
            context={"failures": len(failures)},
        )
        if args.stop_on_error and exit_code == 0:
            exit_code = 1

    if aborted and args.stop_on_error:
        exit_code = 1

    artifacts = {"aggregate": str(aggregate_path)} if aggregate_path else None
    success = exit_code == 0 and not failures and not aborted
    message = "LeadLag scenarios completed." if success else "LeadLag scenarios completed with errors."

    text_lines = [f"Results root: {results_root}"]
    if summary:
        text_lines.append("Scenario outcomes:")
        for row in summary:
            details = row.get("output") or row.get("error") or row.get("reason") or ""
            if details:
                text_lines.append(f"  - {row['scenario']}: {row['status']} ({details})")
            else:
                text_lines.append(f"  - {row['scenario']}: {row['status']}")
    if aggregate_path:
        text_lines.append(f"Aggregate: {aggregate_path}")

    final_data = {
        **summary_payload_base,
        "summary": summary,
        "aggregate": str(aggregate_path) if aggregate_path else None,
        "dry_run": False,
    }

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
