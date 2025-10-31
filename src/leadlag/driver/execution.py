"""Scenario execution and aggregation flow."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from leadlag.evaluation.aggregate import aggregate
from leadlag.training.run_scenario import run_scenario
from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema

from .dto import (
    ExecutionResult,
    ScenarioExecutionContext,
    ScenarioResult,
    ScenarioSelection,
)
from .execution_setup import ExecutionOptions
from .selection import has_successful_run


class _NullLogger:
    """Fallback logger that accepts the structured logging interface."""

    def info(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def warning(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def exception(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None


def _pick_runner(preference: str, config: dict[str, object]) -> str:
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


def execute_scenarios(
    selected: Sequence[Path],
    options: ExecutionOptions,
    *,
    logger=None,
) -> ExecutionResult:
    """Execute *selected* scenarios with the provided options."""

    if logger is None:
        logger = _NullLogger()
    results_root = options.results_root

    if options.dry_run:
        dry_entries: list[ScenarioSelection] = []
        for sc in selected:
            sc_path = Path(sc)
            try:
                display = sc_path.relative_to(Path.cwd())
            except ValueError:
                display = sc_path
            logger.info(f"[dry-run] {display}")
            dry_entries.append(
                ScenarioSelection(
                    name=sc_path.stem,
                    display=str(display),
                    path=str(sc_path),
                )
            )
        return ExecutionResult(dry_run=True, dry_run_entries=dry_entries)

    results_root.mkdir(parents=True, exist_ok=True)

    summary: list[ScenarioResult] = []
    errors_list: list[dict[str, object]] = []
    aggregate_path: Path | None = None
    exit_code = 0
    aborted = False

    for sc in selected:
        context, result, error = load_scenario_context(sc, options, results_root, logger)
        if result is not None:
            error_occurred = record_outcome(summary, errors_list, result, error)
            if error_occurred and options.stop_on_error:
                exit_code = 1
                aborted = True
                break
            continue

        if context is None:  # pragma: no cover - safety net
            continue

        run_result, run_error = run_scenario_with_context(context, logger)
        error_occurred = record_outcome(summary, errors_list, run_result, run_error)
        if error_occurred and options.stop_on_error:
            exit_code = 1
            aborted = True
            break

    if not aborted:
        (
            aggregate_path,
            aggregate_errors,
            aggregate_exit,
            aggregate_aborted,
        ) = trigger_aggregation(summary, options, results_root, logger)
        errors_list.extend(aggregate_errors)
        exit_code = max(exit_code, aggregate_exit)
        aborted = aborted or aggregate_aborted

    return ExecutionResult(
        summary=summary,
        errors=errors_list,
        aggregate=aggregate_path,
        exit_code=exit_code,
        aborted=aborted,
        dry_run=False,
    )


def load_scenario_context(
    scenario_path: Path,
    options: ExecutionOptions,
    results_root: Path,
    logger,
) -> tuple[ScenarioExecutionContext | None, ScenarioResult | None, dict[str, object] | None]:
    """Load scenario configuration and build an execution context."""

    name = scenario_path.stem
    if options.skip_existing and has_successful_run(name, results_root):
        logger.info(
            "Skipping scenario due to existing successful run",
            context={"scenario": name},
        )
        return (
            None,
            ScenarioResult(
                scenario=name,
                status="skipped",
                runner=None,
                reason="existing_results",
            ),
            None,
        )

    try:
        config = _merge_extends(scenario_path)
        _validate_scenario_schema(config, scenario=name)
    except Exception as exc:
        logger.exception("Failed to load scenario config", context={"scenario": name})
        return (
            None,
            ScenarioResult(
                scenario=name,
                status="load_failed",
                runner=None,
                error=str(exc),
            ),
            {
                "code": "scenario_load_failed",
                "message": "Scenario load failed",
                "details": {"scenario": name, "error": str(exc)},
            },
        )

    runner = _pick_runner(options.runner_preference, config)
    return (
        ScenarioExecutionContext(
            scenario=name,
            path=scenario_path,
            results_root=results_root,
            config=config,
            runner=runner,
        ),
        None,
        None,
    )


def run_scenario_with_context(
    context: ScenarioExecutionContext,
    logger,
) -> tuple[ScenarioResult, dict[str, object] | None]:
    """Execute the provided context and capture the outcome."""

    logger.info(
        "Running scenario",
        context={"scenario": context.scenario, "runner": context.runner},
    )
    try:
        out_dir = _execute_runner(context.runner, context.path, context.results_root)
        logger.info(
            "Scenario completed",
            context={"scenario": context.scenario, "output": out_dir},
        )
        return (
            ScenarioResult(
                scenario=context.scenario,
                status="success",
                runner=context.runner,
                output=str(out_dir),
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("Scenario execution failed", context={"scenario": context.scenario})
        return (
            ScenarioResult(
                scenario=context.scenario,
                status="error",
                runner=context.runner,
                error=str(exc),
            ),
            {
                "code": "scenario_execution_failed",
                "message": "Scenario execution failed",
                "details": {"scenario": context.scenario, "error": str(exc)},
            },
        )


def record_outcome(
    summary: list[ScenarioResult],
    errors: list[dict[str, object]],
    result: ScenarioResult,
    error: dict[str, object] | None,
) -> bool:
    """Record a scenario outcome and return ``True`` when an error occurred."""

    summary.append(result)
    if error is not None:
        errors.append(error)
        return True
    return False


def trigger_aggregation(
    summary: Sequence[ScenarioResult],
    options: ExecutionOptions,
    results_root: Path,
    logger,
) -> tuple[Path | None, list[dict[str, object]], int, bool]:
    """Run aggregation when appropriate and report any errors."""

    successes = [row for row in summary if row.status == "success"]
    if not successes:
        return None, [], 0, False

    try:
        aggregate_path = aggregate(str(results_root))
        logger.info(
            "Aggregated comparison complete", context={"aggregate": aggregate_path}
        )
        return aggregate_path, [], 0, False
    except Exception as exc:  # pragma: no cover
        logger.exception("Aggregation failed", context={"results_root": results_root})
        error_entry = {
            "code": "aggregation_failed",
            "message": "Aggregation failed",
            "details": {
                "results_root": str(results_root),
                "error": str(exc),
            },
        }
        exit_code = 1 if options.stop_on_error else 0
        aborted = options.stop_on_error
        return None, [error_entry], exit_code, aborted


__all__ = [
    "ExecutionResult",
    "execute_scenarios",
    "load_scenario_context",
    "record_outcome",
    "run_scenario_with_context",
    "trigger_aggregation",
]
