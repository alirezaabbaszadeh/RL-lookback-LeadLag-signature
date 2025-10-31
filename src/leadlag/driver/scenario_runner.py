"""Scenario execution helpers and coordinator."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema

from .dto import (
    ScenarioExecutionContext,
    ScenarioResult,
)
from .execution_setup import ExecutionOptions
from .outcome import OutcomeRecorder
from .runners import (
    RunnerNotAvailableError,
    RunnerNotRegisteredError,
    get_runner,
)
from .selection import has_successful_run


def _pick_runner(preference: str, config: dict[str, object]) -> str:
    if preference in {"scenario", "dynamic", "rl"}:
        return preference

    if "dynamic" in config:
        return "dynamic"
    if "rl" in config:
        return "rl"
    return "scenario"


def _execute_runner(runner: str, scenario_path: Path, results_root: Path) -> Path:
    runner_callable = get_runner(runner)
    return runner_callable(Path(scenario_path), Path(results_root))


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
    try:
        get_runner(runner)
    except RunnerNotAvailableError as exc:
        logger.warning(
            "Runner unavailable",
            context={"scenario": name, "runner": runner, "error": str(exc)},
        )
        return (
            None,
            ScenarioResult(
                scenario=name,
                status="error",
                runner=runner,
                error=str(exc),
            ),
            {
                "code": "runner_unavailable",
                "message": "Runner unavailable",
                "details": {"scenario": name, "runner": runner, "error": str(exc)},
            },
        )
    except RunnerNotRegisteredError as exc:
        logger.exception(
            "Unknown runner selected",
            context={"scenario": name, "runner": runner},
        )
        message = str(exc)
        return (
            None,
            ScenarioResult(
                scenario=name,
                status="error",
                runner=runner,
                error=message,
            ),
            {
                "code": "runner_unknown",
                "message": "Runner unavailable",
                "details": {"scenario": name, "runner": runner, "error": message},
            },
        )

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


class ScenarioRunner:
    """Execute scenario selections and record their outcomes."""

    def __init__(
        self,
        options: ExecutionOptions,
        logger,
        results_root: Path,
        recorder: OutcomeRecorder,
        *,
        loader: Callable[..., tuple[ScenarioExecutionContext | None, ScenarioResult | None, dict[str, object] | None]] = load_scenario_context,
        runner: Callable[[ScenarioExecutionContext, object], tuple[ScenarioResult, dict[str, object] | None]] = run_scenario_with_context,
    ) -> None:
        self.options = options
        self.logger = logger
        self.results_root = results_root
        self.recorder = recorder
        self._loader = loader
        self._runner = runner

    def run(self, selected: Sequence[Path]) -> tuple[int, bool]:
        exit_code = 0
        aborted = False

        for sc in selected:
            context, result, error = self._loader(
                sc, self.options, self.results_root, self.logger
            )
            if result is not None:
                error_occurred = self.recorder.record(result, error)
                if error_occurred and self.options.stop_on_error:
                    exit_code = 1
                    aborted = True
                    break
                continue

            if context is None:  # pragma: no cover - safety net
                continue

            run_result, run_error = self._runner(context, self.logger)
            error_occurred = self.recorder.record(run_result, run_error)
            if error_occurred and self.options.stop_on_error:
                exit_code = 1
                aborted = True
                break

        return exit_code, aborted


__all__ = [
    "ScenarioRunner",
    "_execute_runner",
    "_pick_runner",
    "load_scenario_context",
    "run_scenario_with_context",
]
