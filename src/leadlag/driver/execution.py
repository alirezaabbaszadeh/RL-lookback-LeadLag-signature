"""Scenario execution and aggregation flow."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from leadlag.evaluation.aggregate import aggregate

from .aggregation import AggregationCoordinator, trigger_aggregation
from .dto import ExecutionResult, ScenarioResult
from .dry_run import DryRunExecutor
from .execution_setup import ExecutionOptions
from .outcome import OutcomeRecorder
from .scenario_runner import (
    ScenarioRunner,
    _execute_runner,
    _pick_runner,
    load_scenario_context,
    run_scenario_with_context,
)


class _NullLogger:
    """Fallback logger that accepts the structured logging interface."""

    def info(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def warning(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def exception(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None


class ScenarioExecutor:
    """Coordinator responsible for executing scenarios under a given configuration."""

    def __init__(
        self,
        options: ExecutionOptions,
        *,
        logger=None,
        aggregator: Callable[[str | Path], Path | None] | None = None,
    ) -> None:
        self.options = options
        self.logger = logger or _NullLogger()
        self.results_root = options.results_root

        self.recorder = OutcomeRecorder()
        self.summary = self.recorder.summary
        self.errors = self.recorder.errors

        self.exit_code = 0
        self.aborted = False
        self.aggregate_path: Path | None = None

        self._dry_runner = DryRunExecutor(self.logger)
        self._scenario_runner = ScenarioRunner(
            options,
            self.logger,
            self.results_root,
            self.recorder,
        )
        self._aggregation = AggregationCoordinator(
            options,
            self.logger,
            self.results_root,
            aggregator=aggregator,
        )

    def run(self, selected_paths: Sequence[Path]) -> ExecutionResult:
        """Execute the provided *selected_paths* and report consolidated results."""

        if self.options.dry_run:
            return self._dry_runner.execute(selected_paths)

        self.results_root.mkdir(parents=True, exist_ok=True)

        runner_exit, runner_aborted = self._scenario_runner.run(selected_paths)
        self.exit_code = max(self.exit_code, runner_exit)
        self.aborted = self.aborted or runner_aborted

        if not self.aborted:
            (
                aggregate_path,
                aggregate_errors,
                aggregate_exit,
                aggregate_aborted,
            ) = self._aggregation.run(self.summary)
            self.aggregate_path = aggregate_path
            if aggregate_errors:
                self.errors.extend(aggregate_errors)
            self.exit_code = max(self.exit_code, aggregate_exit)
            self.aborted = self.aborted or aggregate_aborted

        return ExecutionResult(
            summary=self.summary,
            errors=self.errors,
            aggregate=self.aggregate_path,
            exit_code=self.exit_code,
            aborted=self.aborted,
            dry_run=False,
        )


def execute_scenarios(
    selected: Sequence[Path],
    options: ExecutionOptions,
    *,
    logger=None,
) -> ExecutionResult:
    """Execute *selected* scenarios with the provided options."""

    executor = ScenarioExecutor(options, logger=logger)
    return executor.run(selected)


def record_outcome(
    summary: list[ScenarioResult],
    errors: list[dict[str, object]],
    result: ScenarioResult,
    error: dict[str, object] | None,
) -> bool:
    """Backwards compatible shim that delegates to :class:`OutcomeRecorder`."""

    recorder = OutcomeRecorder(summary, errors)
    return recorder.record(result, error)


__all__ = [
    "aggregate",
    "ExecutionResult",
    "OutcomeRecorder",
    "ScenarioExecutor",
    "execute_scenarios",
    "load_scenario_context",
    "record_outcome",
    "run_scenario_with_context",
    "trigger_aggregation",
    "_execute_runner",
    "_pick_runner",
]
