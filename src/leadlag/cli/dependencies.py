"""Factories and typed collaborators used by the LeadLag CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from leadlag.driver import dto, execution, execution_setup, scenario_registry, selection
from leadlag.training.run_scenario import run_scenario


@dataclass(frozen=True)
class DriverService:
    """High level driver collaborators required by the CLI layer."""

    DriverSummary: type[dto.DriverSummary]
    ExecutionOptions: type[execution_setup.ExecutionOptions]
    ExecutionResult: type[dto.ExecutionResult]
    ExecutionSetup: type[execution_setup.ExecutionSetup]
    RunStatusEntry: type[dto.RunStatusEntry]
    ScenarioExecutionContext: type[dto.ScenarioExecutionContext]
    ScenarioResult: type[dto.ScenarioResult]
    ScenarioSelection: type[dto.ScenarioSelection]
    OutcomeRecorder: type[execution.OutcomeRecorder]
    aggregate: Callable[[str | Path], Path | None]
    collect_status: Callable[[Path], Iterable[dto.RunStatusEntry]]
    discover_scenarios: Callable[[], Iterable[Path]]
    execute_scenarios: Callable[..., dto.ExecutionResult]
    filter_scenarios: Callable[[Iterable[Path], Sequence[str] | None, Sequence[str] | None], Sequence[Path]]
    has_successful_run: Callable[[Iterable[dto.RunStatusEntry]], bool]
    load_scenario_context: Callable[..., dto.ScenarioExecutionContext]
    matches_filters: Callable[[str, Sequence[str] | None, Sequence[str] | None], bool]
    prepare_execution: Callable[[object], execution_setup.ExecutionSetup]
    resolve_scenario_reference: Callable[[str], Path]
    resolve_scenario_references: Callable[[Sequence[str]], tuple[Sequence[Path], Sequence[str]]]
    run_scenario: Callable[..., dto.ScenarioResult]
    run_scenario_with_context: Callable[..., dto.ScenarioResult]
    trigger_aggregation: Callable[[str | Path], Path | None]
    _execute_runner: Callable[..., Path]


def build_driver_service() -> DriverService:
    """Construct the default driver service used by the CLI."""

    return DriverService(
        DriverSummary=dto.DriverSummary,
        ExecutionOptions=execution_setup.ExecutionOptions,
        ExecutionResult=dto.ExecutionResult,
        ExecutionSetup=execution_setup.ExecutionSetup,
        RunStatusEntry=dto.RunStatusEntry,
        ScenarioExecutionContext=dto.ScenarioExecutionContext,
        ScenarioResult=dto.ScenarioResult,
        ScenarioSelection=dto.ScenarioSelection,
        OutcomeRecorder=execution.OutcomeRecorder,
        aggregate=execution.aggregate,
        collect_status=selection.collect_status,
        discover_scenarios=scenario_registry.discover_scenarios,
        execute_scenarios=execution.execute_scenarios,
        filter_scenarios=selection.filter_scenarios,
        has_successful_run=selection.has_successful_run,
        load_scenario_context=execution.load_scenario_context,
        matches_filters=selection.matches_filters,
        prepare_execution=execution_setup.prepare_execution,
        resolve_scenario_reference=scenario_registry.resolve_scenario_reference,
        resolve_scenario_references=scenario_registry.resolve_scenario_references,
        run_scenario=run_scenario,
        run_scenario_with_context=execution.run_scenario_with_context,
        trigger_aggregation=execution.trigger_aggregation,
        _execute_runner=execution._execute_runner,
    )


__all__ = ["DriverService", "build_driver_service"]

