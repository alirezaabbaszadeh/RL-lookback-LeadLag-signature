"""Backwards-compatible facade for driver orchestration helpers."""
from __future__ import annotations

from leadlag.evaluation.aggregate import aggregate  # re-exported for legacy patches
from leadlag.training.scenario_config import (  # re-exported for legacy patches
    _merge_extends,
    _validate_scenario_schema,
)

from .dto import (
    DriverSummary,
    ExecutionResult,
    RunStatusEntry,
    ScenarioExecutionContext,
    ScenarioResult,
    ScenarioSelection,
)
from .execution import (
    _execute_runner,
    _pick_runner,
    execute_scenarios,
    load_scenario_context,
    record_outcome,
    run_scenario_with_context,
    trigger_aggregation,
)
from .outcome import OutcomeRecorder
from .execution_setup import ExecutionOptions, ExecutionSetup, prepare_execution
from .runners import (
    RunnerNotAvailableError,
    RunnerNotRegisteredError,
    clear_runner_cache,
    get_runner,
    register_runner,
)
from .scenario_registry import (
    discover_scenarios,
    resolve_scenario_reference,
    resolve_scenario_references,
)
from .selection import collect_status, filter_scenarios, has_successful_run, matches_filters

__all__ = [
    "DriverSummary",
    "ExecutionOptions",
    "ExecutionResult",
    "ExecutionSetup",
    "OutcomeRecorder",
    "RunStatusEntry",
    "ScenarioExecutionContext",
    "ScenarioResult",
    "ScenarioSelection",
    "aggregate",
    "clear_runner_cache",
    "collect_status",
    "discover_scenarios",
    "execute_scenarios",
    "filter_scenarios",
    "get_runner",
    "has_successful_run",
    "load_scenario_context",
    "matches_filters",
    "prepare_execution",
    "record_outcome",
    "register_runner",
    "resolve_scenario_reference",
    "resolve_scenario_references",
    "run_scenario_with_context",
    "RunnerNotAvailableError",
    "RunnerNotRegisteredError",
    "trigger_aggregation",
]
