"""Backwards-compatible facade for driver orchestration helpers."""
from __future__ import annotations

from leadlag.evaluation.aggregate import aggregate  # re-exported for legacy patches
from leadlag.training.run_scenario import (  # re-exported for legacy patches
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
from .execution_setup import ExecutionOptions, ExecutionSetup, prepare_execution
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
    "RunStatusEntry",
    "ScenarioExecutionContext",
    "ScenarioResult",
    "ScenarioSelection",
    "aggregate",
    "collect_status",
    "discover_scenarios",
    "execute_scenarios",
    "filter_scenarios",
    "has_successful_run",
    "load_scenario_context",
    "matches_filters",
    "prepare_execution",
    "record_outcome",
    "resolve_scenario_reference",
    "resolve_scenario_references",
    "run_scenario_with_context",
    "trigger_aggregation",
]
