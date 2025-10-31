"""Driver service utilities for LeadLag scenario execution."""

from .dto import (
    DriverSummary,
    ExecutionResult,
    RunStatusEntry,
    ScenarioResult,
    ScenarioSelection,
)
from .execution import execute_scenarios
from .execution_setup import ExecutionOptions, ExecutionSetup, prepare_execution
from .logging import (
    DryRunRender,
    StatusRender,
    configure_driver_logger,
    render_dry_run_summary,
    render_status_summary,
)
from .scenario_registry import (
    discover_scenarios,
    resolve_scenario_reference,
    resolve_scenario_references,
)
from .selection import collect_status, filter_scenarios, matches_filters

__all__ = [
    "DryRunRender",
    "DriverSummary",
    "ExecutionOptions",
    "ExecutionResult",
    "ExecutionSetup",
    "StatusRender",
    "RunStatusEntry",
    "ScenarioResult",
    "ScenarioSelection",
    "collect_status",
    "configure_driver_logger",
    "discover_scenarios",
    "execute_scenarios",
    "filter_scenarios",
    "matches_filters",
    "prepare_execution",
    "render_dry_run_summary",
    "render_status_summary",
    "resolve_scenario_reference",
    "resolve_scenario_references",
]
