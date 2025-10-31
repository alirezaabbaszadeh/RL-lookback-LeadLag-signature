"""Utilities for configuring driver logging and rendering CLI outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from leadlag.driver.service import (
    DriverSummary,
    RunStatusEntry,
    ScenarioResult,
    ScenarioSelection,
)
from leadlag.reporting.logging_utils import get_logger, setup_logging


@dataclass(slots=True)
class StatusRender:
    """Deterministic payload for rendering run status information."""

    text: str
    data: dict[str, object]
    success: bool
    errors: list[dict[str, object]] | None


@dataclass(slots=True)
class DryRunRender:
    """Deterministic payload for rendering dry-run selections."""

    text: str
    data: dict[str, object]


@dataclass(slots=True)
class ExecutionRender:
    """Deterministic payload for rendering full execution results."""

    text: str
    data: dict[str, object]
    message: str
    success: bool
    errors: list[dict[str, object]] | None
    artifacts: dict[str, object] | None


def configure_driver_logger(
    results_root: Path,
    *,
    log_level: str,
    log_path: Path | None = None,
    logger_name: str = "leadlag.main",
    base_context: Mapping[str, object] | None = None,
):
    """Initialise logging for the driver and return a configured adapter."""

    target_path = log_path if log_path is not None else results_root / "main.log"
    setup_logging(target_path, level=log_level.upper(), context={"module": "driver"})
    context = {"results_root": str(results_root)}
    if base_context:
        context.update(base_context)
    return get_logger(logger_name, context=context)


def _format_status_lines(runs: Iterable[RunStatusEntry]) -> list[str]:
    lines: list[str] = []
    for entry in runs:
        status = entry.status or "unknown"
        run_dir = entry.run_dir
        scenario = entry.scenario or "<unknown>"
        lines.append(f"{status:>10}  {scenario}  {run_dir}")
    return lines


def render_status_summary(
    results_root: Path, runs: Sequence[RunStatusEntry]
) -> StatusRender:
    """Return deterministic text and payload for run status output."""

    lines = _format_status_lines(runs)
    if not lines:
        text = f"No runs found under {results_root}"
        success = False
        errors = [{"code": "no_runs", "message": "No runs found."}]
    else:
        text = "\n".join(lines)
        success = True
        errors = None

    payload_runs = [entry.to_payload() for entry in runs]
    data = {"results_root": str(results_root), "runs": payload_runs}
    return StatusRender(text=text, data=data, success=success, errors=errors)


def render_dry_run_summary(summary: DriverSummary) -> DryRunRender:
    """Return deterministic text and payload for dry-run mode."""

    entries: Sequence[ScenarioSelection] = summary.dry_run_entries or []
    if entries:
        lines = ["Selected scenarios:"]
        for entry in entries:
            label = entry.name or entry.display or "<unknown>"
            lines.append(f"  - {label}")
        text = "\n".join(lines)
    else:
        text = "No scenarios selected for execution."

    return DryRunRender(text=text, data=summary.to_payload())


def render_execution_summary(
    results_root: Path,
    *,
    summary: Sequence[ScenarioResult],
    aggregate: Path | None,
    selected: Sequence[str],
    errors: Sequence[dict[str, object]] | None,
    exit_code: int,
    aborted: bool,
) -> ExecutionRender:
    """Return deterministic text and payload for a completed execution."""

    text_lines = [f"Results root: {results_root}"]
    if summary:
        text_lines.append("Scenario outcomes:")
        for row in summary:
            details = row.output or row.error or row.reason or ""
            if details:
                text_lines.append(f"  - {row.scenario}: {row.status} ({details})")
            else:
                text_lines.append(f"  - {row.scenario}: {row.status}")
    if aggregate:
        text_lines.append(f"Aggregate: {aggregate}")

    failures = [row for row in summary if row.status not in {"success", "skipped"}]
    success = exit_code == 0 and not failures and not aborted
    message = (
        "LeadLag scenarios completed."
        if success
        else "LeadLag scenarios completed with errors."
    )
    artifacts = {"aggregate": str(aggregate)} if aggregate else None

    payload_summary = DriverSummary(
        selected=list(selected),
        results_root=str(results_root),
        summary=list(summary),
        aggregate=str(aggregate) if aggregate else None,
        dry_run=False,
    ).to_payload()

    return ExecutionRender(
        text="\n".join(text_lines),
        data=payload_summary,
        message=message,
        success=success,
        errors=list(errors) if errors else None,
        artifacts=artifacts,
    )


__all__ = [
    "DryRunRender",
    "ExecutionRender",
    "StatusRender",
    "configure_driver_logger",
    "render_execution_summary",
    "render_dry_run_summary",
    "render_status_summary",
]
