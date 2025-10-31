"""Reusable response builders for CLI command execution phases."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class DryRunResponder:
    """Construct response payloads for dry-run executions."""

    build_driver_summary: Callable[[Sequence[str], Path, object], object]
    render_dry_run_summary: Callable[[object], object]

    def __call__(
        self,
        execution: object,
        *,
        selected: Sequence[str],
        command: str,
        results_root: Path,
    ) -> dict[str, object]:
        summary_payload = self.build_driver_summary(selected, results_root, execution)
        dry_render = self.render_dry_run_summary(summary_payload)
        return {
            "exit_code": getattr(execution, "exit_code", 0),
            "message": "Dry-run completed.",
            "text": getattr(dry_render, "text", None),
            "data": getattr(dry_render, "data", None),
            "command": command,
            "results_root": results_root,
        }


@dataclass(frozen=True)
class ExecutionResponder:
    """Construct response payloads for executed scenarios."""

    render_execution_summary: Callable[..., object]

    def __call__(
        self,
        execution: object,
        *,
        selected: Sequence[str],
        command: str,
        results_root: Path,
        exit_code: int,
    ) -> dict[str, object]:
        execution_render = self.render_execution_summary(
            results_root,
            execution=execution,
            selected=list(selected),
        )
        return {
            "exit_code": exit_code,
            "message": getattr(execution_render, "message", None),
            "text": getattr(execution_render, "text", None),
            "data": getattr(execution_render, "data", None),
            "artifacts": getattr(execution_render, "artifacts", None),
            "errors": getattr(execution_render, "errors", None),
            "success": getattr(execution_render, "success", None),
            "command": command,
            "results_root": results_root,
        }


__all__ = ["DryRunResponder", "ExecutionResponder"]
