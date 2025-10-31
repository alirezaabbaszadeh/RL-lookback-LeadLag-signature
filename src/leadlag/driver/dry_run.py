"""Dry-run executor responsible for logging selections."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .dto import ExecutionResult, ScenarioSelection


class DryRunExecutor:
    """Emit dry-run entries for the provided scenario selections."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def execute(self, selected: Sequence[Path]) -> ExecutionResult:
        dry_entries: list[ScenarioSelection] = []
        for sc in selected:
            sc_path = Path(sc)
            try:
                display = sc_path.relative_to(Path.cwd())
            except ValueError:
                display = sc_path
            self.logger.info(f"[dry-run] {display}")
            dry_entries.append(
                ScenarioSelection(
                    name=sc_path.stem,
                    display=str(display),
                    path=str(sc_path),
                )
            )

        return ExecutionResult(dry_run=True, dry_run_entries=dry_entries)


__all__ = ["DryRunExecutor"]
