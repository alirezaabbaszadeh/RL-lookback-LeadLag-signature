"""Aggregation helpers used by scenario execution."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from leadlag.evaluation.aggregate import aggregate

from .dto import ScenarioResult
from .execution_setup import ExecutionOptions


class AggregationCoordinator:
    """Coordinate aggregation over successful scenario runs."""

    def __init__(
        self,
        options: ExecutionOptions,
        logger,
        results_root: Path,
        *,
        aggregator: Callable[[str | Path], Path | None] | None = None,
    ) -> None:
        self.options = options
        self.logger = logger
        self.results_root = results_root
        self._aggregator = aggregator or aggregate

    def run(
        self, summary: Sequence[ScenarioResult]
    ) -> tuple[Path | None, list[dict[str, object]], int, bool]:
        """Execute aggregation and return error details when present."""

        successes = [row for row in summary if row.status == "success"]
        if not successes:
            return None, [], 0, False

        try:
            aggregate_path = self._aggregator(str(self.results_root))
            self.logger.info(
                "Aggregated comparison complete", context={"aggregate": aggregate_path}
            )
            return aggregate_path, [], 0, False
        except Exception as exc:  # pragma: no cover
            self.logger.exception(
                "Aggregation failed", context={"results_root": self.results_root}
            )
            error_entry = {
                "code": "aggregation_failed",
                "message": "Aggregation failed",
                "details": {
                    "results_root": str(self.results_root),
                    "error": str(exc),
                },
            }
            exit_code = 1 if self.options.stop_on_error else 0
            aborted = self.options.stop_on_error
            return None, [error_entry], exit_code, aborted


def trigger_aggregation(
    summary: Sequence[ScenarioResult],
    options: ExecutionOptions,
    results_root: Path,
    logger,
    *,
    aggregator: Callable[[str | Path], Path | None] | None = None,
) -> tuple[Path | None, list[dict[str, object]], int, bool]:
    """Run aggregation when appropriate and report any errors."""

    coordinator = AggregationCoordinator(
        options,
        logger,
        results_root,
        aggregator=aggregator,
    )
    return coordinator.run(summary)


__all__ = ["AggregationCoordinator", "trigger_aggregation"]
