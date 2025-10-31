"""Outcome recording helpers for scenario execution."""
from __future__ import annotations

from .dto import ScenarioResult


class OutcomeRecorder:
    """Capture scenario results and associated error metadata."""

    def __init__(
        self,
        summary: list[ScenarioResult] | None = None,
        errors: list[dict[str, object]] | None = None,
    ) -> None:
        self.summary = summary if summary is not None else []
        self.errors = errors if errors is not None else []

    def record(
        self,
        result: ScenarioResult,
        error: dict[str, object] | None,
    ) -> bool:
        """Append *result* and return ``True`` when *error* is provided."""

        self.summary.append(result)
        if error is not None:
            self.errors.append(error)
            return True
        return False

    def extend(self, other: "OutcomeRecorder") -> None:
        """Merge entries from *other* into this recorder."""

        self.summary.extend(other.summary)
        self.errors.extend(other.errors)

    def __iter__(self):  # pragma: no cover - simple delegation
        return iter(self.summary)


__all__ = ["OutcomeRecorder"]
