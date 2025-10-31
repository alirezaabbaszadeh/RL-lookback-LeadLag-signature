"""Factory functions for common CLI command responses."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from leadlag.cli.commands import CommandResponse


def no_scenarios_available(*, command: str | None, results_root: Path) -> "CommandResponse":
    """Return a response for when no packaged scenarios are discoverable."""
    from leadlag.cli.commands import CommandResponse  # Imported lazily to avoid cycles.

    return CommandResponse(
        exit_code=1,
        code="no_scenarios_available",
        message="No scenarios found in packaged scenarios (leadlag.configs.scenarios)",
        details={"results_root": str(results_root)},
        command=command,
        results_root=results_root,
    )


def invalid_scenarios(
    *,
    errors: Sequence[str],
    requested: Sequence[str] | None,
    command: str,
    results_root: Path,
) -> "CommandResponse":
    """Return a response describing invalid scenario selections."""
    from leadlag.cli.commands import CommandResponse  # Imported lazily to avoid cycles.

    return CommandResponse(
        exit_code=1,
        code="invalid_scenarios",
        message="One or more scenarios not found",
        details={
            "errors": list(errors),
            "requested": list(requested or []),
            "results_root": str(results_root),
        },
        command=command,
        results_root=results_root,
    )


def no_matches(
    *,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    command: str,
    results_root: Path,
) -> "CommandResponse":
    """Return a response when scenario filters do not match any results."""
    from leadlag.cli.commands import CommandResponse  # Imported lazily to avoid cycles.

    return CommandResponse(
        exit_code=1,
        code="no_scenarios_matched",
        message="No scenarios match the provided filters.",
        details={
            "include": list(include or []),
            "exclude": list(exclude or []),
            "results_root": str(results_root),
        },
        command=command,
        results_root=results_root,
    )
