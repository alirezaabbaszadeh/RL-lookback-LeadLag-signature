"""Scenario discovery and resolution utilities for the LeadLag driver."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Sequence

from leadlag.utils.resources import resolve_path

__all__ = [
    "discover_scenarios",
    "resolve_scenario_reference",
    "resolve_scenario_references",
]


def discover_scenarios() -> list[Path]:
    """Return all discoverable scenario configuration files."""

    scenarios: list[Path] = []
    local_dir = Path("configs") / "scenarios"
    if local_dir.exists():
        local_scenarios = sorted(local_dir.glob("*.yaml"))
        if local_scenarios:
            return [p.resolve() for p in local_scenarios]

    try:
        base = resources.files("leadlag.configs").joinpath("scenarios")
        for entry in base.iterdir():
            if entry.name.endswith(".yaml"):
                resolved = resolve_path("leadlag.configs", f"scenarios/{entry.name}")
                if resolved:
                    scenarios.append(resolved)
    except (ModuleNotFoundError, AttributeError):
        pass

    if not scenarios:
        fallback_dir = resolve_path("leadlag.configs", "scenarios")
        if fallback_dir and fallback_dir.is_dir():
            scenarios.extend(sorted(fallback_dir.glob("*.yaml")))

    return sorted({path.resolve() for path in scenarios})


def resolve_scenario_reference(entry: str) -> Path:
    """Resolve a user-supplied scenario reference to a concrete path."""

    candidate = Path(entry)
    if candidate.exists():
        return candidate.resolve()

    name = candidate.name
    resource = name if name.endswith(".yaml") else f"{name}.yaml"
    resolved = resolve_path("leadlag.configs", f"scenarios/{resource}")
    if resolved is not None and resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Scenario '{entry}' not found in packaged resources or filesystem paths."
    )


def resolve_scenario_references(entries: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Resolve multiple scenario references, returning successes and failures."""

    resolved: list[Path] = []
    errors: list[str] = []
    for entry in entries:
        try:
            resolved.append(resolve_scenario_reference(entry))
        except FileNotFoundError as exc:
            errors.append(str(exc))
    return resolved, errors

