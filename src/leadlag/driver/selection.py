"""Scenario discovery and selection helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .dto import RunStatusEntry


def matches_filters(name: str, include: Iterable[str] | None, exclude: Iterable[str] | None) -> bool:
    """Return ``True`` when *name* matches the include/exclude filters."""

    if include:
        if not any(token.lower() in name.lower() for token in include):
            return False
    if exclude:
        if any(token.lower() in name.lower() for token in exclude):
            return False
    return True


def filter_scenarios(
    scenarios: Sequence[Path],
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
) -> list[Path]:
    """Filter scenarios by name using include/exclude tokens."""

    return [sc for sc in scenarios if matches_filters(sc.stem, include, exclude)]


def has_successful_run(run_name: str, results_root: Path) -> bool:
    """Return ``True`` if a prior successful run exists for *run_name*."""

    if not results_root.exists():
        return False

    prefix = f"{run_name}_"
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            if (child / "summary.csv").exists():
                return True
    return False


def collect_status(results_root: Path) -> list[RunStatusEntry]:
    """Collect execution status metadata under *results_root*."""

    runs: list[RunStatusEntry] = []
    if not results_root.exists():
        return runs

    for child in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        if child.name == "aggregate":
            runs.append(
                RunStatusEntry(run_dir=str(child), status="aggregate", path=str(child))
            )
            continue

        entry = RunStatusEntry(run_dir=str(child), status="empty")
        metadata_path = child / "run_metadata.json"
        summary_path = child / "summary.csv"

        scenario_name: str | None = None
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                config_path = meta.get("config_path")
                if isinstance(config_path, str) and config_path:
                    scenario_name = Path(config_path).stem
                scenario_name = scenario_name or meta.get("scenario") or meta.get("run_name")
            except Exception:  # pragma: no cover - defensive parsing path
                scenario_name = None
        if scenario_name:
            entry.scenario = scenario_name

        if summary_path.exists():
            entry.status = "success"
            entry.summary_path = str(summary_path)
        elif metadata_path.exists():
            entry.status = "incomplete"
            entry.metadata_path = str(metadata_path)

        runs.append(entry)

    return runs


__all__ = [
    "collect_status",
    "filter_scenarios",
    "has_successful_run",
    "matches_filters",
]
