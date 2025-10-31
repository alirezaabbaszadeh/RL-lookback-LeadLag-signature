"""Data transfer objects used by the driver service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScenarioSelection:
    """Structured representation of a selected scenario reference."""

    name: str
    display: str
    path: str

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-serialisable payload for the selection."""

        return asdict(self)


@dataclass(slots=True)
class ScenarioResult:
    """Scenario execution outcome captured by the driver."""

    scenario: str
    status: str
    runner: str | None
    output: str | None = None
    error: str | None = None
    reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload matching the historical JSON schema."""

        data = asdict(self)
        payload: dict[str, object] = {
            "scenario": data["scenario"],
            "status": data["status"],
            "runner": data["runner"],
        }
        for optional in ("output", "error", "reason"):
            value = data[optional]
            if value is not None:
                payload[optional] = value
        return payload


@dataclass(slots=True)
class RunStatusEntry:
    """Metadata describing the status of a run discovered on disk."""

    run_dir: str
    status: str
    scenario: str | None = None
    path: str | None = None
    summary_path: str | None = None
    metadata_path: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload mirroring the legacy run status mapping."""

        data = asdict(self)
        payload: dict[str, object] = {
            "run_dir": data["run_dir"],
            "status": data["status"],
        }
        if data["scenario"] is not None:
            payload["scenario"] = data["scenario"]
        if data["status"] == "aggregate" and data["path"] is not None:
            payload["path"] = data["path"]
        if data["summary_path"] is not None:
            payload["summary_path"] = data["summary_path"]
        if data["metadata_path"] is not None:
            payload["metadata_path"] = data["metadata_path"]
        return payload


@dataclass(slots=True)
class DriverSummary:
    """Serializable payload summarising the driver execution."""

    selected: list[str]
    results_root: str
    summary: list[ScenarioResult]
    aggregate: str | None
    dry_run: bool
    dry_run_entries: list[ScenarioSelection] | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload preserving historical JSON structure."""

        payload: dict[str, object] = {
            "selected": list(self.selected),
            "results_root": self.results_root,
            "summary": [entry.to_payload() for entry in self.summary],
            "aggregate": self.aggregate,
            "dry_run": self.dry_run,
        }
        if self.dry_run_entries:
            payload["dry_run_entries"] = [entry.to_payload() for entry in self.dry_run_entries]
        return payload


@dataclass(slots=True)
class ExecutionResult:
    """Structured result returned from scenario execution."""

    summary: list[ScenarioResult] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)
    aggregate: Path | None = None
    exit_code: int = 0
    aborted: bool = False
    dry_run: bool = False
    dry_run_entries: list[ScenarioSelection] = field(default_factory=list)


@dataclass(slots=True)
class ScenarioExecutionContext:
    """Context describing a scenario prior to execution."""

    scenario: str
    path: Path
    results_root: Path
    config: dict[str, object]
    runner: str


__all__ = [
    "DriverSummary",
    "ExecutionResult",
    "RunStatusEntry",
    "ScenarioExecutionContext",
    "ScenarioResult",
    "ScenarioSelection",
]
