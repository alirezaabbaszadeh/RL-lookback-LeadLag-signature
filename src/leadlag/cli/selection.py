"""Value objects for scenario selection results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class SelectionStatus(str, Enum):
    """Enumerate outcomes of resolving scenario selections."""

    OK = "ok"
    INVALID = "invalid"
    NO_MATCHES = "no_matches"


@dataclass(frozen=True)
class SelectionResult:
    """Resolved scenario paths and metadata about the selection outcome."""

    paths: Sequence[Path]
    errors: Sequence[str]
    status: SelectionStatus

    @property
    def succeeded(self) -> bool:
        """Return ``True`` when the selection completed without errors."""

        return self.status is SelectionStatus.OK
