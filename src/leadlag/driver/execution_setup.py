"""Execution preparation utilities for the LeadLag driver CLI."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

__all__ = ["ExecutionOptions", "ExecutionSetup", "prepare_execution"]


@dataclass(slots=True)
class ExecutionOptions:
    """Configuration for executing scenarios."""

    results_root: Path
    runner_preference: str = "auto"
    skip_existing: bool = False
    stop_on_error: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class ExecutionSetup:
    """Prepared resources for executing scenarios via the CLI."""

    results_root: Path
    logger: Logger
    options: ExecutionOptions
    command: str


def prepare_execution(args: Namespace) -> ExecutionSetup:
    """Prepare execution resources for the CLI entrypoint."""

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    log_path_value = getattr(args, "log_path", None)
    log_path = Path(log_path_value).expanduser().resolve() if log_path_value else None
    from .logging import configure_driver_logger

    logger = configure_driver_logger(
        results_root,
        log_level=getattr(args, "log_level", None),
        log_path=log_path,
    )

    execution_options = ExecutionOptions(
        results_root=results_root,
        runner_preference=getattr(args, "runner", "auto"),
        skip_existing=getattr(args, "skip_existing", False),
        stop_on_error=getattr(args, "stop_on_error", False),
        dry_run=getattr(args, "dry_run", False),
    )

    command_string = getattr(args, "_leadlag_command", "leadlag")

    return ExecutionSetup(
        results_root=results_root,
        logger=logger,
        options=execution_options,
        command=command_string,
    )

