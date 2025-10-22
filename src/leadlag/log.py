"""Compatibility shim for environments expecting a ``log`` module.

Kaggle's notebook runtime imports ``log`` from :mod:`sitecustomize` during
startup.  The bespoke module is not available in the stripped-down execution
environment that powers the automated tests, which results in an immediate
``ModuleNotFoundError`` before any project code can run.  Providing a lightweight
implementation keeps the import happy while still emitting useful messages to
stderr.

The shim intentionally mirrors the handful of logging-style helpers used by the
original Kaggle module (``info``, ``warning``, ``error``, ``debug`` and
``exception``).  Any other attributes fallback to a no-op logger so that future
changes to Kaggle's bootstrap code do not cause hard failures.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

_LOGGER_NAME = "kaggle.log"
_logger = logging.getLogger(_LOGGER_NAME)


def _ensure_handler() -> None:
    """Attach a simple stderr handler the first time the module is imported."""

    if _logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def _format_message(*args: Any, sep: str = " ", end: str = "\n") -> str:
    """Join positional arguments in a ``print``-like fashion."""

    message = sep.join(str(arg) for arg in args)
    if end:
        message += end
    return message.rstrip("\n")


def _log(level: int, *args: Any, sep: str = " ", end: str = "\n") -> None:
    _ensure_handler()
    message = _format_message(*args, sep=sep, end=end)
    if message:
        _logger.log(level, message)


def debug(*args: Any, **kwargs: Any) -> None:
    _log(logging.DEBUG, *args, **kwargs)


def info(*args: Any, **kwargs: Any) -> None:
    _log(logging.INFO, *args, **kwargs)


def warning(*args: Any, **kwargs: Any) -> None:
    _log(logging.WARNING, *args, **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    _log(logging.ERROR, *args, **kwargs)


def critical(*args: Any, **kwargs: Any) -> None:
    _log(logging.CRITICAL, *args, **kwargs)


def exception(*args: Any, **kwargs: Any) -> None:
    _log(logging.ERROR, *args, **kwargs)
    _ensure_handler()
    _logger.exception("")


def send(event: str, payload: Any | None = None) -> None:
    """Best-effort serialization for Kaggle's ``log.send`` helper."""

    if payload is None:
        info(event)
        return
    try:
        serialized = json.dumps(payload)
    except Exception:  # pragma: no cover - defensive fallback
        serialized = str(payload)
    info(f"{event}: {serialized}")


def __getattr__(name: str) -> Callable[..., None]:
    """Return a permissive no-op logger for unexpected attributes."""

    def _fallback(*args: Any, **kwargs: Any) -> None:
        info(f"[{name}]", *args, **kwargs)

    return _fallback


__all__ = [
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "send",
    "warning",
]
