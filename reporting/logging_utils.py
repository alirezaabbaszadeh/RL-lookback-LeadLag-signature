from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Mapping, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


class ContextFilter(logging.Filter):
    """Inject structured context data into log records."""

    def __init__(self, base_context: Optional[Mapping[str, object]] = None) -> None:
        super().__init__()
        self._base_context = dict(base_context or {})

    @staticmethod
    def _format(context_map: Mapping[str, object]) -> str:
        if not context_map:
            return "-"
        parts = []
        for key in sorted(context_map):
            value = context_map[key]
            parts.append(f"{key}={value}")
        return " ".join(parts)

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - thin shim
        current = getattr(record, "context_map", {}) or {}
        merged = dict(self._base_context)
        merged.update(current)
        record.context = self._format(merged)
        return True


class StructuredAdapter(logging.LoggerAdapter):
    """Logger adapter that carries structured context across log calls."""

    def process(self, msg, kwargs):  # pragma: no cover - glue layer
        supplied = kwargs.pop("context", None) or {}
        base_context = dict(self.extra.get("context_map", {}))
        base_context.update(supplied)

        extra = kwargs.setdefault("extra", {})
        context_map = dict(extra.get("context_map", {}))
        context_map.update(base_context)
        extra["context_map"] = context_map
        return msg, kwargs


def get_logger(name: str, *, context: Optional[Mapping[str, object]] = None) -> logging.LoggerAdapter:
    """Return a logger that automatically carries structured context."""
    base_logger = logging.getLogger(name)
    return StructuredAdapter(base_logger, {"context_map": dict(context or {})})


def setup_logging(
    log_path: Path,
    level: str = "INFO",
    config_path: Optional[Path] = None,
    *,
    context: Optional[Mapping[str, object]] = None,
) -> None:
    """Configure logging for a run with consistent format and context injection."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    filter_instance = ContextFilter(context)

    if config_path and config_path.exists() and yaml is not None:
        try:  # pragma: no cover - integration path
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            logging.getLogger().addFilter(filter_instance)
            return
        except Exception:
            # continue to fallback configuration
            pass

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(context)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(filter_instance)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    file_handler.addFilter(filter_instance)
    logger.addHandler(file_handler)
