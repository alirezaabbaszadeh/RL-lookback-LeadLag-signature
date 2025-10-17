from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def setup_logging(log_path: Path, level: str = "INFO", config_path: Optional[Path] = None) -> None:
    """Configure logging.

    If `config_path` is provided and valid, attempts to load YAML logging config.
    Falls back to a standard console+file configuration otherwise.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path and config_path.exists() and yaml is not None:
        try:  # pragma: no cover - integration path
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            return
        except Exception:
            # continue to fallback configuration
            pass

    # Fallback basic configuration: stream + file handler
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplication when reconfiguring
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

