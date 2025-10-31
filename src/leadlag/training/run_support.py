from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:  # optional dependency
    import yaml
except Exception:  # pragma: no cover - yaml is optional
    yaml = None

from leadlag.governance.dataset import build_manifest, record_manifest, run_quality_checks
from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.reporting.profiling import profile_to
from leadlag.utils.resources import resolve_path


PriceLoader = Callable[[Dict[str, Any]], Tuple[pd.DataFrame, Optional[Path]]]


@dataclass
class RunPreparation:
    """Container for shared run artefacts."""

    out_dir: Path
    logger: logging.LoggerAdapter
    prices: pd.DataFrame
    manifest_path: Path
    timestamp: str
    seed: int
    run_name: str
    resolved_price_path: Optional[Path]


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and the default random generator."""

    random.seed(seed)
    np.random.seed(seed)


def _detect_git() -> Dict[str, Any]:
    import subprocess

    meta: Dict[str, Any] = {}
    try:
        meta["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        meta["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        meta["git_dirty"] = len(status.strip()) > 0
    except Exception:  # pragma: no cover - best effort metadata capture
        meta["git_commit"] = None
        meta["git_branch"] = None
        meta["git_dirty"] = None
    return meta


def _env_info() -> Dict[str, Any]:
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    def version_for(pkg: str) -> Optional[str]:
        try:
            module = __import__(pkg)
            return getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependencies
            return None

    for pkg in ["numpy", "pandas", "scipy", "sklearn", "tqdm", "iisignature", "dcor"]:
        info[f"{pkg}_version"] = version_for(pkg)
    return info


def read_prices(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Path]]:
    price_path = Path(cfg.get("data", {}).get("price_csv", "raw_data/daily_price.csv"))
    resolved_path: Optional[Path] = None
    if not price_path.exists():
        candidates = list(Path("raw_data").glob("daily_prices_*.csv"))
        if candidates:
            price_path = candidates[0]
    if price_path.exists():
        resolved_path = price_path
    else:
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        rng = np.random.default_rng(seed=cfg.get("run", {}).get("seed", 42))
        data = rng.normal(0, 0.01, size=(len(dates), 3)).cumsum(axis=0) + 100
        df = pd.DataFrame(data, index=dates, columns=["AssetA", "AssetB", "AssetC"])
        return df, resolved_path

    df = pd.read_csv(price_path)
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    elif "Date" in df.columns:
        idx = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"])
    else:
        idx = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]
    df.index = idx
    df = df.sort_index()

    limit_days = cfg.get("data", {}).get("limit_days")
    if limit_days is not None:
        try:
            n = int(limit_days)
            if n > 0:
                df = df.iloc[:n]
        except Exception:  # pragma: no cover - guardrail for malformed input
            pass

    placebo = False
    try:
        placebo = bool(cfg.get("data", {}).get("placebo_shuffle", False))
    except Exception:  # pragma: no cover - guardrail for malformed input
        placebo = False
    if placebo and len(df) > 1:
        rng = np.random.default_rng(seed=cfg.get("run", {}).get("seed", 42))
        idx_perm = rng.permutation(len(df))
        df = df.iloc[idx_perm]

    return df, resolved_path


def prepare_run_environment(
    cfg: Dict[str, Any],
    *,
    cfg_path: Path,
    module: str,
    logger_name: str,
    read_prices_fn: PriceLoader = read_prices,
    out_root: Optional[str] = None,
    run_name: Optional[str] = None,
    extra_logging_context: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    profile_label: Optional[str] = None,
) -> RunPreparation:
    """Prepare the common execution environment for training scripts."""

    run_section = cfg.get("run", {})
    seed = int(run_section.get("seed", 42))
    _set_seed(seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or run_section.get("run_name", module)
    output_root = Path(out_root or run_section.get("output_root", "results"))
    out_dir = output_root / f"{resolved_run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_context: Dict[str, Any] = {
        "module": module,
        "run_name": resolved_run_name,
        "seed": seed,
    }
    if extra_logging_context:
        logging_context.update(dict(extra_logging_context))

    config_file = resolve_path("leadlag.configs", "logging_config.yaml")
    try:
        setup_logging(
            out_dir / "run.log",
            level="INFO",
            config_path=config_file,
            context=logging_context,
        )
    except Exception:  # pragma: no cover - fallback path
        setup_logging(out_dir / "run.log", level="INFO", context=logging_context)
    logger = get_logger(logger_name, context=logging_context)

    config_path = out_dir / "config_merged.yaml"
    if yaml is not None:
        config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    else:
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if profile_label:
        with profile_to(out_dir, label=profile_label):
            prices, resolved_price_path = read_prices_fn(cfg)
    else:
        prices, resolved_price_path = read_prices_fn(cfg)

    manifest = build_manifest(
        prices,
        source_path=resolved_price_path,
        extras={"quality": run_quality_checks(prices)},
    )
    manifest_path = record_manifest(manifest, out_dir)

    metadata: Dict[str, Any] = {
        "config_path": str(cfg_path.resolve()),
        "created_at": ts,
        "git": _detect_git(),
        "env": _env_info(),
        "data_source_config": cfg.get("data", {}).get("price_csv", ""),
        "data_manifest": str(manifest_path),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return RunPreparation(
        out_dir=out_dir,
        logger=logger,
        prices=prices,
        manifest_path=manifest_path,
        timestamp=ts,
        seed=seed,
        run_name=resolved_run_name,
        resolved_price_path=resolved_price_path,
    )
