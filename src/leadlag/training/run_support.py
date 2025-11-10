from __future__ import annotations

import json
import logging
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
from leadlag.utils import (
    collect_determinism_settings,
    collect_environment_manifest,
    set_all_seeds,
    update_run_manifest,
)
from leadlag.utils.resources import resolve_path


PriceLoader = Callable[[Dict[str, Any]], Tuple[pd.DataFrame, Optional[Path]]]


def _set_seed(seed: int) -> int:
    """Compat wrapper returning the normalised seed value.

    The offline research utilities historically imported ``_set_seed`` from this
    module.  During the refactor that introduced :func:`set_all_seeds` the helper
    disappeared which now breaks those imports.  Re-introducing the thin wrapper
    keeps the research scripts working while ensuring the behaviour funnels
    through :func:`set_all_seeds` so determinism stays centralised in
    ``leadlag.utils``.

    Parameters
    ----------
    seed:
        Any integer-like value provided by the caller.

    Returns
    -------
    int
        The normalised integer seed that was applied.
    """

    normalised_seed = int(seed)
    set_all_seeds(normalised_seed)
    return normalised_seed


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
    run_manifest_path: Path
    requested_env_steps: Optional[int]


def _resolve_preset_name(section: Mapping[str, Any] | None, fallback: Optional[str] = None) -> Optional[str]:
    if isinstance(section, Mapping):
        candidate = section.get("preset_name") or section.get("preset")
        if isinstance(candidate, str) and candidate:
            return candidate
    return fallback


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
    requested_env_steps: Optional[int] = None,
) -> RunPreparation:
    """Prepare the common execution environment for training scripts."""

    run_section = cfg.get("run", {})
    seed = int(run_section.get("seed", 42))
    set_all_seeds(seed)
    determinism = collect_determinism_settings(seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or run_section.get("run_name", module)
    output_root = Path(out_root or run_section.get("output_root", "results"))
    out_dir = output_root / f"{resolved_run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_preset = _resolve_preset_name(cfg.get("training"), run_section.get("training_preset"))
    hardware_preset = _resolve_preset_name(cfg.get("hardware"), run_section.get("hardware_preset"))

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

    run_manifest_path = out_dir / "run_manifest.json"
    base_manifest: Dict[str, Any] = {
        "run": {
            "module": module,
            "name": resolved_run_name,
            "seed": seed,
            "timestamp": ts,
            "output_dir": str(out_dir),
        },
        "presets": {
            "training": training_preset,
            "hardware": hardware_preset,
        },
        "determinism": determinism,
    }
    if requested_env_steps is not None:
        base_manifest["requested_env_steps"] = int(requested_env_steps)

    update_run_manifest(run_manifest_path, base_manifest)

    environment_manifest = collect_environment_manifest()
    metadata: Dict[str, Any] = {
        "config_path": str(cfg_path.resolve()),
        "created_at": ts,
        "environment": environment_manifest,
        "determinism": determinism,
        "data_source_config": cfg.get("data", {}).get("price_csv", ""),
        "data_manifest": str(manifest_path),
        "run_manifest": str(run_manifest_path),
        "hardware_preset": hardware_preset,
        "training_preset": training_preset,
    }
    git_commit = environment_manifest.get("git_commit")
    if git_commit:
        metadata["git"] = {
            "commit": git_commit,
            "dirty": environment_manifest.get("git_dirty"),
            "branch": environment_manifest.get("git_branch"),
        }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    update_run_manifest(
        run_manifest_path,
        {
            "data_manifest": str(manifest_path),
            "config": {
                "source": str(cfg_path.resolve()),
                "merged": str(config_path),
            },
        },
    )

    return RunPreparation(
        out_dir=out_dir,
        logger=logger,
        prices=prices,
        manifest_path=manifest_path,
        timestamp=ts,
        seed=seed,
        run_name=resolved_run_name,
        resolved_price_path=resolved_price_path,
        run_manifest_path=run_manifest_path,
        requested_env_steps=requested_env_steps,
    )
