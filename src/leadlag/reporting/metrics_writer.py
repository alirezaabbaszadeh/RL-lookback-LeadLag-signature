"""Canonical metrics export utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

CANONICAL_COLUMNS = [
    "experiment_id",
    "agent",
    "action_space",
    "policy",
    "features_signature",
    "signature_depth",
    "features_leadlag",
    "time_channel",
    "lookback",
    "horizon",
    "universe",
    "timeframe",
    "split_scheme",
    "cost_fee_bps",
    "slippage_bps",
    "reward",
    "seed",
    "window_index",
    "Sharpe",
    "Sortino",
    "MaxDD",
    "Turnover",
    "PnL",
    "Costs",
    "Exposure",
    "EnvSteps",
]


@dataclass
class MetricsWriter:
    """Persist canonical metrics for downstream aggregation."""

    config: Mapping[str, object]

    def _ensure_header(self, path: Path) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
            writer.writeheader()

    def write_row(self, path: Path, row: Mapping[str, object]) -> None:
        self._ensure_header(path)
        filtered = {key: row.get(key, None) for key in CANONICAL_COLUMNS}
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
            writer.writerow(filtered)

    def dataframe(self, rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(list(rows), columns=CANONICAL_COLUMNS)


def build_metadata_row(
    run_id: str,
    cfg: Mapping[str, object],
    metrics: Mapping[str, float],
    seed: int,
    window_idx: int,
    turnover: float = 0.0,
    exposure: float = 0.0,
    costs: float = 0.0,
    env_steps: int | None = None,
) -> Dict[str, object]:
    """Compose a canonical metrics row for ``metrics.csv``."""

    features_cfg = cfg.get("features", {})
    signature_cfg = features_cfg.get("signature", {}) if isinstance(features_cfg, Mapping) else {}
    leadlag_cfg = features_cfg.get("leadlag", {}) if isinstance(features_cfg, Mapping) else {}
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    return {
        "experiment_id": run_id,
        "agent": cfg.get("agent", {}).get("name") if isinstance(cfg.get("agent"), Mapping) else cfg.get("agent"),
        "action_space": cfg.get("env", {}).get("action_space") if isinstance(cfg.get("env"), Mapping) else None,
        "policy": cfg.get("policy", {}).get("name") if isinstance(cfg.get("policy"), Mapping) else None,
        "features_signature": signature_cfg.get("enabled", False),
        "signature_depth": signature_cfg.get("depth", 0),
        "features_leadlag": leadlag_cfg.get("enabled", False),
        "time_channel": features_cfg.get("time_channel", False) if isinstance(features_cfg, Mapping) else False,
        "lookback": cfg.get("window", {}).get("lookback") if isinstance(cfg.get("window"), Mapping) else None,
        "horizon": cfg.get("target", {}).get("horizon") if isinstance(cfg.get("target"), Mapping) else None,
        "universe": data_cfg.get("universe") if isinstance(data_cfg, Mapping) else None,
        "timeframe": data_cfg.get("timeframe") if isinstance(data_cfg, Mapping) else None,
        "split_scheme": cfg.get("split", {}).get("scheme") if isinstance(cfg.get("split"), Mapping) else None,
        "cost_fee_bps": cfg.get("costs", {}).get("fee_bps") if isinstance(cfg.get("costs"), Mapping) else None,
        "slippage_bps": cfg.get("slippage", {}).get("bps") if isinstance(cfg.get("slippage"), Mapping) else None,
        "reward": cfg.get("reward", {}).get("name") if isinstance(cfg.get("reward"), Mapping) else None,
        "seed": seed,
        "window_index": window_idx,
        "Sharpe": metrics.get("sharpe"),
        "Sortino": metrics.get("sortino"),
        "MaxDD": metrics.get("max_drawdown"),
        "Turnover": turnover,
        "PnL": metrics.get("pnl"),
        "Costs": costs,
        "Exposure": exposure,
        "EnvSteps": env_steps,
    }
