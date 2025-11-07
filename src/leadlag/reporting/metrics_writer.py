"""Canonical metrics export utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

SCHEMA: list[tuple[str, str]] = [
    ("experiment_id", "string"),
    ("agent", "string"),
    ("action_space", "string"),
    ("policy", "string"),
    ("features_signature", "boolean"),
    ("signature_depth", "Int64"),
    ("features_leadlag", "boolean"),
    ("time_channel", "boolean"),
    ("lookback", "Int64"),
    ("horizon", "Int64"),
    ("universe", "string"),
    ("timeframe", "string"),
    ("split_scheme", "string"),
    ("cost_fee_bps", "float64"),
    ("slippage_bps", "float64"),
    ("reward", "string"),
    ("seed", "Int64"),
    ("window_index", "Int64"),
    ("Sharpe", "float64"),
    ("Sortino", "float64"),
    ("MaxDD", "float64"),
    ("Turnover", "float64"),
    ("PnL", "float64"),
    ("Costs", "float64"),
    ("Exposure", "float64"),
    ("EnvSteps", "Int64"),
]

METRICS_COLUMNS = [name for name, _ in SCHEMA]
METRICS_DTYPES = {name: dtype for name, dtype in SCHEMA}
# Backwards compatibility for legacy imports
CANONICAL_COLUMNS = METRICS_COLUMNS


def _coerce_series_dtype(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "string":
        return series.astype("string")
    if dtype == "Int64":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.astype("Int64")
    if dtype == "float64":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.astype("float64")
    if dtype == "boolean":
        return series.astype("boolean")
    raise ValueError(f"Unsupported metrics dtype: {dtype}")


def enforce_metrics_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` coerced to the canonical metrics schema."""

    if frame.empty:
        return pd.DataFrame(columns=METRICS_COLUMNS).astype({col: dtype for col, dtype in METRICS_DTYPES.items()})

    working = frame.copy()
    for name in METRICS_COLUMNS:
        if name not in working.columns:
            working[name] = pd.NA

    coerced = working.loc[:, METRICS_COLUMNS].copy()
    for name, dtype in METRICS_DTYPES.items():
        coerced[name] = _coerce_series_dtype(coerced[name], dtype)
    return coerced


def _serialise_scalar(value: object, dtype: str) -> object:
    if pd.isna(value):
        return None
    if dtype == "Int64":
        return int(value)
    if dtype == "float64":
        return float(value)
    if dtype == "boolean":
        return bool(value)
    return str(value)


@dataclass
class MetricsWriter:
    """Persist canonical metrics for downstream aggregation."""

    config: Mapping[str, object]

    def _ensure_header(self, path: Path) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
            writer.writeheader()

    def _prepare_row(self, row: Mapping[str, object]) -> Dict[str, object]:
        frame = pd.DataFrame([row])
        coerced = enforce_metrics_schema(frame)
        serialised = {}
        series = coerced.iloc[0]
        for name in METRICS_COLUMNS:
            serialised[name] = _serialise_scalar(series[name], METRICS_DTYPES[name])
        return serialised

    def write_row(self, path: Path, row: Mapping[str, object]) -> None:
        self._ensure_header(path)
        filtered = self._prepare_row(row)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
            writer.writerow(filtered)

    def dataframe(self, rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
        frame = pd.DataFrame(list(rows))
        return enforce_metrics_schema(frame)


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

    agent_cfg = cfg.get("agent", {})
    if not isinstance(agent_cfg, Mapping):
        agent_cfg = {}
    agent_policy_cfg = agent_cfg.get("policy")

    policy_value: object | None = None
    if isinstance(agent_policy_cfg, Mapping):
        policy_value = (
            agent_policy_cfg.get("name")
            or agent_policy_cfg.get("policy")
            or agent_policy_cfg.get("target")
        )
    elif agent_policy_cfg is not None:
        policy_value = agent_policy_cfg

    if policy_value is None:
        policy_cfg = cfg.get("policy")
        if isinstance(policy_cfg, Mapping):
            policy_value = policy_cfg.get("name") or policy_cfg.get("policy")
        else:
            policy_value = policy_cfg

    return {
        "experiment_id": run_id,
        "agent": cfg.get("agent", {}).get("name") if isinstance(cfg.get("agent"), Mapping) else cfg.get("agent"),
        "action_space": cfg.get("env", {}).get("action_space") if isinstance(cfg.get("env"), Mapping) else None,
        "policy": policy_value,
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
