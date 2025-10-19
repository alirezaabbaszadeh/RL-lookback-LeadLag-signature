from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


CANDIDATE_RETURN_COLUMNS = [
    "portfolio_return",
    "strategy_return",
    "reward",
    "reward_step_mean",
    "pnl",
]
FALLBACK_COLUMN = "mean_abs_matrix"


@dataclass
class RunInfo:
    run_dir: Path
    scenario: str
    method: str
    lookback_label: str
    seed_label: str
    label: str
    equity: pd.Series


def _load_yaml(path: Path) -> Dict:
    if yaml is None:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _derive_returns(df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str]]:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 2:
                return series, col
    if FALLBACK_COLUMN in df.columns:
        base = pd.to_numeric(df[FALLBACK_COLUMN], errors="coerce").dropna()
        if len(base) > 5:
            returns = base.pct_change().dropna()
            if len(returns) > 2:
                return returns, f"{FALLBACK_COLUMN}_pct_change"
    return None, None


def _compute_equity(returns: pd.Series, start_balance: float) -> pd.Series:
    equity = (1 + returns).cumprod() * start_balance
    equity.index = range(len(equity))
    return equity


def _scenario_from_metadata(run_dir: Path) -> str:
    meta_p = run_dir / "run_metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            cfg_path = meta.get("config_path")
            if isinstance(cfg_path, str):
                return Path(cfg_path).stem
        except Exception:
            pass
    name = run_dir.name
    parts = name.split("_")
    if len(parts) > 2:
        return "_".join(parts[:-2])
    return name


def _extract_config_info(run_dir: Path, scenario: str) -> tuple[str, str, str, str]:
    cfg = _load_yaml(run_dir / "config_merged.yaml")
    analysis = cfg.get("analysis", {}) if isinstance(cfg, dict) else {}
    runner = cfg.get("runner", "scenario") if isinstance(cfg, dict) else "scenario"
    lookback = analysis.get("lookback") if isinstance(analysis, dict) else None
    method = analysis.get("method") if isinstance(analysis, dict) else None

    if method is None:
        if runner == "rl":
            method = "rl"
        elif "dynamic" in scenario:
            method = "dynamic"
        elif "ccf" in scenario:
            method = "ccf_at_lag"
        else:
            method = "signature"
    method = str(method)

    if lookback is None:
        if runner == "rl":
            lookback_label = "adaptive"
        else:
            lookback_label = "unknown"
    else:
        lookback_label = f"L={lookback}"

    seed_label = ""
    seed = cfg.get("run", {}).get("seed") if isinstance(cfg, dict) else None
    if seed is None:
        for tok in run_dir.name.split("_"):
            if tok.startswith("seed"):
                try:
                    seed = int(tok.replace("seed", ""))
                except Exception:
                    seed = None
    if seed is not None:
        seed_label = f"seed={seed}"

    label_parts = [scenario]
    if lookback_label:
        label_parts.append(lookback_label)
    if seed_label:
        label_parts.append(seed_label)
    label = " | ".join(label_parts)
    return method, lookback_label, seed_label, label


def collect_run_infos(results_root: Path, start_balance: float) -> List[RunInfo]:
    infos: List[RunInfo] = []
    for metrics_path in sorted(results_root.rglob("metrics_timeseries.csv")):
        run_dir = metrics_path.parent
        try:
            df = pd.read_csv(metrics_path)
            if "date" in df.columns:
                try:
                    df = df.sort_values("date")
                except Exception:
                    pass
            returns, source = _derive_returns(df)
            if returns is None or returns.empty:
                continue
            scenario = _scenario_from_metadata(run_dir)
            method, lookback_label, seed_label, label = _extract_config_info(run_dir, scenario)
            equity = _compute_equity(returns, start_balance)
            infos.append(
                RunInfo(
                    run_dir=run_dir,
                    scenario=scenario,
                    method=method,
                    lookback_label=lookback_label,
                    seed_label=seed_label,
                    label=label,
                    equity=equity,
                )
            )
        except Exception:
            continue
    return infos


def _plot_group(infos: List[RunInfo], title: str, out_path: Path, legend: bool = True) -> None:
    if not infos:
        return
    plt.figure(figsize=(12, 6))
    for info in infos:
        plt.plot(info.equity.index, info.equity.values, label=info.label, linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Balance")
    if legend:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_all_charts(
    results_root: Path,
    out_dir: Path,
    start_balance: float,
    max_lines: Optional[int] = None,
) -> None:
    infos = collect_run_infos(results_root, start_balance)
    if not infos:
        print("[plot_balance] No qualifying runs found.")
        return

    if max_lines is not None:
        _plot_group(
            infos[:max_lines],
            "Portfolio Balance History - All Runs (Subset)",
            out_dir / "balance_all_runs_subset.png",
        )
    _plot_group(infos, "Portfolio Balance History - All Runs", out_dir / "balance_all_runs.png")

    by_scenario: Dict[str, List[RunInfo]] = {}
    by_method: Dict[str, List[RunInfo]] = {}
    by_lookback: Dict[str, List[RunInfo]] = {}

    for info in infos:
        by_scenario.setdefault(info.scenario, []).append(info)
        by_method.setdefault(info.method, []).append(info)
        by_lookback.setdefault(info.lookback_label, []).append(info)

    for scenario, items in by_scenario.items():
        _plot_group(
            items,
            f"Portfolio Balance History - Scenario: {scenario}",
            out_dir / "scenario" / f"balance_{scenario}.png",
        )

    for method, items in by_method.items():
        _plot_group(
            items,
            f"Portfolio Balance History - Method: {method}",
            out_dir / "method" / f"balance_method_{method}.png",
        )

    for lookback, items in by_lookback.items():
        label = lookback or "unknown"
        _plot_group(
            items,
            f"Portfolio Balance History - Lookback: {label}",
            out_dir / "lookback" / f"balance_lookback_{label.replace('=','')}.png",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot equity curves for all runs, grouped by scenario/method/lookback.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/plots/balance"))
    parser.add_argument("--start-balance", type=float, default=100_000.0)
    parser.add_argument("--max-lines", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    plot_all_charts(args.results_root, args.out, args.start_balance, args.max_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
