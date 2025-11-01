"""Command line interface for computing paper-grade statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import stats


def _table_to_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def _load_returns(run_dir: Path) -> pd.Series | None:
    returns_path = run_dir / "returns.csv"
    if not returns_path.exists():
        return None
    df = pd.read_csv(returns_path)
    if "returns" not in df.columns:
        raise ValueError(f"returns.csv in {run_dir} is missing the 'returns' column")
    return pd.Series(df["returns"], dtype=float)


def _load_metrics(run_dir: Path) -> pd.DataFrame | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def _plot_hac_forest(df: pd.DataFrame, out_path: Path) -> None:
    filtered = df.dropna(subset=["hac_lower", "hac_upper"]).copy()
    if filtered.empty:
        return
    filtered.sort_values("run_id", inplace=True)
    centers = (filtered["hac_lower"].to_numpy() + filtered["hac_upper"].to_numpy()) / 2
    lower_errors = centers - filtered["hac_lower"].to_numpy()
    upper_errors = filtered["hac_upper"].to_numpy() - centers
    errors = np.vstack([lower_errors, upper_errors])

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(filtered) + 1)))
    positions = np.arange(len(filtered))
    ax.errorbar(
        centers,
        positions,
        xerr=errors,
        fmt="o",
        ecolor="#1f77b4",
        capsize=4,
        markersize=6,
        color="#1f77b4",
    )
    ax.axvline(0.0, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.set_xlabel("Annualised Mean Return (HAC CI)")
    ax.set_yticks(positions)
    ax.set_yticklabels(filtered["run_id"].tolist())
    ax.set_title("HAC Confidence Intervals per Run")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_sharpe_heatmap(metrics: pd.DataFrame, out_path: Path) -> None:
    if metrics.empty or "agent" not in metrics.columns or "timeframe" not in metrics.columns:
        return
    subset = metrics.dropna(subset=["agent", "timeframe"])
    if subset.empty:
        return
    pivot = subset.pivot_table(index="agent", columns="timeframe", values="Sharpe", aggfunc="mean")
    if pivot.empty:
        return
    data = pivot.to_numpy(dtype=float)
    mask = np.isnan(data)
    if mask.all():
        return
    data = np.ma.masked_array(data, mask=mask)

    fig, ax = plt.subplots(figsize=(1.5 * max(2, pivot.shape[1]), 0.6 * max(2, pivot.shape[0])))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Average Sharpe by Agent/Timeframe")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe Ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Root directory with run outputs")
    parser.add_argument("--out", type=Path, required=True, help="Directory to store aggregated artifacts")
    parser.add_argument("--periods", type=int, default=252, help="Trading periods per year")
    parser.add_argument("--spa-iterations", type=int, default=500, help="Bootstrap iterations for SPA")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap routines")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    metrics_frames = []
    returns_map: Dict[str, pd.Series] = {}
    for run_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        metrics = _load_metrics(run_dir)
        if metrics is not None:
            metrics_frames.append(metrics)
        returns = _load_returns(run_dir)
        if returns is not None:
            returns_map[run_dir.name] = returns

    if metrics_frames:
        all_metrics = pd.concat(metrics_frames, ignore_index=True)
    else:
        all_metrics = pd.DataFrame()

    metrics_path = args.out / "all_metrics_raw.csv"
    all_metrics.to_csv(metrics_path, index=False)

    if not all_metrics.empty:
        summary = (
            all_metrics.groupby("agent")[
                ["Sharpe", "Sortino", "MaxDD", "PnL", "Turnover", "Exposure"]
            ]
            .agg(["mean", "std"])
        )
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
        summary.sort_index(inplace=True)
        summary.to_csv(args.out / "summary_table.csv")

        best_idx = all_metrics.groupby("agent")["Sharpe"].idxmax()
        best = all_metrics.loc[best_idx].sort_values("Sharpe", ascending=False)
        best.to_csv(args.out / "best_per_agent.csv", index=False)
    else:
        summary = pd.DataFrame()
        best = pd.DataFrame()

    advanced_records = []
    for run_id, returns in returns_map.items():
        psr = stats.probabilistic_sharpe_ratio(returns, periods_per_year=args.periods)
        dsr = stats.deflated_sharpe_ratio(
            returns,
            periods_per_year=args.periods,
            num_trials=max(1, len(returns_map)),
        )
        hac_low, hac_high = stats.hac_confidence_interval(returns, periods_per_year=args.periods)
        advanced_records.append(
            {
                "run_id": run_id,
                "psr": psr,
                "dsr": dsr,
                "hac_lower": hac_low,
                "hac_upper": hac_high,
            }
        )
    advanced_df = pd.DataFrame.from_records(advanced_records)
    advanced_df.to_csv(args.out / "advanced_metrics.csv", index=False)
    if not advanced_df.empty:
        advanced_df[["run_id", "psr", "dsr"]].to_csv(
            args.out / "psr_dsr_pvalues.csv", index=False
        )
        advanced_df[["run_id", "hac_lower", "hac_upper"]].to_csv(
            args.out / "hac_confidence_intervals.csv", index=False
        )
        _plot_hac_forest(advanced_df, args.out / "forest_hac_ci.png")

    spa_df = stats.spa_reality_check(
        returns_map,
        periods_per_year=args.periods,
        iterations=args.spa_iterations,
        seed=args.seed,
    )
    spa_df.to_csv(args.out / "spa_results.csv", index=False)
    if not spa_df.empty:
        spa_df.to_csv(args.out / "spa_pvalues.csv", index=False)

    mcs_members = stats.model_confidence_set(returns_map, periods_per_year=args.periods)
    with (args.out / "mcs.json").open("w", encoding="utf-8") as handle:
        json.dump({"members": mcs_members}, handle, indent=2)

    _plot_sharpe_heatmap(all_metrics, args.out / "heatmap_agent_timeframe.png")

    summary_lines = ["# Paper Results", ""]
    if not all_metrics.empty:
        summary_lines.append(f"Total runs: {len(all_metrics)}")
    if not summary.empty:
        summary_lines.append("\n## Agent Summary")
        summary_lines.append(_table_to_text(summary.reset_index()))
    if not best.empty:
        summary_lines.append("\n## Best per Agent")
        summary_lines.append(_table_to_text(best))
    if not advanced_df.empty:
        summary_lines.append("\n## Advanced Metrics")
        summary_lines.append(_table_to_text(advanced_df))
    if not spa_df.empty:
        summary_lines.append("\n## SPA Reality Check")
        summary_lines.append(_table_to_text(spa_df))
    summary_path = args.out / "paper_results.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
