"""Command line interface for computing paper-grade statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import stats

CANDIDATE_RETURN_COLUMNS = [
    "portfolio_return",
    "strategy_return",
    "reward",
    "reward_step_mean",
    "pnl",
]
FALLBACK_COLUMN = "mean_abs_matrix"


def _table_to_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def _derive_returns_from_metrics(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in CANDIDATE_RETURN_COLUMNS:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(series) > 2:
                return series.astype(float)
    if FALLBACK_COLUMN in df.columns:
        base = pd.to_numeric(df[FALLBACK_COLUMN], errors="coerce").dropna()
        if len(base) > 5:
            pct = base.pct_change().dropna()
            if len(pct) > 2:
                return pct.astype(float)
    return None


def _load_returns(run_dir: Path) -> pd.Series | None:
    returns_path = run_dir / "returns.csv"
    if returns_path.exists():
        df = pd.read_csv(returns_path)
        if "returns" not in df.columns:
            raise ValueError(f"returns.csv in {run_dir} is missing the 'returns' column")
        series = pd.Series(df["returns"], dtype=float).dropna()
        return series if not series.empty else None

    metrics_timeseries = run_dir / "metrics_timeseries.csv"
    if metrics_timeseries.exists():
        df = pd.read_csv(metrics_timeseries)
        if "date" in df.columns:
            try:
                df = df.sort_values("date")
            except Exception:
                pass
        series = _derive_returns_from_metrics(df)
        if series is not None:
            return series

    equity_path = run_dir / "equity.csv"
    if equity_path.exists():
        df = pd.read_csv(equity_path)
        if "equity" in df.columns:
            equity = pd.Series(df["equity"], dtype=float).dropna()
            if len(equity) > 1:
                returns = equity.pct_change().dropna()
                if not returns.empty:
                    return returns
    return None


def _load_metrics(run_dir: Path) -> pd.DataFrame | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def _plot_hac_forest(df: pd.DataFrame, out_path: Path) -> None:
    filtered = df.dropna(subset=["hac_sharpe_lower", "hac_sharpe_upper"]).copy()
    if filtered.empty:
        return
    filtered.sort_values("run_id", inplace=True)
    centers = (filtered["hac_sharpe_lower"].to_numpy() + filtered["hac_sharpe_upper"].to_numpy()) / 2
    lower_errors = centers - filtered["hac_sharpe_lower"].to_numpy()
    upper_errors = filtered["hac_sharpe_upper"].to_numpy() - centers
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
    ax.set_xlabel("Annualised Sharpe (HAC CI)")
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


def _plot_pnl_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    if metrics.empty or "PnL" not in metrics.columns:
        return
    subset = metrics.dropna(subset=["PnL"]).copy()
    if subset.empty:
        return
    labels = subset.get("experiment_id")
    if labels is None or labels.isna().all():
        resolved_labels = pd.Index(subset.index.astype(str))
    else:
        string_labels = labels.astype("string")
        fallback = pd.Series(subset.index.astype(str), index=string_labels.index)
        resolved_labels = string_labels.where(string_labels.notna(), fallback)
    subset["label"] = resolved_labels.astype(str)
    ordered = subset.sort_values("PnL", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(ordered))))
    ax.barh(ordered["label"].iloc[::-1], ordered["PnL"].iloc[::-1].to_numpy(dtype=float))
    ax.set_xlabel("PnL")
    ax.set_ylabel("Run")
    ax.set_title("Top PnL by Run (Top 20)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_workflow(
    results_root: Path,
    out_dir: Path,
    *,
    periods: int = 252,
    spa_iterations: int = 500,
    block_length: int | None = None,
    seed: int = 0,
) -> Dict[str, Path]:
    """Aggregate per-run metrics and export paper-ready artefacts."""

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_frames = []
    returns_map: Dict[str, pd.Series] = {}
    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
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

    if not returns_map:
        raise RuntimeError(f"No returns series discovered under {results_root}")

    artifact_paths: Dict[str, Path] = {}

    metrics_path = out_dir / "all_metrics_raw.csv"
    all_metrics.to_csv(metrics_path, index=False)
    artifact_paths["all_metrics"] = metrics_path

    if not all_metrics.empty:
        summary = (
            all_metrics.groupby("agent")[
                ["Sharpe", "Sortino", "MaxDD", "PnL", "Turnover", "Exposure"]
            ]
            .agg(["mean", "std"])
        )
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
        summary.sort_index(inplace=True)
        summary_path = out_dir / "summary_table.csv"
        summary.to_csv(summary_path)
        artifact_paths["summary_table"] = summary_path

        best_idx = all_metrics.groupby("agent")["Sharpe"].idxmax()
        best = all_metrics.loc[best_idx].sort_values("Sharpe", ascending=False)
        best_path = out_dir / "best_per_agent.csv"
        best.to_csv(best_path, index=False)
        artifact_paths["best_per_agent"] = best_path
    else:
        summary = pd.DataFrame()
        best = pd.DataFrame()

    advanced_records = []
    returns_cache: Dict[str, pd.Series] = {}
    for run_id, returns in returns_map.items():
        returns_cache[run_id] = returns
        psr = stats.probabilistic_sharpe_ratio(returns, periods_per_year=periods)
        dsr = stats.deflated_sharpe_ratio(
            returns,
            periods_per_year=periods,
            num_trials=max(1, len(returns_map)),
        )
        hac_low, hac_high = stats.hac_sharpe_confidence_interval(returns, periods_per_year=periods)
        sharpe = stats.annualized_sharpe(returns, periods_per_year=periods)
        sortino = stats.sortino_ratio(returns, periods_per_year=periods)
        advanced_records.append(
            {
                "run_id": run_id,
                "n_obs": len(returns),
                "sharpe": sharpe,
                "sortino": sortino,
                "psr": psr,
                "dsr": dsr,
                "hac_sharpe_lower": hac_low,
                "hac_sharpe_upper": hac_high,
            }
        )
    advanced_df = pd.DataFrame.from_records(advanced_records)
    advanced_path = out_dir / "advanced_metrics.csv"
    advanced_df.to_csv(advanced_path, index=False)
    artifact_paths["advanced_metrics"] = advanced_path
    if not advanced_df.empty:
        psr_path = out_dir / "psr_dsr_pvalues.csv"
        advanced_df[["run_id", "psr", "dsr"]].to_csv(psr_path, index=False)
        artifact_paths["psr_dsr"] = psr_path

        hac_path = out_dir / "hac_sharpe_ci.csv"
        advanced_df[["run_id", "hac_sharpe_lower", "hac_sharpe_upper"]].to_csv(
            hac_path, index=False
        )
        artifact_paths["hac_sharpe_ci"] = hac_path

        forest_path = out_dir / "forest.png"
        _plot_hac_forest(advanced_df, forest_path)
        artifact_paths["forest_plot"] = forest_path

    spa_df = stats.spa_reality_check(
        returns_cache,
        periods_per_year=periods,
        iterations=spa_iterations,
        block_length=block_length,
        seed=seed,
    )
    spa_path = out_dir / "spa_table.csv"
    spa_df.to_csv(spa_path, index=False)
    artifact_paths["spa_results"] = spa_path
    if not spa_df.empty:
        spa_pvalues_path = out_dir / "spa_pvalues.csv"
        spa_df[["run_id", "spa_pvalue", "spa_sup_pvalue"]].to_csv(spa_pvalues_path, index=False)
        artifact_paths["spa_pvalues"] = spa_pvalues_path

    mcs_members = stats.model_confidence_set(
        returns_cache,
        periods_per_year=periods,
        iterations=spa_iterations,
        block_length=block_length,
        seed=seed,
    )
    mcs_path = out_dir / "mcs_table.csv"
    mcs_frame = pd.DataFrame({"run_id": mcs_members})
    mcs_frame.to_csv(mcs_path, index=False)
    artifact_paths["mcs_table"] = mcs_path

    heatmap_path = out_dir / "heatmap.png"
    _plot_sharpe_heatmap(all_metrics, heatmap_path)
    artifact_paths["heatmap"] = heatmap_path

    pnl_path = out_dir / "pnl.png"
    _plot_pnl_bars(all_metrics, pnl_path)
    artifact_paths["pnl_plot"] = pnl_path

    summary_lines = ["# Paper Results", ""]
    summary_lines.append("## Artifacts")
    for label, filename in [
        ("All metrics", "all_metrics_raw.csv"),
        ("Summary table", "summary_table.csv"),
        ("Best per agent", "best_per_agent.csv"),
        ("Advanced metrics", "advanced_metrics.csv"),
        ("HAC Sharpe CI", "hac_sharpe_ci.csv"),
        ("PSR/DSR p-values", "psr_dsr_pvalues.csv"),
        ("SPA table", "spa_table.csv"),
        ("MCS members", "mcs_table.csv"),
        ("Forest plot", "forest.png"),
        ("Heatmap", "heatmap.png"),
        ("PnL plot", "pnl.png"),
    ]:
        if (out_dir / filename).exists():
            summary_lines.append(f"- [{label}]({filename})")

    if not all_metrics.empty:
        summary_lines.append("\n## Dataset")
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
    summary_path = out_dir / "paper_results.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    artifact_paths["summary_markdown"] = summary_path

    return artifact_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Root directory with run outputs")
    parser.add_argument("--out", type=Path, required=True, help="Directory to store aggregated artifacts")
    parser.add_argument("--periods", type=int, default=252, help="Trading periods per year")
    parser.add_argument("--spa-iterations", type=int, default=500, help="Bootstrap iterations for SPA")
    parser.add_argument("--block-length", type=int, default=None, help="Block length for stationary bootstrap")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap routines")
    args = parser.parse_args()

    run_workflow(
        args.results,
        args.out,
        periods=args.periods,
        spa_iterations=args.spa_iterations,
        block_length=args.block_length,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
