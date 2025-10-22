from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore

    MPL = True
except Exception:  # pragma: no cover
    MPL = False

from leadlag.reporting.logging_utils import get_logger, setup_logging


def find_aggregate_dirs(root: Path) -> List[Path]:
    # Search recursively to capture aggregates under results/manual/*
    return [p for p in root.rglob("*_aggregate") if p.is_dir()]


def load_stats(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        stats_p = p / "stats.csv"
        if stats_p.exists():
            df = pd.read_csv(stats_p)
            df["_aggregate_dir"] = str(p)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_metric_bars(df: pd.DataFrame, out_dir: Path, metric: str = "mean_abs_matrix") -> None:
    if not MPL or df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filter to desired metric and display mean_mean per scenario
    needed_cols = {"scenario", "metric", "mean_mean"}
    if not needed_cols.issubset(df.columns):
        return
    sub = df[df["metric"] == metric]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="scenario", values="mean_mean", aggfunc="mean")
    pivot.sort_values(by="mean_mean", ascending=False, inplace=True)
    plt.figure(figsize=(8, 4))
    pivot.plot(kind="bar", legend=False)
    plt.title(f"{metric} (mean_mean)")
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_{metric}.png", dpi=140)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare scenarios across aggregate stats")
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("evaluation"))
    ap.add_argument(
        "--metric",
        type=str,
        default="mean_abs_matrix",
        help="Metric name in stats.csv (column 'metric')",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    ap.add_argument(
        "--log-path",
        type=Path,
        help="Optional log file path.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List aggregate directories and exit without writing artifacts.",
    )
    args = ap.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_path or out_dir / "compare.log"
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "compare"})
    logger = get_logger(
        "reporting.compare_scenarios",
        context={"results_root": args.results_root.resolve()},
    )

    agg_dirs = find_aggregate_dirs(args.results_root)
    logger.info("Discovered aggregate directories", context={"count": len(agg_dirs)})
    if args.dry_run:
        for path in agg_dirs:
            logger.info("[dry-run] aggregate directory", context={"path": str(path)})
        return 0

    df = load_stats(agg_dirs)
    if df.empty:
        logger.warning("No aggregate stats found")
        return 0

    df.to_csv(out_dir / "aggregate_comparison.csv", index=False)

    # Example key plots (mean_mean of chosen metrics)
    plots_dir = out_dir / "plots"
    plot_metric_bars(df, plots_dir, args.metric)
    for m in ["row_sum_std", "row_sum_range"]:
        try:
            plot_metric_bars(df, plots_dir, m)
        except Exception as exc:  # pragma: no cover - plotting errors shouldn't stop CLI
            logger.warning("Plot generation failed", context={"metric": m, "error": repr(exc)})

    logger.info("Saved comparison artifacts", context={"output": str(out_dir)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
