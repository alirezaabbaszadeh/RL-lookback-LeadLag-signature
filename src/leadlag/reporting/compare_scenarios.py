from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore

    MPL = True
except Exception:  # pragma: no cover
    MPL = False

from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
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


def plot_metric_bars(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "mean_abs_matrix",
) -> Optional[Path]:
    if not MPL or df.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filter to desired metric and display mean_mean per scenario
    needed_cols = {"scenario", "metric", "mean_mean"}
    if not needed_cols.issubset(df.columns):
        return None
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="scenario", values="mean_mean", aggfunc="mean")
    pivot.sort_values(by="mean_mean", ascending=False, inplace=True)
    plt.figure(figsize=(8, 4))
    pivot.plot(kind="bar", legend=False)
    plt.title(f"{metric} (mean_mean)")
    plt.tight_layout()
    out_path = out_dir / f"bar_{metric}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
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
    add_format_flags(ap, default="text")
    args = ap.parse_args(list(argv) if argv is not None else None)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-compare"
    if argv:
        command = "leadlag-compare " + " ".join(argv)

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
    summary: Dict[str, object] = {
        "results_root": str(args.results_root.resolve()),
        "output_dir": str(out_dir),
        "dry_run": bool(args.dry_run),
        "aggregates": [str(path) for path in agg_dirs],
    }
    artifacts: Dict[str, object] = {}

    text_lines: List[str] = []
    if agg_dirs:
        text_lines.append(f"Discovered {len(agg_dirs)} aggregate directories.")
    else:
        text_lines.append("No aggregate directories found.")

    if args.dry_run:
        for path in agg_dirs:
            logger.info("[dry-run] aggregate directory", context={"path": str(path)})
            text_lines.append(str(path))
        summary["artifacts"] = artifacts
        emit_formatted_output(
            args,
            data=summary,
            text="\n".join(text_lines),
            message="Comparison dry-run completed.",
            artifacts=artifacts or None,
            pretty=True,
            command=command,
        )
        return 0

    df = load_stats(agg_dirs)
    if df.empty:
        logger.warning("No aggregate stats found")
        summary["artifacts"] = artifacts
        text_lines.append("No aggregate stats found.")
        emit_formatted_output(
            args,
            data=summary,
            text="\n".join(text_lines),
            message="No aggregate stats found.",
            artifacts=artifacts or None,
            errors=[{"code": "no_stats", "message": "No aggregate stats found."}],
            pretty=True,
            command=command,
            success=False,
        )
        return 0

    comparison_csv = out_dir / "aggregate_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    plots_dir = out_dir / "plots"
    generated_plots: List[str] = []
    primary_plot = plot_metric_bars(df, plots_dir, args.metric)
    if primary_plot:
        generated_plots.append(str(primary_plot))
    for m in ["row_sum_std", "row_sum_range"]:
        try:
            plot_path = plot_metric_bars(df, plots_dir, m)
        except Exception as exc:  # pragma: no cover - plotting errors shouldn't stop CLI
            logger.warning("Plot generation failed", context={"metric": m, "error": repr(exc)})
        else:
            if plot_path:
                generated_plots.append(str(plot_path))

    artifacts = {
        "csv": str(comparison_csv),
        "plots": generated_plots,
    }
    summary["artifacts"] = artifacts
    text_lines.append(f"Wrote CSV: {comparison_csv}")
    if generated_plots:
        text_lines.append(f"Generated {len(generated_plots)} plot(s).")
    else:
        text_lines.append("No plots generated.")

    logger.info(
        "Saved comparison artifacts",
        context={"output": str(out_dir), "plots": len(generated_plots)},
    )
    emit_formatted_output(
        args,
        data=summary,
        text="\n".join(text_lines),
        message="Comparison artifacts generated.",
        artifacts=artifacts or None,
        pretty=True,
        command=command,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
