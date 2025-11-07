"""Aggregation utilities and CLI for paper-ready result tables.

Examples
--------
Aggregate results and emit a JSON summary::

    python -m leadlag.reporting.main_results \
        --results results/paper_run \
        --out paper_outputs \
        --winsor 0.001 \
        --format json

Use ``--format text`` (the default) for a concise human-readable summary.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from leadlag.cli.errors import emit_exception
from leadlag.cli.formatters import (
    add_format_flags,
    emit_formatted_output,
    finalize_format_args,
)

from .latex import to_latex

from .metrics_writer import enforce_metrics_schema

KEY_METRICS: Sequence[str] = (
    "Sharpe",
    "Sortino",
    "MaxDD",
    "PnL",
    "Turnover",
    "Exposure",
    "EnvSteps",
)

MAIN_GROUP_COLUMNS: Sequence[str] = (
    "agent",
    "policy",
    "universe",
    "timeframe",
    "split_scheme",
    "reward",
    "features_signature",
    "signature_depth",
    "features_leadlag",
    "time_channel",
    "cost_fee_bps",
    "slippage_bps",
)

ABLATION_GROUP_COLUMNS: Sequence[str] = (
    "agent",
    "universe",
    "timeframe",
    "features_signature",
    "signature_depth",
    "features_leadlag",
    "time_channel",
    "cost_fee_bps",
    "slippage_bps",
)


@dataclass(frozen=True)
class AggregateResult:
    """Returned object from :func:`aggregate_main_results`."""

    main_results: pd.DataFrame
    ablations: pd.DataFrame
    all_metrics: pd.DataFrame


def _load_metrics_from_results(results_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metrics_path in sorted(results_root.glob("*/metrics.csv")):
        frame = pd.read_csv(metrics_path)
        coerced = enforce_metrics_schema(frame)
        coerced["run_dir"] = metrics_path.parent.name
        frames.append(coerced)
    if not frames:
        raise RuntimeError(f"No metrics.csv files discovered under {results_root}")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _load_existing_candidate(paths: Iterable[Path]) -> pd.DataFrame | None:
    for path in paths:
        if path is None:
            continue
        if path.exists():
            frame = pd.read_csv(path)
            coerced = enforce_metrics_schema(frame)
            if "run_dir" not in frame.columns and "run_dir" not in coerced.columns:
                coerced["run_dir"] = pd.NA
            return coerced
    return None


def _winsorise(series: pd.Series, alpha: float | None) -> pd.Series:
    if alpha is None or alpha <= 0:
        return series
    if len(series) < 2:
        return series
    lower = series.quantile(alpha)
    upper = series.quantile(1 - alpha)
    return series.clip(lower=lower, upper=upper)


def _summarise(values: pd.Series) -> tuple[float, float, float, float]:
    if values.empty:
        return (math.nan, math.nan, math.nan, math.nan)
    mean = float(values.mean())
    if len(values) < 2:
        return (mean, math.nan, math.nan, math.nan)
    std = float(values.std(ddof=1))
    if math.isnan(std):
        return (mean, math.nan, math.nan, math.nan)
    half_width = 1.96 * std / math.sqrt(len(values))
    return (mean, std, mean - half_width, mean + half_width)


def _summarise_series(series: pd.Series, winsor_alpha: float | None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    numeric = _winsorise(numeric, winsor_alpha)
    mean, std, lower, upper = _summarise(numeric)
    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    )


def _aggregate(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    *,
    winsor_alpha: float | None,
) -> pd.DataFrame:
    working = frame.copy()
    for column in group_columns:
        if column not in working.columns:
            working[column] = pd.NA
    if working.empty:
        empty_columns = list(group_columns) + ["n_runs", "n_seeds", "n_windows"]
        return pd.DataFrame(columns=empty_columns)

    grouped = working.groupby(list(group_columns), dropna=False, sort=True)
    counts = grouped.size().rename("n_runs")
    result = counts.to_frame()

    if "seed" in working.columns:
        seeds = grouped["seed"].nunique(dropna=True)
        result["n_seeds"] = seeds.astype("Int64")
    else:
        result["n_seeds"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    if "window_index" in working.columns:
        windows = grouped["window_index"].nunique(dropna=True)
        result["n_windows"] = windows.astype("Int64")
    else:
        result["n_windows"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    result.reset_index(inplace=True)

    for metric in metrics:
        if metric not in working.columns:
            continue
        summary = grouped[metric].apply(_summarise_series, winsor_alpha=winsor_alpha)
        summary = summary.unstack()
        summary = summary.rename(
            columns={
                "mean": f"{metric}",
                "std": f"{metric}_std",
                "ci_lower": f"{metric}_lo",
                "ci_upper": f"{metric}_hi",
            }
        )
        summary = summary.reset_index()
        result = result.merge(summary, on=list(group_columns), how="left")

    metric_columns: list[str] = []
    for metric in metrics:
        if metric not in working.columns:
            continue
        metric_columns.extend(
            [
                f"{metric}",
                f"{metric}_std",
                f"{metric}_lo",
                f"{metric}_hi",
            ]
        )

    ordered_columns = list(group_columns) + ["n_runs", "n_seeds", "n_windows"] + metric_columns
    # Ensure all expected columns are present even if entirely missing
    for column in ordered_columns:
        if column not in result.columns:
            result[column] = math.nan
    result = result.loc[:, ordered_columns]
    result.sort_values(list(group_columns), inplace=True, na_position="last")
    result.reset_index(drop=True, inplace=True)
    return result


def aggregate_main_results(
    results_root: Path,
    out_dir: Path,
    *,
    winsor_alpha: float | None = None,
    seed_aggregate: Path | None = None,
) -> AggregateResult:
    results_root = results_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_candidate(
        [seed_aggregate, out_dir / "aggregate.csv", out_dir / "all_metrics_raw.csv"]
    )
    if existing is not None and not existing.empty:
        all_metrics = existing
    else:
        all_metrics = _load_metrics_from_results(results_root)

    all_metrics_path = out_dir / "all_metrics_raw.csv"
    all_metrics.to_csv(all_metrics_path, index=False)

    main_df = _aggregate(
        all_metrics,
        MAIN_GROUP_COLUMNS,
        KEY_METRICS,
        winsor_alpha=winsor_alpha,
    )
    if winsor_alpha is not None and winsor_alpha > 0:
        main_df["winsor_alpha"] = float(winsor_alpha)
    else:
        main_df["winsor_alpha"] = 0.0

    ablations_df = _aggregate(
        all_metrics,
        ABLATION_GROUP_COLUMNS,
        KEY_METRICS,
        winsor_alpha=winsor_alpha,
    )
    if winsor_alpha is not None and winsor_alpha > 0:
        ablations_df["winsor_alpha"] = float(winsor_alpha)
    else:
        ablations_df["winsor_alpha"] = 0.0

    for column in ("n_runs", "n_seeds", "n_windows"):
        if column in main_df.columns:
            main_df[column] = main_df[column].astype("Int64")
        if column in ablations_df.columns:
            ablations_df[column] = ablations_df[column].astype("Int64")

    main_path = out_dir / "main_results.csv"
    ablations_path = out_dir / "ablations.csv"
    main_df.to_csv(main_path, index=False)
    ablations_df.to_csv(ablations_path, index=False)

    main_tex_path = out_dir / "main_results.tex"
    ablations_tex_path = out_dir / "ablations.tex"
    to_latex(main_path, main_tex_path, metrics=KEY_METRICS)
    to_latex(ablations_path, ablations_tex_path, metrics=KEY_METRICS)

    return AggregateResult(main_df, ablations_df, all_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Root directory containing run outputs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Directory to write aggregated paper tables (CSV and LaTeX)."
        ),
    )
    parser.add_argument(
        "--winsor",
        type=float,
        default=0.0,
        help="Optional winsorisation alpha (e.g. 0.001).",
    )
    parser.add_argument(
        "--seed-aggregate",
        type=Path,
        default=None,
        help="Optional existing aggregate.csv to reuse.",
    )
    add_format_flags(parser, default="text")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    finalize_format_args(args)

    winsor_alpha = args.winsor if args.winsor and args.winsor > 0 else None

    try:
        aggregate = aggregate_main_results(
            args.results,
            args.out,
            winsor_alpha=winsor_alpha,
            seed_aggregate=args.seed_aggregate,
        )
    except Exception as exc:  # pragma: no cover - exercised in CLI tests
        emit_exception(args, exc, message="Failed to aggregate paper tables.")
        return 1

    out_dir = Path(args.out).resolve()
    main_path = out_dir / "main_results.csv"
    ablations_path = out_dir / "ablations.csv"
    all_metrics_path = out_dir / "all_metrics_raw.csv"
    main_tex_path = out_dir / "main_results.tex"
    ablations_tex_path = out_dir / "ablations.tex"

    text_lines = [
        f"Output directory: {out_dir}",
        f"Main results: {main_path} (rows={len(aggregate.main_results)})",
        f"Main results (LaTeX): {main_tex_path}",
        f"Ablations: {ablations_path} (rows={len(aggregate.ablations)})",
        f"Ablations (LaTeX): {ablations_tex_path}",
    ]
    message = "Aggregation completed."

    payload = {
        "results_root": str(Path(args.results).resolve()),
        "output_dir": str(out_dir),
        "winsor_alpha": float(winsor_alpha or 0.0),
        "tables": {
            "main_results": {
                "path": str(main_path),
                "rows": int(len(aggregate.main_results)),
                "columns": list(aggregate.main_results.columns),
                "latex_path": str(main_tex_path),
            },
            "ablations": {
                "path": str(ablations_path),
                "rows": int(len(aggregate.ablations)),
                "columns": list(aggregate.ablations.columns),
                "latex_path": str(ablations_tex_path),
            },
            "all_metrics": {
                "path": str(all_metrics_path),
                "rows": int(len(aggregate.all_metrics)),
                "columns": list(aggregate.all_metrics.columns),
            },
        },
    }

    artifacts = {
        "main_results": str(main_path),
        "main_results_tex": str(main_tex_path),
        "ablations": str(ablations_path),
        "ablations_tex": str(ablations_tex_path),
        "all_metrics": str(all_metrics_path),
    }

    emit_formatted_output(
        args,
        text="\n".join(text_lines),
        message=message,
        data=payload,
        artifacts=artifacts,
        pretty=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
