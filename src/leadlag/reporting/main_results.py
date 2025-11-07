"""Aggregation utilities for paper-ready result tables."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

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
        empty_columns = list(group_columns) + ["n_runs"]
        return pd.DataFrame(columns=empty_columns)

    grouped = working.groupby(list(group_columns), dropna=False, sort=True)
    counts = grouped.size().rename("n_runs").reset_index()

    result = counts
    for metric in metrics:
        if metric not in working.columns:
            continue
        summary = grouped[metric].apply(_summarise_series, winsor_alpha=winsor_alpha)
        summary = summary.unstack()
        summary = summary.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "ci_lower": f"{metric}_ci_lower",
                "ci_upper": f"{metric}_ci_upper",
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
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci_lower",
                f"{metric}_ci_upper",
            ]
        )

    ordered_columns = list(group_columns) + ["n_runs"] + metric_columns
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

    main_path = out_dir / "main_results.csv"
    ablations_path = out_dir / "ablations.csv"
    main_df.to_csv(main_path, index=False)
    ablations_df.to_csv(ablations_path, index=False)

    return AggregateResult(main_df, ablations_df, all_metrics)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        help="Directory to write aggregated paper tables.",
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
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    winsor_alpha = args.winsor if args.winsor and args.winsor > 0 else None
    aggregate_main_results(
        args.results,
        args.out,
        winsor_alpha=winsor_alpha,
        seed_aggregate=args.seed_aggregate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
