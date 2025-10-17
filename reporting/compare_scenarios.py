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


def find_aggregate_dirs(root: Path) -> List[Path]:
    return [p for p in root.glob("*_aggregate") if p.is_dir()]


def load_stats(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        stats_p = p / "stats.csv"
        if stats_p.exists():
            df = pd.read_csv(stats_p)
            df["_aggregate_dir"] = str(p)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_metric_bars(df: pd.DataFrame, out_dir: Path, column: str = "mean_abs_matrix_mean") -> None:
    if not MPL or df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Expect columns: scenario, metric, <numerics> from aggregated stats
    if column not in df.columns:
        return
    pivot = df.pivot_table(index="scenario", values=column, aggfunc="mean")
    pivot.sort_values(by=column, ascending=False, inplace=True)
    plt.figure(figsize=(8, 4))
    pivot.plot(kind="bar", legend=False)
    plt.title(column)
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_{column}.png", dpi=140)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare scenarios across aggregate stats")
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("evaluation"))
    args = ap.parse_args()

    agg_dirs = find_aggregate_dirs(args.results_root)
    df = load_stats(agg_dirs)
    if df.empty:
        print("No aggregate stats found.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "aggregate_comparison.csv", index=False)

    # Example key plots
    plot_metric_bars(df, args.out / "plots", "mean_abs_matrix_mean")
    plot_metric_bars(df, args.out / "plots", "row_sum_std_mean")

    print(f"Saved comparison CSV and plots to: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

