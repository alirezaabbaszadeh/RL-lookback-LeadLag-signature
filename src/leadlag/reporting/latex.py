"""Utilities for exporting aggregated results to LaTeX tables.

The :func:`to_latex` helper mirrors the CSV layout produced by
``leadlag.reporting.main_results`` while formatting numeric metrics with
SIunitx-friendly commands (``\num``/``\numrange``).  This keeps the CSV artefacts
machine-readable and provides paper-ready tables alongside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import math

import pandas as pd


Number = float | int


def _as_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def _format_number(value: Number, digits: int) -> str:
    return f"{value:.{digits}f}"


def _format_ci(value: Number, lower: Number, upper: Number, digits: int) -> str:
    value_fmt = _format_number(value, digits)
    lower_fmt = _format_number(lower, digits)
    upper_fmt = _format_number(upper, digits)
    return f"\\num{{{value_fmt}}}~(\\numrange{{{lower_fmt}}}{{{upper_fmt}}})"


def _maybe_ci_string(
    row: Mapping[str, object],
    metric: str,
    digits: int,
    na_rep: str,
) -> str:
    base = row.get(metric)
    if base is None or (isinstance(base, float) and math.isnan(base)):
        return na_rep

    lo_key = f"{metric}_lo"
    hi_key = f"{metric}_hi"
    lower = row.get(lo_key)
    upper = row.get(hi_key)

    if lower is None or upper is None:
        return f"\\num{{{_format_number(float(base), digits)}}}"

    if any(
        isinstance(val, float) and math.isnan(val)
        for val in (lower, upper)
    ):
        return f"\\num{{{_format_number(float(base), digits)}}}"

    return _format_ci(float(base), float(lower), float(upper), digits)


def _detect_metric_columns(columns: Iterable[str]) -> list[str]:
    metrics: list[str] = []
    seen = set()
    for column in columns:
        if column.endswith("_lo") or column.endswith("_hi"):
            base = column.rsplit("_", 1)[0]
            if base not in seen:
                metrics.append(base)
                seen.add(base)
    return metrics


def _prepare_dataframe(
    frame: pd.DataFrame,
    metrics: Sequence[str] | None,
    *,
    digits: int,
    na_rep: str,
) -> pd.DataFrame:
    working = frame.copy()

    metric_names = list(metrics) if metrics else _detect_metric_columns(working.columns)

    for metric in metric_names:
        if metric not in working.columns:
            continue

        ci_values = working.apply(
            lambda row: _maybe_ci_string(row, metric, digits, na_rep), axis=1
        )
        working[metric] = ci_values

        for suffix in ("_std", "_lo", "_hi"):
            column = f"{metric}{suffix}"
            if column in working.columns:
                working.drop(columns=column, inplace=True)

    return working


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

ROW_TERMINATOR = " " + ("\\" * 2)


def _escape_latex(text: str) -> str:
    escaped = []
    for char in text:
        escaped.append(LATEX_SPECIALS.get(char, char))
    return "".join(escaped)


def _format_cell(value: object, *, na_rep: str) -> str:
    if value is None:
        return na_rep
    if isinstance(value, float) and math.isnan(value):
        return na_rep
    text = str(value)
    if text == "<NA>":
        return na_rep
    if text.startswith("\\"):
        return text
    return _escape_latex(text)


def _render_basic_table(
    frame: pd.DataFrame,
    *,
    index: bool,
    na_rep: str,
    column_format: str | None,
) -> str:
    columns = list(frame.columns)
    if index:
        columns = [""] + columns
    if column_format is None:
        align_cols = "l" * len(columns)
    else:
        align_cols = column_format

    lines = [f"\\begin{{tabular}}{{{align_cols}}}"]
    lines.append("\\toprule")
    header_cells = [_escape_latex(str(col)) for col in columns]
    lines.append(" & ".join(header_cells) + ROW_TERMINATOR)
    lines.append("\\midrule")

    if frame.empty:
        lines.append(ROW_TERMINATOR)
    else:
        for idx, row in frame.iterrows():
            cells = []
            if index:
                cells.append(_escape_latex(str(idx)))
            for value in row:
                cells.append(_format_cell(value, na_rep=na_rep))
            lines.append(" & ".join(cells) + ROW_TERMINATOR)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def to_latex(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    digits: int = 3,
    na_rep: str = r"\textemdash",
    column_format: str | None = None,
    index: bool = False,
) -> Path:
    """Render a CSV table produced by ``main_results`` into LaTeX.

    Parameters
    ----------
    csv_path:
        Source CSV path.  The file should follow the schema emitted by
        :func:`leadlag.reporting.main_results.aggregate_main_results`.
    out_path:
        Destination ``.tex`` path.
    metrics:
        Optional iterable of metric names to format using the
        ``value ~(low, high)`` template.  When omitted, all metrics with ``_lo``
        or ``_hi`` companions are automatically detected.
    digits:
        Number of decimal places for metric formatting.
    na_rep:
        Replacement string for missing values.
    column_format:
        Optional ``pandas.DataFrame.to_latex`` ``column_format`` argument.
    index:
        Whether to include the DataFrame index.
    """

    csv_path = _as_path(csv_path)
    out_path = _as_path(out_path)

    frame = pd.read_csv(csv_path)
    prepared = _prepare_dataframe(frame, metrics, digits=digits, na_rep=na_rep)

    latex_text = _render_basic_table(
        prepared, index=index, na_rep=na_rep, column_format=column_format
    )
    out_path.write_text(latex_text, encoding="utf-8")
    return out_path


__all__ = ["to_latex"]
