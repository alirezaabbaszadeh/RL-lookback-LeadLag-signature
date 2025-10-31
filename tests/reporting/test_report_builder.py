from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leadlag.reporting.data_access import ScenarioAggregate
from leadlag.reporting.report_builder import ReportBuilder


def _make_dataframe(rows):
    return pd.DataFrame(rows)


def test_builder_creates_markdown_sections_with_metrics():
    stats = _make_dataframe(
        [
            {
                "scenario": "alpha",
                "metric": "mean_abs_matrix",
                "mean_mean": 0.5,
                "mean_std": 0.1,
            },
            {
                "scenario": "alpha",
                "metric": "row_sum_std",
                "mean_mean": 1.25,
                "mean_std": 0.2,
            },
        ]
    )
    significance = _make_dataframe(
        [
            {
                "scenario": "alpha",
                "metric": "mean_abs_matrix",
                "mea_boot_low": 0.45,
                "mea_boot_high": 0.55,
            }
        ]
    )
    welch = _make_dataframe(
        [
            {
                "scenario_a": "alpha",
                "scenario_b": "beta",
                "p_value": 0.0123,
            }
        ]
    )
    runs = [
        {
            "seed": 1,
            "created_at": "2023-09-01T00:00:00Z",
            "config_path": "config.yaml",
            "data_path": "data.csv",
            "python_version": "3.11.0",
            "platform": "linux",
        }
    ]
    aggregate = ScenarioAggregate(
        name="alpha",
        aggregate_dir=Path("/tmp/alpha_aggregate"),
        stats=stats,
        significance=significance,
        welch=welch,
        runs=runs,
    )

    builder = ReportBuilder(now_fn=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc))
    artifacts = builder.build([aggregate])

    assert "# Lead-Lag Signature RL Research Report" in artifacts.report_markdown
    assert "Generated on 2024-01-02 00:00:00Z" in artifacts.report_markdown
    assert "### Scenario: alpha" in artifacts.report_markdown
    assert "| mean_abs_matrix | 0.5000 | 0.1000 |" in artifacts.report_markdown
    assert "Bootstrap confidence intervals:" in artifacts.report_markdown
    assert "Welch significance checks:" in artifacts.report_markdown
    assert "Seed 1, created 2023-09-01T00:00:00Z" in artifacts.report_markdown

    assert artifacts.appendix_markdown.startswith("# Reproducibility Appendix")
    assert "| alpha | 1 | 2023-09-01T00:00:00Z |" in artifacts.appendix_markdown
    assert "`/tmp/alpha_aggregate`" in artifacts.appendix_markdown

    assert artifacts.aggregate_names == ["alpha"]
    assert artifacts.metadata["scenarios"] == ["alpha"]


def test_builder_handles_empty_aggregates():
    builder = ReportBuilder(now_fn=lambda: datetime(2024, 5, 1, tzinfo=timezone.utc))
    artifacts = builder.build([])

    assert "No aggregate data available to summarise." in artifacts.report_markdown
    assert "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in artifacts.appendix_markdown
    assert artifacts.aggregate_names == []
    assert artifacts.metadata["scenarios"] == []
