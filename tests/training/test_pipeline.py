from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from leadlag.training.pipeline import ScenarioPipeline, ScenarioPipelineHooks
from leadlag.training.run_support import RunPreparation


def test_pipeline_coordinates_analysis_and_hooks(tmp_path: Path) -> None:
    cfg = {
        "run": {"run_name": "stub"},
        "analysis": {"method": "signature", "lookback": 30},
    }

    prices = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    rolling_index = pd.date_range("2021-01-01", periods=2, freq="D")
    rolling = pd.Series(
        [pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": [3.0, 4.0]})],
        index=rolling_index,
    )

    metrics_df = pd.DataFrame({"metric": [1.0, 2.0]}, index=rolling_index)
    summary_df = pd.DataFrame({"metric": ["foo"], "value": [42.0]})

    artifact_calls: list[tuple[bool, bool]] = []
    hook_calls: list[str] = []
    analysis_calls: list[str] = []

    class DummyAnalyzer:
        def __init__(self) -> None:
            self.called = False

        def analyze(self, prices, *, return_rolling=True):
            self.called = True
            analysis_calls.append(f"analyze:{prices.shape}")
            return rolling

    def analyzer_factory(config):
        assert config is cfg
        return DummyAnalyzer()

    def analysis_runner(analyzer: DummyAnalyzer, prices, config):
        assert config is cfg
        assert isinstance(analyzer, DummyAnalyzer)
        return analyzer.analyze(prices, return_rolling=True)

    def metrics_computer(_rolling):
        assert _rolling is rolling
        return metrics_df

    def metrics_summarizer(df):
        assert df is metrics_df
        return summary_df

    def artifact_generator(context):
        artifact_calls.append((context.metrics_path.exists(), context.summary_path.exists()))

    hooks = ScenarioPipelineHooks(
        mlflow=lambda context: hook_calls.append(f"mlflow:{context.summary_path.name}"),
        plotting=lambda context: hook_calls.append(f"plot:{len(context.metrics)}"),
        extras=[lambda context: hook_calls.append(f"extra:{context.cfg['run']['run_name']}")],
    )

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    preparation = RunPreparation(
        out_dir=tmp_path,
        logger=logging.getLogger("test"),
        prices=prices,
        manifest_path=tmp_path / "manifest.json",
        timestamp="ts",
        seed=123,
        run_name="stub",
        resolved_price_path=None,
    )

    pipeline = ScenarioPipeline(
        analyzer_factory=analyzer_factory,
        analysis_runner=analysis_runner,
        metrics_computer=metrics_computer,
        metrics_summarizer=metrics_summarizer,
        artifact_generators=[artifact_generator],
        hooks=hooks,
    )

    result = pipeline.run(cfg, preparation)

    assert result.out_dir == tmp_path
    assert (tmp_path / "metrics_timeseries.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert artifact_calls == [(True, True)]
    assert "analyze:" in analysis_calls[0]
    assert hook_calls == ["mlflow:summary.csv", "plot:2", "extra:stub"]
