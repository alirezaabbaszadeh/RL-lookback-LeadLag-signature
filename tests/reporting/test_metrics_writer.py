"""Tests for metrics writer metadata extraction."""

from __future__ import annotations

from leadlag.reporting.metrics_writer import build_metadata_row


def test_build_metadata_row_prefers_agent_policy_mapping() -> None:
    cfg = {
        "agent": {"name": "ExampleAgent", "policy": {"name": "agent_policy"}},
        "policy": {"name": "legacy_policy"},
    }

    row = build_metadata_row(
        run_id="exp-123",
        cfg=cfg,
        metrics={},
        seed=42,
        window_idx=0,
    )

    assert row["policy"] == "agent_policy"


def test_build_metadata_row_falls_back_to_global_policy() -> None:
    cfg = {
        "agent": {"name": "ExampleAgent"},
        "policy": {"name": "legacy_policy"},
    }

    row = build_metadata_row(
        run_id="exp-456",
        cfg=cfg,
        metrics={},
        seed=21,
        window_idx=1,
    )

    assert row["policy"] == "legacy_policy"
