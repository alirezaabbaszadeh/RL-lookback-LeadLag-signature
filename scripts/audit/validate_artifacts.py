from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Finding:
    path: str
    level: str  # INFO | WARN | ERROR
    message: str
    details: Dict[str, object]


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def validate_stats_csv(p: Path) -> List[Finding]:
    findings: List[Finding] = []
    df = _read_csv(p)
    if df is None:
        findings.append(Finding(str(p), "ERROR", "Failed to read stats.csv", {}))
        return findings
    required = {"scenario", "metric", "count"}
    if not required.issubset(df.columns):
        missing = sorted(list(required.difference(df.columns)))
        findings.append(Finding(str(p), "ERROR", "stats.csv missing required columns", {"missing": missing}))
    # At least one of the aggregated columns should exist
    expected_any = {
        "mean_mean",
        "median_mean",
        "std_mean",
        "max_mean",
    }
    if expected_any.isdisjoint(df.columns):
        findings.append(Finding(str(p), "WARN", "stats.csv lacks aggregated *_mean columns", {"have": list(df.columns)}))
    return findings


def validate_significance_csv(p: Path) -> List[Finding]:
    findings: List[Finding] = []
    df = _read_csv(p)
    if df is None:
        findings.append(Finding(str(p), "ERROR", "Failed to read significance.csv", {}))
        return findings
    # Check for truncated bootstrap keys
    bad_cols = [c for c in df.columns if c.startswith("mea_boot_") or c.startswith("media_boot_") or c.startswith("st_boot_") or c.startswith("ma_boot_") or c.startswith("mi_boot_")]
    if bad_cols:
        findings.append(Finding(str(p), "ERROR", "Truncated bootstrap key names detected", {"columns": bad_cols}))
    # Prefer mean_boot_low/high presence
    want = {"mean_boot_low", "mean_boot_high"}
    if not want.issubset(df.columns):
        findings.append(Finding(str(p), "WARN", "mean_boot_low/high not present", {"have": list(df.columns)}))
    return findings


def validate_run_dir(run_dir: Path) -> List[Finding]:
    findings: List[Finding] = []
    # Required files
    required = ["summary.csv", "run_metadata.json"]
    for name in required:
        if not (run_dir / name).exists():
            findings.append(Finding(str(run_dir), "ERROR", f"Missing required file: {name}", {}))
    # Optional metrics_timeseries.csv
    mts = run_dir / "metrics_timeseries.csv"
    if not mts.exists():
        findings.append(Finding(str(run_dir), "WARN", "metrics_timeseries.csv not found (plots & KPIs may be limited)", {}))
    # Metadata checks
    meta_p = run_dir / "run_metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if not meta.get("data_price_hash"):
                findings.append(Finding(str(meta_p), "WARN", "data_price_hash missing in run metadata", {}))
            git = meta.get("git", {}) or {}
            if git.get("git_commit") is None:
                findings.append(Finding(str(meta_p), "WARN", "git commit not recorded", {}))
        except Exception:
            findings.append(Finding(str(meta_p), "ERROR", "Failed to parse run_metadata.json", {}))
    return findings


def discover_aggregates(root: Path) -> List[Path]:
    return [p for p in root.rglob("*_aggregate") if p.is_dir()]


def discover_run_dirs(root: Path) -> List[Path]:
    # seed dirs pattern: <name>_seed*
    return [p for p in root.rglob("*_seed*") if p.is_dir()]


def main() -> int:
    results_root = Path("results")
    out_dir = Path("docs/audit/phase-1")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_findings: List[Finding] = []

    # Validate aggregates
    for agg in discover_aggregates(results_root):
        stats_p = agg / "stats.csv"
        if stats_p.exists():
            all_findings.extend(validate_stats_csv(stats_p))
        else:
            all_findings.append(Finding(str(agg), "WARN", "stats.csv not found in aggregate", {}))
        sig_p = agg / "significance.csv"
        if sig_p.exists():
            all_findings.extend(validate_significance_csv(sig_p))
        # welch.csv is optional

    # Validate runs
    for run in discover_run_dirs(results_root):
        all_findings.extend(validate_run_dir(run))

    # Write JSON report
    report_json = [asdict(f) for f in all_findings]
    (out_dir / "scan_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    # Write Markdown report
    lines: List[str] = []
    lines.append("# Phase 1 — Artifact Scan Report")
    lines.append("")
    if not all_findings:
        lines.append("No findings.")
    else:
        summary: Dict[str, int] = {"INFO": 0, "WARN": 0, "ERROR": 0}
        for f in all_findings:
            summary[f.level] = summary.get(f.level, 0) + 1
        lines.append(f"Summary: INFO={summary.get('INFO',0)} WARN={summary.get('WARN',0)} ERROR={summary.get('ERROR',0)}")
        lines.append("")
        lines.append("| Level | Path | Message | Details |")
        lines.append("| --- | --- | --- | --- |")
        for f in all_findings:
            details = json.dumps(f.details, ensure_ascii=False)
            lines.append(f"| {f.level} | {f.path} | {f.message} | {details} |")
    (out_dir / "scan_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(all_findings)} findings to {out_dir / 'scan_report.{json,md}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

