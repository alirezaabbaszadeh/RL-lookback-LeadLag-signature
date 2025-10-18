from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from governance.dataset import build_manifest, record_manifest, run_quality_checks
from training.run_scenario import _read_prices


def _load_prices(path: Path) -> tuple[object, Optional[Path]]:
    cfg = {
        "data": {"price_csv": str(path)},
        "run": {"seed": 42},
    }
    return _read_prices(cfg)


def _detect_issues(findings: Dict[str, object], *, missing_tolerance: float) -> List[str]:
    issues: List[str] = []
    if findings.get("duplicate_index"):
        issues.append("Index contains duplicates.")
    missing_ratio = float(findings.get("missing_ratio", 0.0))
    if missing_ratio > missing_tolerance:
        issues.append(f"Missing ratio {missing_ratio:.2%} exceeds tolerance {missing_tolerance:.2%}.")
    zero_variance = findings.get("zero_variance_assets", [])
    if zero_variance:
        issues.append(f"{len(zero_variance)} asset(s) with zero variance: {', '.join(zero_variance[:5])}")
    if not findings.get("monotonic_index", True):
        issues.append("Index is not monotonically increasing.")
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run governance checks over a price dataset.")
    parser.add_argument("--path", type=Path, help="Path to the price CSV file.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the dataset manifest JSON."
    )
    parser.add_argument(
        "--missing-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed missing value ratio before failing (default: 0.01 == 1%).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.path is None:
        parser.error("--path is required")

    prices, resolved_path = _load_prices(args.path)
    manifest = build_manifest(
        prices,
        source_path=resolved_path or args.path,
    )
    quality = run_quality_checks(prices)
    manifest["quality"] = quality

    issues = _detect_issues(quality, missing_tolerance=args.missing_tolerance)

    if args.output:
        record_manifest(manifest, args.output.parent, args.output.name)
    else:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if issues:
        sys.stderr.write("Dataset quality check failed:\n")
        for issue in issues:
            sys.stderr.write(f"- {issue}\n")
        return 1

    sys.stdout.write("Dataset quality check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
