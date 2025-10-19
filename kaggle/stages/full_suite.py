"""Run the project's full experiment and audit suite as a single stage.

This calls pipelines/run_full_suite.py and captures artifacts under the
stage output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_full_suite(output_root: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "pipelines" / "run_full_suite.py"),
        "--output-root",
        str(output_root),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-suite stage")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    result = run_full_suite(run_root)

    (args.output_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (args.output_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

    meta = {
        "stage": "full_suite",
        "timestamp": time.time(),
        "returncode": result.returncode,
        "output_root": str(run_root),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

