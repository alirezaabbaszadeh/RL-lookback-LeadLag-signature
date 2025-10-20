"""Grand one-command runner for Kaggle.

Performs optional wheel prefetch for speed, sets local pip cache, then executes
the multi-stack orchestrator to run the complete experiment suite without
duplicates (full suite + production RL + dopamine).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd: list[str]) -> None:
    print("[run_all]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prefetch_wheels(wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    # Core
    run([sys.executable, "-m", "pip", "download", "-d", str(wheelhouse), "-r", str(ROOT / "requirements-kaggle.txt")])
    # RL stack (Gymnasium + SB3 2.x)
    run([sys.executable, "-m", "pip", "download", "-d", str(wheelhouse),
         "gymnasium==0.29.1", "stable-baselines3==2.1.0", "sb3-contrib==2.1.0", "torch>=2.1,<2.7"])
    # Dopamine stack (Gymnasium 1.x)
    run([sys.executable, "-m", "pip", "download", "-d", str(wheelhouse), "dopamine-rl==4.1.2", "gymnasium==1.0.0"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Grand one-command Kaggle runner")
    ap.add_argument("--artifacts-root", type=Path, default=None)
    ap.add_argument("--no-prefetch", action="store_true", help="Skip wheel prefetch step")
    args = ap.parse_args()

    artifacts_root = args.artifacts_root or (Path("/kaggle/working") / "multi_stage_artifacts")
    artifacts_root = artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    wheelhouse = Path("/kaggle/working/wheelhouse").resolve()
    pip_cache = Path("/kaggle/working/.cache/pip").resolve()
    pip_cache.mkdir(parents=True, exist_ok=True)

    if not args.no_prefetch:
        prefetch_wheels(wheelhouse)

    # Speed up installs by using local wheels
    os.environ.setdefault("PIP_CACHE_DIR", str(pip_cache))
    os.environ.setdefault("PIP_FIND_LINKS", str(wheelhouse))
    os.environ.setdefault("PIP_NO_INDEX", "1")

    # Provide sensible defaults for production RL unless user overrides
    os.environ.setdefault("SB3_DEVICE", "auto")
    os.environ.setdefault("SB3_TIMESTEPS", "300000")

    # Execute comprehensive stages in order:
    #   1) full_suite (baselines, ablations, audits, reports)
    #   2) sb3_leadlag (production RL on LeadLag)
    #   3) dopamine (Gymnasium 1.x sanity)
    run([
        sys.executable,
        str(HERE / "run_multi_stage.py"),
        "--artifacts-root",
        str(artifacts_root),
        "--stage", "full_suite",
        "--stage", "sb3_leadlag",
        "--stage", "dopamine",
    ])

    # Bundle artifacts for download
    bundle = artifacts_root.parent / "multi_stage_artifacts.zip"
    try:
        if bundle.exists():
            bundle.unlink()
        shutil.make_archive(str(bundle.with_suffix("")), "zip", root_dir=str(artifacts_root))
        print(f"[run_all] Wrote bundle: {bundle}")
    except Exception as exc:  # pragma: no cover
        print(f"[run_all] WARN: failed to bundle artifacts: {exc}")


if __name__ == "__main__":
    main()
