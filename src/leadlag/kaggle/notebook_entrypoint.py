"""Notebook-friendly Kaggle entrypoint.

This helper makes it easy to kick off the grand Kaggle run from a single cell:

```
!python kaggle/notebook_entrypoint.py
```

It will copy the repository contents from an attached dataset (if provided),
locate the bundled ``kaggle/run_all.py`` script, and delegate to it with
defaults that match the documented notebook flow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_PATH = Path("/kaggle/input/leadlag-signature")
DEFAULT_WORK_DIR = Path("/kaggle/working")


def ensure_repo_root(work_dir: Path, dataset_path: Path | None = None) -> Path:
    """Return a directory that contains ``kaggle/run_all.py``.

    If the working directory is already the repository root, it is returned
    unchanged. Otherwise, the helper attempts to copy files from a Kaggle
    dataset and search for the script.
    """

    if (work_dir / "kaggle" / "run_all.py").exists():
        return work_dir

    if dataset_path and dataset_path.exists():
        print(f"[notebook-entrypoint] Copying repo from {dataset_path} -> {work_dir}")
        for item in dataset_path.iterdir():
            destination = work_dir / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

    try:
        run_all_path = next(work_dir.rglob("kaggle/run_all.py"))
    except StopIteration:
        raise SystemExit(
            "Could not locate kaggle/run_all.py. Attach the repository dataset "
            "or set --dataset-path to the extracted source."
        )

    resolved_root = run_all_path.parent.parent
    print(f"[notebook-entrypoint] Using repository root: {resolved_root}")
    return resolved_root


def launch(run_all_path: Path, *, artifacts_root: Path | None, prefetch: bool) -> None:
    """Invoke ``kaggle/run_all.py`` with optional arguments."""

    cmd = [
        sys.executable,
        str(run_all_path),
    ]
    if artifacts_root is not None:
        cmd.extend(["--artifacts-root", str(artifacts_root)])
    if not prefetch:
        cmd.append("--no-prefetch")

    print("[notebook-entrypoint]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-cell Kaggle runner")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the attached Kaggle dataset that holds the repository files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Working directory where artifacts and caches will be written.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional override for the orchestrator artifact root.",
    )
    parser.add_argument(
        "--prefetch/--no-prefetch",
        default=True,
        help="Enable or disable wheel prefetch before running all stages.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    repo_root = ensure_repo_root(work_dir, dataset_path=args.dataset_path)
    launch(repo_root / "kaggle" / "run_all.py", artifacts_root=args.artifacts_root, prefetch=args.prefetch)


if __name__ == "__main__":
    main()
