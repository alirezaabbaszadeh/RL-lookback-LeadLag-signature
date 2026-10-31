"""Zero-dependency Python zip builder for RL-leader-pack.zip (avoids missing CLI zip errors)."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import zipfile


EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "raw_data", ".cache"}


def build_leader_pack(repo_root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(repo_root):
            # Prune excluded directories in-place to prevent recursive walk into .venv and .git
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for f in files:
                if f == output_zip.name or f.endswith((".pyc", ".pyo")):
                    continue
                file_path = Path(root) / f
                arcname = file_path.relative_to(repo_root)
                zf.write(file_path, arcname=str(arcname))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    local_zip = root / "RL-leader-pack.zip"
    build_leader_pack(root, local_zip)
    print(f"Built {local_zip} ({local_zip.stat().st_size / 1024:.1f} KB)")
    
    # Copy to D:\cod\IDE\rl\ if available (Windows or WSL)
    candidates = [
        Path(r"D:\cod\IDE\rl\RL-leader-pack.zip"),
        Path("/mnt/d/cod/IDE/rl/RL-leader-pack.zip"),
    ]
    for target in candidates:
        if target.parent.exists():
            shutil.copy2(local_zip, target)
            print(f"Copied to {target}")
            break
