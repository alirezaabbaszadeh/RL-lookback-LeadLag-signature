"""Zero-dependency Python zip builder for RL-leader-pack.zip (avoids missing CLI zip errors)."""

from __future__ import annotations

from pathlib import Path
import shutil
import zipfile


EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "raw_data", ".cache"}


def build_leader_pack(repo_root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(repo_root.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(repo_root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if path.name == output_zip.name or path.suffix in {".pyc", ".pyo"}:
                continue
            zf.write(path, arcname=str(rel))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    local_zip = root / "RL-leader-pack.zip"
    build_leader_pack(root, local_zip)
    print(f"Built {local_zip} ({local_zip.stat().st_size / 1024:.1f} KB)")
    win_target = Path("/mnt/d/cod/IDE/rl/RL-leader-pack.zip")
    if win_target.parent.exists():
        shutil.copy2(local_zip, win_target)
        print(f"Copied to {win_target}")
