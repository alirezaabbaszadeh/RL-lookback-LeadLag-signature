from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class Task:
    id: str
    title: str
    phase: str
    owner: Optional[str]
    status: str
    path: Path


def _parse_tasks_fallback(text: str) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    in_tasks = False
    current: Optional[Dict[str, str]] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.strip() == "tasks:":
            in_tasks = True
            continue
        if not in_tasks:
            continue
        # Start of a new task only for top-level list items (indent 2 spaces)
        if line.startswith("  - "):
            if current:
                tasks.append(current)
            current = {}
            kv = line.lstrip()[2:]
            if ":" in kv:
                k, v = kv.split(":", 1)
                current[k.strip()] = v.strip()
            continue
        # Ignore nested list items (e.g., outputs: - path)
        if line.strip().startswith("- "):
            continue
        if current is not None and ":" in line:
            k, v = line.split(":", 1)
            current[k.strip()] = v.strip()
    if current:
        tasks.append(current)
    return tasks


def load_tasks(phase_dir: Path) -> List[Task]:
    tasks_path = phase_dir / "tasks.yaml"
    if not tasks_path.exists():
        return []
    data: Dict[str, object] = {}
    items: List[Dict[str, object]] = []
    text = tasks_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        items = list(data.get("tasks", []))  # type: ignore[arg-type]
    else:
        items = _parse_tasks_fallback(text)  # minimal fields only
    tasks = []
    for item in items:
        tasks.append(
            Task(
                id=str(item.get("id")),  # type: ignore[arg-type]
                title=str(item.get("title")),  # type: ignore[arg-type]
                phase=phase_dir.name,
                owner=item.get("owner") if isinstance(item, dict) else None,  # type: ignore[union-attr]
                status=str(item.get("status")),  # type: ignore[arg-type]
                path=tasks_path,
            )
        )
    return tasks


def main() -> int:
    root = Path("docs/audit")
    if not root.exists():
        print("No audit directory found.")
        return 0
    phases = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("phase-"))
    all_tasks: List[Task] = []
    for phase in phases:
        all_tasks.extend(load_tasks(phase))
    if not all_tasks:
        print("No tasks found.")
        return 0
    # Print summary
    print("Phase\tStatus\tID\tTitle")
    for t in all_tasks:
        print(f"{t.phase}\t{t.status}\t{t.id}\t{t.title}")
    # Write a machine-readable index
    index = [
        {
            "phase": t.phase,
            "id": t.id,
            "title": t.title,
            "owner": t.owner,
            "status": t.status,
            "tasks_file": str(t.path),
        }
        for t in all_tasks
    ]
    (root / "tasks_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
