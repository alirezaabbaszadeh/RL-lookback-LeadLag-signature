from __future__ import annotations

import cProfile
import io
import json
import pstats
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator


@contextmanager
def profile_to(out_dir: Path, label: str = "run") -> Iterator[None]:
    """Profile a code block and write both .pstats and .json summaries.

    - Writes `profiles/{label}.pstats` and `profiles/{label}.json` under `out_dir`.
    - JSON contains top 30 functions by cumulative time.
    """
    out_dir = Path(out_dir)
    prof_dir = out_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    prof = cProfile.Profile()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        pstats_path = prof_dir / f"{label}.pstats"
        prof.dump_stats(str(pstats_path))

        # Build a JSON summary of top functions by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(200)

        # Extract structured data
        stats = ps.stats  # type: ignore[attr-defined]
        rows = []
        for (filename, line, funcname), stat in stats.items():
            cc, nc, tt, ct, callers = stat  # primitive fields from pstats
            rows.append(
                {
                    "file": filename,
                    "line": line,
                    "func": funcname,
                    "callcount": cc,
                    "reccallcount": nc,
                    "tottime": tt,
                    "cumtime": ct,
                }
            )

        rows.sort(key=lambda r: r["cumtime"], reverse=True)
        summary = {"top": rows[:30]}
        with open(prof_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Retention/rotation: keep only the most recent N profile files per type
        try:
            max_keep = int(os.getenv("PROFILES_MAX_KEEP", "10"))
        except Exception:
            max_keep = 10
        try:
            for suffix in (".pstats", ".json"):
                files = sorted(
                    (p for p in prof_dir.glob(f"*{suffix}") if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for old in files[max_keep:]:
                    try:
                        old.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
