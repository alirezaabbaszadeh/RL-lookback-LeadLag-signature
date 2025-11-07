#!/usr/bin/env bash
set -euo pipefail

ENTRY=${ENTRY:-"python -m leadlag.pipelines.run_full_suite"}
RES=${RES:-"/kaggle/working/results"}
OUT=${OUT:-"/kaggle/working/paper_outputs"}
CFG_OVERRIDES=${CFG_OVERRIDES:-"training=paper hardware=gpu split=walk_forward_purged"}

mkdir -p "${RES}" "${OUT}"

echo "[1/3] Running leadlag full suite (${CFG_OVERRIDES})..."
${ENTRY} ${CFG_OVERRIDES}

echo "[2/3] Computing paper-grade statistics..."
python -m leadlag.eval.stats_cli --results "${RES}" --out "${OUT}" --alpha 0.05

echo "[3/3] Aggregating metrics to ${OUT}/all_metrics_raw.csv..."
python - <<'EOF'
import os
from pathlib import Path
import pandas as pd

results_root = Path(os.environ.get("RES", "/kaggle/working/results")).resolve()
out_root = Path(os.environ.get("OUT", "/kaggle/working/paper_outputs")).resolve()
out_root.mkdir(parents=True, exist_ok=True)
records = []
for metrics_path in results_root.glob("*/metrics.csv"):
    try:
        frame = pd.read_csv(metrics_path)
    except Exception:
        continue
    frame["run_dir"] = metrics_path.parent.name
    records.append(frame)
if records:
    combined = pd.concat(records, ignore_index=True)
    combined.to_csv(out_root / "all_metrics_raw.csv", index=False)
    print(f"Aggregated {len(records)} metrics files.")
else:
    print("No metrics.csv files discovered; skipping aggregation.")
EOF

echo "Done. Paper artifacts are available under ${OUT}."
