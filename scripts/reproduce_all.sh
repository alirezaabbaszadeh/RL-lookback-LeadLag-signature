#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/reproduce_all.sh [--dry-run] [--skip-stats] [--profiles "training=paper data=kaggle"]
# Environment overrides:
#   ENTRY           Pipeline entry point command (default: python -m leadlag.pipelines.run_full_suite)
#   RES             Directory for run outputs (default: /kaggle/working/results)
#   OUT             Directory for paper artefacts (default: /kaggle/working/paper_outputs)
#   CFG_OVERRIDES   Base Hydra-style overrides passed to the entrypoint

ENTRY=${ENTRY:-"python -m leadlag.pipelines.run_full_suite"}
RES=${RES:-"/kaggle/working/results"}
OUT=${OUT:-"/kaggle/working/paper_outputs"}
CFG_OVERRIDES=${CFG_OVERRIDES:-"training=paper hardware=cpu split=walk_forward_purged"}

export RES OUT

DRY_RUN=0
SKIP_STATS=0
PROFILE_OVERRIDES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-stats)
            SKIP_STATS=1
            shift
            ;;
        --profiles)
            PROFILE_OVERRIDES="${2:-}"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: reproduce_all.sh [options]
  --dry-run       Print diagnostics only
  --skip-stats    Skip the statistics aggregation step
  --profiles ARG  Additional Hydra overrides (e.g. "training=paper data=demo")
USAGE
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

EFFECTIVE_OVERRIDES="${CFG_OVERRIDES}"
if [[ -n "${PROFILE_OVERRIDES}" ]]; then
    EFFECTIVE_OVERRIDES="${EFFECTIVE_OVERRIDES} ${PROFILE_OVERRIDES}"
fi
EFFECTIVE_OVERRIDES=$(echo "${EFFECTIVE_OVERRIDES}" | xargs)

echo "[diagnostics] python version"
python -V || true
echo "[diagnostics] pip snapshot"
pip list | head -n 20 || true
echo "[diagnostics] nvidia-smi"
nvidia-smi || echo "nvidia-smi unavailable"
echo "[diagnostics] torch"
python - <<'PY' || true
try:
    import torch

    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
except Exception as exc:  # pragma: no cover - diagnostic helper
    print(f"torch diagnostics unavailable: {exc}")
PY

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] Skipping execution as requested."
    exit 0
fi

mkdir -p "${RES}" "${OUT}"

read -r -a entry_args <<< "${ENTRY}"
if [[ -n "${EFFECTIVE_OVERRIDES}" ]]; then
    read -r -a override_args <<< "${EFFECTIVE_OVERRIDES}"
    stage_cmd=("${entry_args[@]}" "${override_args[@]}")
else
    stage_cmd=("${entry_args[@]}")
fi

echo "[1/3] Running leadlag full suite (${EFFECTIVE_OVERRIDES})..."
stage_start=$(date +%s)
"${stage_cmd[@]}"
stage_end=$(date +%s)
echo "[1/3] Completed in $((stage_end - stage_start))s"

metrics_count=$(find "${RES}" -maxdepth 2 -name "metrics.csv" | wc -l | tr -d ' ')
if [[ ${metrics_count} -eq 0 ]]; then
    echo "No metrics.csv files discovered under ${RES}" >&2
    exit 1
fi

if [[ ${SKIP_STATS} -eq 0 ]]; then
    echo "[2/3] Computing paper-grade statistics..."
    stage_start=$(date +%s)
    python -m leadlag.eval.stats_cli --results "${RES}" --out "${OUT}" --alpha 0.05
    stage_end=$(date +%s)
    echo "[2/3] Completed in $((stage_end - stage_start))s"
else
    echo "[2/3] Skipped statistics step"
fi

echo "[3/3] Aggregating metrics to ${OUT}/all_metrics_raw.csv..."
stage_start=$(date +%s)
python - <<'EOF'
import os
from pathlib import Path

import pandas as pd

results_root = Path(os.environ["RES"]).resolve()
out_root = Path(os.environ["OUT"]).resolve()
out_root.mkdir(parents=True, exist_ok=True)
records = []
for metrics_path in results_root.glob("*/metrics.csv"):
    try:
        frame = pd.read_csv(metrics_path)
    except Exception as exc:  # pragma: no cover - defensive read
        print(f"Failed to load {metrics_path}: {exc}")
        continue
    frame["run_dir"] = metrics_path.parent.name
    records.append(frame)

if not records:
    raise SystemExit("No metrics.csv files discovered during aggregation")

combined = pd.concat(records, ignore_index=True)
combined.to_csv(out_root / "all_metrics_raw.csv", index=False)
print(f"Aggregated {len(records)} metrics files.")
EOF
stage_end=$(date +%s)
echo "[3/3] Completed in $((stage_end - stage_start))s"

echo "Done. Paper artifacts are available under ${OUT}."
