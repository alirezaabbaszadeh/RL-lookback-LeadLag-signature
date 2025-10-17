#!/usr/bin/env bash
set -euo pipefail

SCENARIO=${1:-fixed_30}
OUTPUT_ROOT=${2:-results}

python hydra_main.py --scenario "$SCENARIO" --output_root "$OUTPUT_ROOT"

