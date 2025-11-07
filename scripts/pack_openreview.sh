#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
OUT=${1:-artifact_anonymous.zip}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

stage_dir="$TMP/artifact"
mkdir -p "$stage_dir/src" "$stage_dir/scripts"

copy_tree() {
    local source=$1
    local dest=$2
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' "$source" "$dest"
    else
        python - "$source" "$dest" <<'EOF'
import shutil
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if src.is_dir():
    shutil.copytree(src, dst, dirs_exist_ok=True)
else:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
EOF
    fi
}

echo "[+] Staging leadlag package..."
copy_tree "$ROOT/src/leadlag" "$stage_dir/src/"

for file in pyproject.toml requirements.txt requirements-kaggle.txt requirements-rl.txt requirements-dev.txt LICENSE README_ANON.md; do
    if [ -f "$ROOT/$file" ]; then
        cp "$ROOT/$file" "$stage_dir/"
    fi
done

if [ -f "$ROOT/scripts/reproduce_all.sh" ]; then
    cp "$ROOT/scripts/reproduce_all.sh" "$stage_dir/scripts/"
fi

cat > "$stage_dir/README.md" <<'EOF'
This archive contains an anonymised snapshot of the lead-lag/signature project
sufficient to reproduce the paper artifacts. Refer to README_ANON.md for Kaggle
instructions.
EOF

echo "[+] Creating archive ${OUT}..."
(cd "$stage_dir" && zip -Xr "$OUT" . >/dev/null)
mv "$stage_dir/$OUT" "$ROOT/$OUT"

echo "[+] Archive ready: $ROOT/$OUT"
