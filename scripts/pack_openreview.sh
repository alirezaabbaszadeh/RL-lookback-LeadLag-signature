#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
OUT="artifact_anonymous.zip"
VERIFY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)
            VERIFY=1
            shift
            ;;
        -o|--output)
            OUT="$2"
            shift 2
            ;;
        *)
            OUT="$1"
            shift
            ;;
    esac
done

TMP=$(mktemp -d)
TMP_VERIFY=""
cleanup() {
    rm -rf "$TMP"
    if [[ -n "$TMP_VERIFY" ]]; then
        rm -rf "$TMP_VERIFY"
    fi
}
trap cleanup EXIT

stage_dir="$TMP/artifact"
mkdir -p "$stage_dir/src" "$stage_dir/scripts"

copy_tree() {
    local source=$1
    local dest=$2
    if command -v rsync >/dev/null 2>&1; then
        rsync -a \
            --exclude '.git' \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            --exclude '.ipynb_checkpoints' \
            --exclude 'results' \
            --exclude '.mypy_cache' \
            --exclude '.pytest_cache' \
            "$source" "$dest"
    else
        python - "$source" "$dest" <<'EOF'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", "results", ".mypy_cache", ".pytest_cache")

if src.is_dir():
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
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

SIZE_MB=$(du -m "$ROOT/$OUT" | awk '{print $1}')
if [[ ${SIZE_MB} -gt 100 ]]; then
    echo "ZIP > 100MB (${SIZE_MB} MB)" >&2
    exit 1
fi

if [[ ${VERIFY} -eq 1 ]]; then
    echo "[+] Verifying archive integrity..."
    TMP_VERIFY=$(mktemp -d)
    unzip -q "$ROOT/$OUT" -d "$TMP_VERIFY"
    pip install -e "$TMP_VERIFY/artifact" >/dev/null
    python -c "import leadlag; print('OK')" >/dev/null
fi

echo "[+] Archive ready: $ROOT/$OUT (${SIZE_MB} MB)"
