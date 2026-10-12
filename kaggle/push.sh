#!/usr/bin/env bash
# Build the Kaggle kernel (source embedded as a tarball of git-tracked files) and push it to run on a T4 GPU.
#   kaggle/push.sh [configs]   configs = model@temperature@runs,...  e.g. qwen2.5:7b@0@3,qwen2.5:7b@0.7@5
# Needs the kaggle CLI >= 1.8 (uvx kaggle) and an API token in ~/.kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIGS="${1:-qwen2.5:7b@0@3,qwen2.5:7b@0.7@5,qwen2.5:3b@0@3}"
KERNEL_ID="${KAGGLE_KERNEL_ID:-aladros/mediquest-triage-desk-evals}"
BUILD=kaggle/build
rm -rf "$BUILD" && mkdir -p "$BUILD"

git ls-files -co --exclude-standard | grep -v '^kaggle/build/' | tar -czf "$BUILD/src.tgz" -T -
python3 - "$CONFIGS" <<'PY'
import base64, sys, pathlib
configs = sys.argv[1]
src = base64.b64encode(pathlib.Path("kaggle/build/src.tgz").read_bytes()).decode()
t = pathlib.Path("kaggle/kernel_template.py").read_text()
t = t.replace("__CONFIGS__", configs).replace("__SOURCE__", src)
pathlib.Path("kaggle/build/triage_desk_evals.py").write_text(t)
PY
cat > "$BUILD/kernel-metadata.json" <<JSON
{
  "id": "$KERNEL_ID",
  "title": "mediquest-triage-desk-evals",
  "code_file": "triage_desk_evals.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
rm "$BUILD/src.tgz"
uvx kaggle kernels push -p "$BUILD"
echo "status: uvx kaggle kernels status $KERNEL_ID"
