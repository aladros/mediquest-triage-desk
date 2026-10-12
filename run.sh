#!/usr/bin/env bash
# One command: checks prerequisites, pulls the open-weights model, runs the triage-desk demo.
#   ./run.sh                 # interactive demo — you are the nurse at the gate
#   ./run.sh recheck         # deterioration re-check for a waiting patient (run after the demo)
#   ./run.sh eval --runs 3   # the eval set
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${TRIAGE_MODEL:-qwen2.5:7b}"

command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }

if ! curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
  echo "Starting ollama serve…"
  (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 20); do curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null && break; sleep 1; done
fi
ollama list | awk '{print $1}' | grep -qx "$MODEL" || ollama pull "$MODEL"

export TRIAGE_MODEL="$MODEL"
exec uv run triage-desk "${@:-demo}"
