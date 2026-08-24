#!/usr/bin/env bash
# Create the uv-managed environment needed by the paper's offline RAG flow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv sync --project "$PROJECT_DIR"

echo "Environment ready: $PROJECT_DIR/.venv"
echo "Next: copy .env.example to .env, set OPENAI_API_KEY, then use: uv run scripts/run_offline_teleqna.py --workers 2"
