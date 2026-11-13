#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/preflight.py || { echo "Preflight failed. Inspect inputs before continuing."; exit 1; }
if ! command -v claude >/dev/null 2>&1; then
  echo "Claude Code CLI is not installed or is not on PATH. Install it, then run claude in this folder."
  exit 1
fi
claude 'Read CLAUDE.md and PROMPT_FOR_CLAUDE.md. Implement the full RHU Catalog Streamlit chatbot in this folder, run the acceptance tests and smoke checks, iterate until they pass, and write FINAL_IMPLEMENTATION_REPORT.md. Do not stop at planning.'
