#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=''
export PYTHONNOUSERSITE=1
for env in .venv311 .venv-ner; do
    [[ -x "$env/bin/python" ]] || { echo "Missing $env/bin/python. Run bash scripts/setup.sh first." >&2; exit 1; }
done
.venv311/bin/python -m pip check
.venv-ner/bin/python -m pip check
.venv311/bin/python launch_check.py
exec .venv311/bin/python run_demo.py "$@"