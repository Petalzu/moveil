#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export MOVEIL_DEVICE="${MOVEIL_DEVICE:-cuda}"
export MOVEIL_NER_PYTHON="${MOVEIL_NER_PYTHON:-$(command -v python3)}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case "${1:-serve}" in
  setup) exec "$MOVEIL_NER_PYTHON" -m pip install -r requirements-spark.txt ;;
  prepare) exec "$MOVEIL_NER_PYTHON" prepare_model.py ;;
  check) "$MOVEIL_NER_PYTHON" launch_check.py; exec "$MOVEIL_NER_PYTHON" tools/check_gpu.py ;;
  serve) shift "$(( $# > 0 ? 1 : 0 ))"; "$MOVEIL_NER_PYTHON" launch_check.py; exec "$MOVEIL_NER_PYTHON" run_demo.py "$@" ;;
  *) echo 'Usage: bash start.sh [setup|prepare|check|serve [--port 18196] [--run runs/name]]' >&2; exit 2 ;;
esac