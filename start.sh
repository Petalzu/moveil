#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${MOVEIL_LOAD_DOTENV:-1}" == 1 && -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH=''
export PYTHONNOUSERSITE=1
export MOVEIL_DEVICE="${MOVEIL_DEVICE:-cuda}"
export MOVEIL_NER_PYTHON="${MOVEIL_NER_PYTHON:-$(command -v python3)}"
export MOVEIL_APP_PYTHON="${MOVEIL_APP_PYTHON:-$MOVEIL_NER_PYTHON}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case "${1:-serve}" in
  setup)
    constraint="$(mktemp)"
    trap 'rm -f "$constraint"' EXIT
    "$MOVEIL_NER_PYTHON" -c 'import torch; assert torch.version.cuda, "CUDA torch must be installed first"; print("torch==" + torch.__version__)' > "$constraint"
    "$MOVEIL_NER_PYTHON" -m pip install -c "$constraint" -r requirements-spark.txt
    if [[ "$MOVEIL_APP_PYTHON" != "$MOVEIL_NER_PYTHON" ]]; then
      "$MOVEIL_APP_PYTHON" -m pip install -r requirements.txt
    fi
    if [[ "${MOVEIL_OCR_BACKEND:-rapid-cpu}" != rapid-cpu ]]; then
      "$MOVEIL_APP_PYTHON" -c 'import torch; assert torch.version.cuda, "GPU OCR requires CUDA torch in the application environment"'
      "$MOVEIL_APP_PYTHON" -m pip install -c "$constraint" -r requirements-ocr-gpu.txt
    fi ;;
  prepare) shift; exec "$MOVEIL_NER_PYTHON" prepare_model.py "$@" ;;
  check) "$MOVEIL_APP_PYTHON" launch_check.py; "$MOVEIL_NER_PYTHON" tools/check_gpu.py; exec "$MOVEIL_APP_PYTHON" -c 'from ocr_backend import make_ocr_provider; ocr = make_ocr_provider(); print(ocr.metadata if ocr else "CPU OCR configured")' ;;
  serve) shift "$(( $# > 0 ? 1 : 0 ))"; "$MOVEIL_APP_PYTHON" launch_check.py; exec "$MOVEIL_APP_PYTHON" run_demo.py "$@" ;;
  *) echo 'Usage: bash start.sh [setup|prepare [--ocr ppocrv6-torch]|check|serve [--port 18196] [--run runs/name]]' >&2; exit 2 ;;
esac