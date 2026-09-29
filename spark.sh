#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${MOVEIL_IMAGE:?Set MOVEIL_IMAGE to a trusted ARM64 CUDA PyTorch image (prefer a registry digest)}"
: "${MOVEIL_HOST_MODEL_DIR:?Set MOVEIL_HOST_MODEL_DIR to the complete model directory on the host}"
[[ -f "$MOVEIL_HOST_MODEL_DIR/pytorch_model.bin" ]] || { echo 'Missing model weight' >&2; exit 1; }
command=serve
case "${1:-}" in
  check|serve) command="$1"; shift ;;
esac
exec docker run --rm --gpus all --network host \
  -v "$PWD:/app" -v "$(realpath "$MOVEIL_HOST_MODEL_DIR"):/models/nvidia-gliner-pii:ro" \
  -w /app -e MOVEIL_DEVICE=cuda -e MOVEIL_MODEL_DIR=/models/nvidia-gliner-pii \
  -e MOVEIL_LOAD_DOTENV=0 -e PYTHONPATH= -e PYTHONNOUSERSITE=1 \
  -e MOVEIL_NER_PYTHON="${MOVEIL_NER_PYTHON:-/usr/bin/python3}" \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --entrypoint bash "$MOVEIL_IMAGE" start.sh "$command" "$@"