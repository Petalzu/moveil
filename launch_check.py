"""Fail early with actionable errors; never download or change model devices."""
import os
from pathlib import Path
import subprocess
import sys

from prepare_model import MODEL_DIR, verify_weight

ROOT = Path(__file__).resolve().parent


def environment_python(root, name, platform=None):
    platform = os.name if platform is None else platform
    return root / name / ('Scripts/python.exe' if platform == 'nt' else 'bin/python')


def check():
    if sys.version_info < (3, 11):
        raise ValueError('Python 3.11 or newer is required')
    worker = Path(os.environ.get('MOVEIL_NER_PYTHON') or environment_python(ROOT, '.venv-ner'))
    if not worker.is_file():
        raise ValueError('Missing model environment; set MOVEIL_NER_PYTHON')
    subprocess.run([sys.executable, '-c', 'import PIL, numpy, requests, rapidocr_onnxruntime, onnxruntime'], check=True)
    subprocess.run([str(worker), '-c', 'import torch, gliner, transformers'], check=True)
    if not (MODEL_DIR / 'gliner_config.json').is_file():
        raise ValueError('Missing model; run prepare_model.py with MOVEIL_MODEL_DIR set')
    verify_weight(MODEL_DIR)


if __name__ == '__main__':
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Startup check failed: {exc}. Check environments and run prepare_model.py before starting.') from None