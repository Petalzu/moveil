"""Download the pinned NVIDIA model; never distribute its weights in Git."""
import hashlib
import os
from pathlib import Path

from nvidia_ner_config import MODEL, REVISION, WEIGHT_SHA256

MODEL_DIR = Path(os.environ.get('MOVEIL_MODEL_DIR', str(
    Path(__file__).resolve().parent / '.venv-ner/nvidia-gliner-pii')))


def verify_weight(directory):
    with (directory / 'pytorch_model.bin').open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != WEIGHT_SHA256:
        raise ValueError('Model weight SHA256 mismatch; do not start inference')


def main():
    from huggingface_hub import snapshot_download
    print('Downloading pinned NVIDIA model. Review THIRD_PARTY_NOTICES.md first.')
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(MODEL_DIR),
                      token=False)
    verify_weight(MODEL_DIR)
    print('Pinned model weight SHA256 verified. Inference uses local files only.')


if __name__ == '__main__':
    main()