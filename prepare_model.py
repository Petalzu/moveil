"""Download the pinned NVIDIA model; never distribute its weights in Git."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

from nvidia_ner_config import MODEL, REVISION, WEIGHT_SHA256

MODEL_DIR = Path(os.environ.get('MOVEIL_MODEL_DIR', str(
    Path(__file__).resolve().parent / '.venv-ner/nvidia-gliner-pii')))


def verify_weight(directory):
    with (directory / 'pytorch_model.bin').open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != WEIGHT_SHA256:
        raise ValueError('Model weight SHA256 mismatch; do not start inference')


def prepare_ocr(backend):
    from urllib.request import urlopen
    from ocr_backend import MODELS
    model = MODELS[backend]
    root = Path(os.environ.get('MOVEIL_OCR_MODEL_DIR', str(
        Path(__file__).resolve().parent / 'models' / 'ppocr')))
    root.mkdir(parents=True, exist_ok=True)
    base = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2'
    assets = [
        (f"{base}/torch/{model['version']}/rec/{model['weights']}",
         model['weights'], model['weights_sha256']),
        (f"{base}/paddle/{model['version']}/rec/{Path(model['weights']).stem}/{model['dictionary']}",
         model['dictionary'], model['dictionary_sha256']),
    ]
    for url, name, expected in assets:
        target = root / name
        if target.exists():
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == expected:
                    continue
            raise ValueError('Existing OCR asset hash mismatch')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=name + '.', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                with urlopen(url, timeout=120) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            with temporary.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Downloaded OCR asset hash mismatch')
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print('Pinned OCR weights and dictionary SHA256 verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', choices=('ppocrv5-torch', 'ppocrv6-torch'))
    args = parser.parse_args()
    if args.ocr:
        prepare_ocr(args.ocr)
        return
    from huggingface_hub import snapshot_download
    print('Downloading pinned NVIDIA model. Review THIRD_PARTY_NOTICES.md first.')
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(MODEL_DIR),
                      token=False)
    verify_weight(MODEL_DIR)
    print('Pinned model weight SHA256 verified. Inference uses local files only.')


if __name__ == '__main__':
    main()