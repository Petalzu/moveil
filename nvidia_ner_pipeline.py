"""Image-only pipeline using NVIDIA GLiNER-PII and the existing OCR geometry."""
from pathlib import Path

from nvidia_ner_config import CONFIG, CONFIG_HASH
from ner_pipeline import NerWorker, WorkerError, run
from ocr_backend import make_ocr_provider

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('agent.py', 'hybrid_geometry.py', 'ner_pipeline.py', 'ner_worker.py', 'ner_config.py',
                'nvidia_ner_worker.py', 'nvidia_ner_config.py', 'nvidia_ner_pipeline.py', 'ocr_backend.py')


def validate_ready(metadata):
    if metadata.get('status') != 'ready' or metadata.get('config_sha256') != CONFIG_HASH:
        raise ValueError('Worker startup failed: ' + str(metadata.get('error_type', 'invalid handshake')))
    expected = 'cuda:0' if CONFIG['device'] == 'cuda' else CONFIG['device']
    if metadata.get('actual_device') != expected:
        raise ValueError('Worker device mismatch; refusing fallback')


class NvidiaNerWorker(NerWorker):
    script = 'nvidia_ner_worker.py'
    config_hash = CONFIG_HASH
    timeout = 900
    validate_metadata = staticmethod(validate_ready)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--outputs', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if len(args.images) != len(args.outputs):
        parser.error('Images and outputs must match')
    ocr = make_ocr_provider()
    with NvidiaNerWorker() as worker:
        outcomes = [run(image, out, worker, ocr_provider=ocr, detector_config=CONFIG,
                        detector_config_hash=CONFIG_HASH, min_score=CONFIG['threshold'],
                        pipeline_name='nvidia-gliner-pii-experimental-v1',
                        source_files=SOURCE_FILES)
                    for image, out in zip(args.images, args.outputs, strict=True)]
    raise SystemExit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()