"""Image-only pipeline using NVIDIA GLiNER-PII and the existing OCR geometry."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from nvidia_ner_config import CONFIG, CONFIG_HASH
from ner_pipeline import run

ROOT = Path(__file__).resolve().parent


def validate_ready(metadata):
    if metadata.get('status') != 'ready' or metadata.get('config_sha256') != CONFIG_HASH:
        raise ValueError('Worker startup failed: ' + str(metadata.get('error_type', 'invalid handshake')))
    expected = 'cuda:0' if CONFIG['device'] == 'cuda' else CONFIG['device']
    if metadata.get('actual_device') != expected:
        raise ValueError('Worker device mismatch; refusing fallback')


class NvidiaNerWorker:
    def __enter__(self):
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH='',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        self.process = subprocess.Popen(
            [os.environ.get('MOVEIL_NER_PYTHON') or str(ROOT / '.venv-ner' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')), str(ROOT / 'nvidia_ner_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', env=env, cwd=ROOT)
        self.responses = queue.Queue()

        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put('')

        threading.Thread(target=reader, daemon=True).start()
        self.sequence = 0
        try:
            self.metadata = self.receive()
            validate_ready(self.metadata)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def receive(self):
        try:
            raw = self.responses.get(timeout=900)
        except queue.Empty:
            raise TimeoutError('Worker timeout') from None
        return json.loads(raw)

    def predict(self, text):
        self.sequence += 1
        self.process.stdin.write(json.dumps({'id': self.sequence, 'text': text}) + '\n')
        self.process.stdin.flush()
        result = self.receive()
        if (set(result) != {'id', 'entities', 'windows', 'config_sha256'} or
                result['id'] != self.sequence or result['config_sha256'] != CONFIG_HASH):
            raise ValueError('Invalid worker response')
        return result

    def __exit__(self, *args):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--outputs', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if len(args.images) != len(args.outputs):
        parser.error('Images and outputs must match')
    with NvidiaNerWorker() as worker:
        outcomes = [run(image, out, worker, detector_config=CONFIG,
                        detector_config_hash=CONFIG_HASH, min_score=CONFIG['threshold'],
                        pipeline_name='nvidia-gliner-pii-experimental-v1',
                        source_files=('agent.py', 'ner_pipeline.py', 'ner_worker.py',
                                      'ner_config.py', 'nvidia_ner_worker.py',
                                      'nvidia_ner_config.py', 'nvidia_ner_pipeline.py'))
                    for image, out in zip(args.images, args.outputs, strict=True)]
    raise SystemExit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()