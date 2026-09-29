"""Image-only NER pipeline; no evaluator, annotation or dataset access."""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from agent import LABELS, projection_ocr, redact, validate
from nvidia_ner_config import CONFIG, CONFIG_HASH

ROOT = Path(__file__).resolve().parent


class WorkerError(RuntimeError):
    """A safe worker error code; never includes document text."""
    def __init__(self, code):
        self.code = code if code in {'ValueError', 'RuntimeError', 'OutOfMemoryError',
                                     'OSError', 'TypeError'} else 'InferenceError'
        super().__init__('Worker failed: ' + self.code)


class NerWorker:
    script = 'nvidia_ner_worker.py'
    config_hash = CONFIG_HASH
    timeout = 900

    def validate_metadata(self, metadata):
        if metadata.get('status') != 'ready' or metadata.get('config_sha256') != self.config_hash:
            raise ValueError('Worker startup failed')

    def __enter__(self):
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        python = os.environ.get('MOVEIL_NER_PYTHON') or str(ROOT / '.venv-ner' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
        self.process = subprocess.Popen([python, str(ROOT / self.script)],
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
            self.validate_metadata(self.metadata)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def receive(self):
        try:
            raw = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError('Worker timeout') from None
        if not raw:
            raise RuntimeError('Worker exited before responding')
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError('Invalid worker response')
        return result

    def predict(self, text):
        if self.process.poll() is not None:
            raise RuntimeError('Worker is no longer running')
        try:
            self.sequence += 1
            self.process.stdin.write(json.dumps({'id': self.sequence, 'text': text}) + '\n')
            self.process.stdin.flush()
            result = self.receive()
            if result.get('id') != self.sequence or result.get('config_sha256') != self.config_hash:
                raise ValueError('Invalid worker response')
            if result.get('status') == 'failed':
                raise WorkerError(result.get('error_type'))
            if set(result) != {'id', 'entities', 'windows', 'config_sha256'}:
                raise ValueError('Invalid worker response')
            return result
        except Exception:
            # Never reuse an EOF/timeout/error stream for another document.
            self.__exit__(None, None, None)
            raise

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


def map_spans(entities, lines, min_score=None):
    document = ' '.join(line['text'] for line in lines)
    if min_score is None:
        min_score = CONFIG['threshold']
    if not isinstance(entities, list) or len(entities) > 128:
        raise ValueError('Invalid entities')
    spans = []
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {'start', 'end', 'label', 'score'}:
            raise ValueError('Invalid entity fields')
        start, end, score = entity['start'], entity['end'], entity['score']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document):
            raise ValueError('Out of bounds')
        if entity['label'] not in LABELS or type(score) not in (int, float) or not math.isfinite(score) or not min_score <= score <= 1:
            raise ValueError('Invalid label or score')
        if not document[start:end].strip():
            raise ValueError('Empty evidence')
        cursor = 0
        for i, line in enumerate(lines):
            a, b = max(start, cursor), min(end, cursor + len(line['text']))
            if a < b:
                spans.append({'label': entity['label'], 'line': i, 'start': a-cursor, 'end': b-cursor})
            cursor += len(line['text']) + 1
    validate({'action': 'redact', 'entities': spans}, 1, lines)
    return spans


def run(image, out, worker, ocr_provider=None, *, detector_config=CONFIG,
        detector_config_hash=CONFIG_HASH, min_score=None, pipeline_name='nvidia-gliner-pii-experimental-v1',
        source_files=None):
    out.mkdir(parents=True, exist_ok=False)
    output, started = out / 'redacted.png', time.monotonic()
    if min_score is None:
        min_score = detector_config['threshold']
    report = {'status': 'failed', 'stage': 'input', 'pipeline': pipeline_name, 'config': detector_config,
              'config_sha256': detector_config_hash, 'worker': worker.metadata,
              'native_tool_calls': False, 'structured_action_loop': False, 'openclaw_host_verified': False}
    try:
        report['sources'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in (source_files or ('agent.py', 'hybrid_geometry.py', 'ner_pipeline.py',
                                 'ner_worker.py', 'ner_config.py', 'nvidia_ner_worker.py', 'nvidia_ner_config.py'))}
        report['input_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
        report['stage'] = 'ocr'
        ocr_started = time.monotonic()
        lines = projection_ocr(image) if ocr_provider is None else ocr_provider(image)
        report['ocr_seconds'] = time.monotonic() - ocr_started
        if ocr_provider is not None:
            report['ocr_backend'] = ocr_provider.metadata
        document = ' '.join(line['text'] for line in lines)
        if len(document) > 6000:
            raise ValueError('Document exceeds validated scope')
        report.update(ocr_units=len(lines), ocr_characters=len(document), stage='ner')
        result = worker.predict(document)
        report['windows'] = result['windows']
        report['stage'] = 'geometry'
        spans = map_spans(result['entities'], lines, min_score=min_score)
        report['ner_spans'] = result['entities']
        from hybrid_geometry import require_geometry
        report['geometry_unmapped_characters'] = sum(len(line.get('geometry_unmapped', [])) for line in lines)
        require_geometry(lines, spans)
        report['detections'] = redact(image, output, spans, lines)
        report['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        report['status'] = 'completed_unassessed' if spans else 'no_detections_unverified'
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        if isinstance(exc, WorkerError):
            report['error_code'] = exc.code
        output.unlink(missing_ok=True)
    report['elapsed_seconds'] = time.monotonic() - started
    (out / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'run': out.name, 'status': report['status'], 'stage': report['stage']}), flush=True)
    return report['status'] != 'failed'