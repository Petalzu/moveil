"""Local, bounded live inference jobs over the labeled development images."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import secrets
import shutil
import threading
import time

from PIL import Image, ImageOps

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_TYPES = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}


def normalize_upload(payload, content_type):
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_UPLOAD_BYTES:
        raise ValueError('Invalid upload size')
    expected = UPLOAD_TYPES.get(content_type)
    if expected is None:
        raise ValueError('Unsupported image type')
    with Image.open(io.BytesIO(payload)) as source:
        if source.format != expected or getattr(source, 'n_frames', 1) != 1:
            raise ValueError('Invalid image format')
        width, height = source.size
        if not 0 < width <= 8192 or not 0 < height <= 8192 or width * height > 16_000_000:
            raise ValueError('Image dimensions too large')
        source.verify()
    with Image.open(io.BytesIO(payload)) as source:
        oriented = ImageOps.exif_transpose(source).convert('RGBA')
        result = Image.new('RGB', oriented.size, 'white')
        result.paste(oriented, mask=oriented.getchannel('A'))
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        return buffer.getvalue()


class LiveBatch:
    def __init__(self, root, review_factory):
        self.root = root
        self.review_factory = review_factory
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.worker = None
        self.ocr = None
        self.closed = False
        self.reviews = {}
        self.job = {'id': None, 'mode': 'samples', 'status': 'idle', 'total': 0, 'completed': 0,
                    'failed': 0, 'items': [], 'metrics': None, 'phase': '选择数量后开始运行'}

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.job)
        result['elapsed'] = round(time.monotonic() - result.pop('started'), 1) if result.get('status') == 'running' else result.get('elapsed', 0)
        return result

    def start(self, count):
        if type(count) is not int or not 1 <= count <= 10:
            raise ValueError('Choose 1 to 10 images')
        with self.lock:
            if self.closed:
                raise RuntimeError('Service is closing')
            if self.job['status'] == 'running':
                raise RuntimeError('A batch is already running')
            batch_id = 'live-' + secrets.token_hex(8)
            self.reviews = {}
            self.job = {'id': batch_id, 'mode': 'samples', 'status': 'running', 'total': count,
                        'completed': 0, 'failed': 0, 'items': [], 'metrics': None,
                        'phase': '正在加载 NVIDIA 模型', 'started': time.monotonic()}
            threading.Thread(target=self._run, args=(batch_id, count), daemon=True).start()
            return self.snapshot()

    def start_upload(self, payload, content_type='image/png'):
        with self.lock:
            if self.closed:
                raise RuntimeError('Service is closing')
            if self.job['status'] == 'running':
                raise RuntimeError('A batch is already running')
            normalized = normalize_upload(payload, content_type)
            batch_id = 'live-' + secrets.token_hex(8)
            inputs = self.root / 'inputs' / batch_id
            source_name = f'inputs/{batch_id}/1.png'
            inputs.mkdir(parents=True, exist_ok=False)
            previous_job, previous_reviews = self.job, self.reviews
            try:
                (self.root / source_name).write_bytes(normalized)
                self.reviews = {}
                self.job = {'id': batch_id, 'mode': 'upload', 'status': 'running',
                            'total': 1, 'completed': 0, 'failed': 0, 'items': [],
                            'metrics': None, 'phase': '正在加载 NVIDIA 模型',
                            'started': time.monotonic()}
                threading.Thread(target=self._run, args=(batch_id, 1, source_name), daemon=True).start()
            except Exception:
                shutil.rmtree(inputs)
                self.job, self.reviews = previous_job, previous_reviews
                raise
            return self.snapshot()

    def get_review(self, key):
        with self.lock:
            return self.reviews[key]

    def _release_models(self):
        worker, self.worker = self.worker, None
        self.ocr = None
        if worker is not None:
            worker.__exit__(None, None, None)

    @contextmanager
    def _models(self):
        from nvidia_ner_pipeline import NvidiaNerWorker, make_ocr_provider
        with self.inference_lock:
            if self.closed:
                raise RuntimeError('Service is closing')
            reused = self.worker is not None and self.worker.process.poll() is None
            started = time.monotonic()
            try:
                if not reused:
                    self._release_models()
                    self.ocr = make_ocr_provider()
                    self.worker = NvidiaNerWorker().__enter__()
                with self.lock:
                    self.job['models_reused'] = reused
                    self.job['model_load_seconds'] = round(time.monotonic() - started, 3)
                yield self.worker, self.ocr
            except Exception:
                self._release_models()
                raise

    def close(self):
        with self.lock:
            self.closed = True
        # Wait for the sole owner; never close JSONL pipes mid-request.
        with self.inference_lock:
            self._release_models()

    def _run(self, batch_id, count, uploaded_source=None):
        final_status, final_phase = 'completed', '运行完成'
        active = None
        try:
            from nvidia_ner_pipeline import run, CONFIG, CONFIG_HASH, SOURCE_FILES
            cohort = self.root / 'cohorts/nvidia-gliner-pii-dev-015'
            if uploaded_source is None:
                from evaluate import measure, summarize
                manifest = json.loads((cohort / 'manifest.json').read_text(encoding='utf-8'))[:count]
                gate = json.loads((self.root / 'quality_gate.json').read_text(encoding='utf-8'))
            else:
                manifest = [{'offset': 1}]
            if len(manifest) != count:
                raise ValueError('Missing samples')
            for entry in manifest if uploaded_source is None else []:
                if type(entry['offset']) is not int:
                    raise ValueError('Invalid sample')
                base = cohort / str(entry['offset'])
                if hashlib.sha256((base / 'inputs/sample.png').read_bytes()).hexdigest() != entry['input_sha256']:
                    raise ValueError('Source changed')
                truth = json.loads((base / 'private/truth.json').read_text(encoding='utf-8'))
                if truth['uid'] != entry['uid'] or truth['revision'] != gate['revision']:
                    raise ValueError('Truth changed')
            destination = self.root / 'runs' / batch_id
            destination.mkdir(parents=True, exist_ok=False)
            inputs = self.root / 'inputs' / batch_id
            if uploaded_source is None:
                inputs.mkdir(exist_ok=False)
            (destination / 'config.json').write_text(json.dumps({'config': CONFIG, 'config_sha256': CONFIG_HASH,
                'mode': 'upload' if uploaded_source else 'samples',
                'source': uploaded_source,
                'selected_offsets': [e['offset'] for e in manifest]}), encoding='utf-8')
            records = []
            with self._models() as (worker, ocr):
                with self.lock:
                    metadata = worker.metadata if isinstance(worker.metadata, dict) else {}
                    self.job['model'] = {key: metadata.get(key) for key in
                        ('actual_device', 'requested_device', 'gpu_name', 'versions', 'precision')}
                    self.job['ocr'] = ocr.metadata if ocr is not None else CONFIG['ocr']
                for index, entry in enumerate(manifest):
                    offset = entry['offset']
                    active = offset
                    base = cohort / str(offset)
                    source_name = uploaded_source or f'inputs/{batch_id}/{offset}.png'
                    if uploaded_source is None:
                        shutil.copyfile(base / 'inputs/sample.png', self.root / source_name)
                    run_name = f'runs/{batch_id}/{offset}'
                    out = self.root / run_name
                    start = time.monotonic()
                    with self.lock:
                        self.job['phase'] = '正在处理上传图片' if uploaded_source else f'正在处理第 {index + 1}/{count} 张 · 样例 {offset}'
                    success = run(self.root / source_name, out, worker, detector_config=CONFIG,
                        ocr_provider=ocr,
                        detector_config_hash=CONFIG_HASH, min_score=CONFIG['threshold'],
                        pipeline_name='nvidia-gliner-pii-experimental-v1',
                        source_files=SOURCE_FILES)
                    metric, summary = None, None
                    if uploaded_source is None:
                        metric = measure(base, out)
                        records.append({'offset': offset, **metric})
                        summary = summarize(records, gate)
                    key = f'{batch_id}-{offset}'
                    review = self.review_factory(self.root, run_name, source_name) if success and (metric is None or not metric.get('failed')) else None
                    item = {'id': key, 'offset': offset, 'status': 'completed' if review else 'failed',
                            'display_name': '上传图片' if uploaded_source else f'样例 {offset}',
                            'detection_count': len(review.audit['detections']) if review else None,
                            'seconds': round(time.monotonic() - start, 2),
                            'full': sum(e['fully_covered'] for e in metric['entities']) if metric else None,
                            'entities': len(metric['entities']) if metric else None}
                    with self.lock:
                        if review:
                            self.reviews[key] = review
                        self.job['items'].append(item)
                        self.job['completed'] += 1
                        self.job['failed'] += int(review is None)
                        self.job['metrics'] = {k: summary[k] for k in ('gate_passed', 'gate_version', 'n', 'entity_count', 'overall',
                            'correct_label', 'zero_missed_images', 'max_nonpii_ink_mask')} if summary else None
                    active = None
                    if summary is not None:
                        (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                    if not success:
                        raise RuntimeError('Inference failed')
                if uploaded_source is None:
                    final_phase = ('运行完成 · 质量门禁通过（两项覆盖率均 >80%）' if summary['gate_passed']
                                   else '运行完成 · 样本不足或质量门禁未通过')
                else:
                    final_phase = '运行完成 · 无标注，质量未评测'
                (destination / 'performance.json').write_text(json.dumps({
                    'models_reused': self.job['models_reused'],
                    'model_load_seconds': self.job['model_load_seconds'],
                    'worker_pid': worker.process.pid,
                    'elapsed_seconds': time.monotonic() - self.job['started'],
                    'ocr': ocr.metadata if ocr is not None else CONFIG['ocr'],
                }, indent=2), encoding='utf-8')
        except Exception as error:
            with self.lock:
                if active is not None:
                    self.job['failed'] += 1
                    self.job['completed'] += 1
                    self.job['items'].append({'id': f'{batch_id}-{active}', 'offset': active,
                        'display_name': '上传图片' if uploaded_source else f'样例 {active}',
                        'detection_count': None, 'status': 'failed', 'seconds': 0,
                        'full': None if uploaded_source else 0, 'entities': None if uploaded_source else 0})
                final_status = 'failed'
                final_phase = f'运行失败（{type(error).__name__}）· 剩余 {count-self.job["completed"]} 张未运行'
        finally:
            with self.lock:
                if self.job['id'] == batch_id:
                    self.job['elapsed'] = round(time.monotonic() - self.job.pop('started'), 1)
                    self.job['status'], self.job['phase'] = final_status, final_phase