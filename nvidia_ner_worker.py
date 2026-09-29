"""Local text-only worker for the pinned NVIDIA GLiNER-PII experiment."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from nvidia_ner_config import CONFIG, CONFIG_HASH, LABEL_GROUPS, WEIGHT_SHA256
from ner_config import LABEL_MAP

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get('MOVEIL_MODEL_DIR', str(ROOT / '.venv-ner/nvidia-gliner-pii')))


def windows(model, text):
    words = list(model.data_processor.words_splitter(text))
    result, start = [], 0
    while start < len(words):
        end = min(start + CONFIG['window_words'], len(words))
        while end > start:
            part = text[words[start][1]:words[end - 1][2]]
            count = len(model.data_processor.transformer_tokenizer(
                part, add_special_tokens=True, truncation=False)['input_ids'])
            if count <= CONFIG['max_subtokens']:
                break
            end -= 1
        if end <= start:
            raise ValueError('Oversized token')
        result.append((words[start][1], part))
        if end == len(words):
            break
        start = max(start + 1, end - CONFIG['overlap_words'])
    return result


def predict(model, text):
    parts = windows(model, text)
    candidates = []
    for offset, part in parts:
        for labels in LABEL_GROUPS:
            batches = model.inference([part], labels, threshold=CONFIG['threshold'],
                                      flat_ner=True, multi_label=False,
                                      batch_size=CONFIG['batch_size'])
            if len(batches) != 1:
                raise ValueError('Invalid model batch')
            for entity in batches[0]:
                start, end = entity['start'], entity['end']
                label, score = entity['label'], float(entity['score'])
                if (type(start) is not int or type(end) is not int or
                        not 0 <= start < end <= len(part)):
                    raise ValueError('Invalid span')
                if part[start:end] != entity['text'] or label not in LABEL_MAP:
                    raise ValueError('Invalid evidence')
                candidates.append({'start': offset + start, 'end': offset + end,
                                   'label': LABEL_MAP[label], 'score': score})
    selected = []
    for item in sorted(candidates, key=lambda e: (-e['score'], e['start'], e['end'], e['label'])):
        if not any(item['start'] < e['end'] and item['end'] > e['start'] for e in selected):
            selected.append(item)
    return sorted(selected, key=lambda e: (e['start'], e['end'])), len(parts)


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    output = sys.stdout
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            import torch
            from gliner import GLiNER
            weight = MODEL_DIR / 'pytorch_model.bin'
            if file_sha256(weight) != WEIGHT_SHA256:
                raise ValueError('Model weight hash mismatch')
            torch.manual_seed(CONFIG['seed'])
            torch.set_num_threads(CONFIG['threads'])
            torch.use_deterministic_algorithms(True)
            if CONFIG['device'].startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
            model = GLiNER.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                           load_tokenizer=True)
            model.to(CONFIG['device'])
            model.eval()
            actual_device = str(next(model.parameters()).device)
            expected_device = 'cuda:0' if CONFIG['device'] == 'cuda' else CONFIG['device']
            if actual_device != expected_device:
                raise RuntimeError('Model device does not match requested device')
            if model.config.max_types < CONFIG['label_group_size']:
                raise ValueError('Label group exceeds model max_types')
            hashes = {str(path.relative_to(MODEL_DIR)): file_sha256(path)
                      for path in sorted(MODEL_DIR.rglob('*'))
                      if path.is_file() and '.cache' not in path.parts}
            ready = {'status': 'ready', 'config_sha256': CONFIG_HASH,
                     'requested_device': CONFIG['device'],
                     'actual_device': actual_device,
                     'cuda_version': torch.version.cuda,
                     'gpu_name': torch.cuda.get_device_name(0) if CONFIG['device'].startswith('cuda') else None,
                     'model_files': hashes,
                     'versions': {p: importlib.metadata.version(p)
                                  for p in ('gliner', 'torch', 'transformers')},
                     'model_class': type(model).__name__,
                     'model_max_types': model.config.max_types}
            print(json.dumps(ready), file=output, flush=True)
        except Exception as exc:
            print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}),
                  file=output, flush=True)
            return 1
        for raw in sys.stdin:
            try:
                request = json.loads(raw)
                if set(request) != {'id', 'text'} or type(request['id']) is not int:
                    raise ValueError('Invalid request')
                text = request['text']
                if not isinstance(text, str) or not text.strip() or len(text) > 6000:
                    raise ValueError('Invalid document')
                entities, count = predict(model, text)
                response = {'id': request['id'], 'entities': entities, 'windows': count,
                            'config_sha256': CONFIG_HASH}
            except Exception as exc:
                response = {'status': 'failed', 'error_type': type(exc).__name__}
            print(json.dumps(response, allow_nan=False), file=output, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())