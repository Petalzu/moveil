"""Local JSONL OCR-only worker. Raw text stays in memory, never in output/logs."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from ner_config import CONFIG, CONFIG_HASH, LABEL_MAP

ROOT = Path(__file__).resolve().parent


def windows(model, text):
    words = list(model.data_processor.words_splitter(text))
    result, start = [], 0
    while start < len(words):
        end = min(start + CONFIG['window_words'], len(words))
        while end > start:
            part = text[words[start][1]:words[end-1][2]]
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
    batches = model.inference([part for _, part in parts], sorted(LABEL_MAP),
                              threshold=CONFIG['threshold'], flat_ner=True,
                              multi_label=False, batch_size=CONFIG['batch_size'])
    candidates = []
    for (offset, part), entities in zip(parts, batches, strict=True):
        for entity in entities:
            start, end = entity['start'], entity['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(part):
                raise ValueError('Invalid span')
            if part[start:end] != entity['text'] or entity['label'] not in LABEL_MAP:
                raise ValueError('Invalid evidence')
            candidates.append({'start': offset+start, 'end': offset+end,
                               'label': LABEL_MAP[entity['label']], 'score': float(entity['score'])})
    selected = []
    for item in sorted(candidates, key=lambda e: (-e['score'], e['start'], e['end'], e['label'])):
        if not any(item['start'] < e['end'] and item['end'] > e['start'] for e in selected):
            selected.append(item)
    return sorted(selected, key=lambda e: (e['start'], e['end'])), len(parts)


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    output = sys.stdout
    # Suppress third-party diagnostics, which may contain input text.
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            import torch
            from gliner import GLiNER
            torch.manual_seed(CONFIG['seed'])
            torch.set_num_threads(CONFIG['threads'])
            torch.use_deterministic_algorithms(True)
            model = GLiNER.from_pretrained(str(ROOT / 'models/gretel-gliner'),
                                          local_files_only=True, load_tokenizer=True)
            model.eval()
            hashes = {}
            for path in sorted((ROOT / 'models/gretel-gliner').iterdir()):
                if path.is_file():
                    with path.open('rb') as stream:
                        hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            ready = {'status': 'ready', 'config_sha256': CONFIG_HASH, 'model_files': hashes,
                     'versions': {p: importlib.metadata.version(p) for p in ('gliner', 'torch', 'transformers')},
                     'model_class': type(model).__name__}
            print(json.dumps(ready), file=output, flush=True)
        except Exception as exc:
            print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=output, flush=True)
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