"""Fail closed unless a real CUDA tensor operation succeeds."""
import json
import platform
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvidia_ner_config import CONFIG
from nvidia_ner_pipeline import NvidiaNerWorker


def main():
    if not CONFIG['device'].startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; set MOVEIL_DEVICE=cuda in a GPU runtime')
    x = torch.ones(32, device=CONFIG['device'])
    evidence = {'python': sys.version, 'architecture': platform.machine(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(0),
                  'capability': torch.cuda.get_device_capability(0),
                  'tensor_device': str(x.device), 'dot': (x @ x).item()}
    started = time.monotonic()
    with NvidiaNerWorker() as worker:
        evidence['worker'] = worker.metadata
        evidence['load_seconds'] = time.monotonic() - started
        started = time.monotonic()
        result = worker.predict('My first name is John and my email is john@example.com.')
        if not result['entities']:
            raise RuntimeError('Synthetic smoke inference returned no entities')
        evidence['predict_seconds'] = time.monotonic() - started
        evidence['synthetic_prediction'] = result
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()