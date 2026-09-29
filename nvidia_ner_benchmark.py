"""Fixed-cohort development benchmark for the pinned NVIDIA PII detector."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark import summarize
from evaluate import evaluate
from nvidia_ner_config import CONFIG, CONFIG_HASH

ROOT = Path(__file__).resolve().parent


def measure(base, out):
    audit = out / 'audit.json'
    if audit.exists() and json.loads(audit.read_text())['status'] != 'failed':
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate(base, out)
    truth = json.loads((base / 'private/truth.json').read_text())
    return {'entities': [{'id': e['id'], 'label': e['label'], 'fully_covered': False,
                          'correct_label_fully_covered': False} for e in truth['entities']],
            'nonsensitive_ink_redacted_fraction': 0, 'failed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--from-cohort', type=Path, required=True)
    parser.add_argument('--single-out', type=Path, required=True)
    args = parser.parse_args()
    gate_bytes = (ROOT / 'quality_gate.json').read_bytes()
    gate = json.loads(gate_bytes)
    if args.single_out.exists():
        raise ValueError('Refusing reused single output')
    args.destination.mkdir(parents=True, exist_ok=False)
    for offset in gate['holdout_offsets']:
        for folder in ('inputs', 'private'):
            shutil.copytree(args.from_cohort / str(offset) / folder,
                            args.destination / str(offset) / folder)
    shutil.copyfile(args.from_cohort / 'manifest.json', args.destination / 'manifest.json')
    source_names = ('agent.py', 'ner_pipeline.py', 'ner_worker.py', 'ner_config.py',
                    'nvidia_ner_worker.py', 'nvidia_ner_config.py',
                    'nvidia_ner_pipeline.py', 'nvidia_ner_benchmark.py',
                    'benchmark.py', 'evaluate.py')
    frozen = {'config': CONFIG, 'config_sha256': CONFIG_HASH,
              'scope': 'development only; model trained on Nemotron-PII',
              'gate_sha256': hashlib.sha256(gate_bytes).hexdigest(),
              'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in source_names}}
    (args.destination / 'inference_config.json').write_text(
        json.dumps(frozen, indent=2), encoding='utf-8')
    bases = [ROOT] + [args.destination / str(offset) for offset in gate['holdout_offsets']]
    outputs = [args.single_out] + [base / 'run' for base in bases[1:]]
    result = subprocess.run(
        [sys.executable, str(ROOT / 'nvidia_ner_pipeline.py'), '--images',
         *[str(base / 'inputs/sample.png') for base in bases], '--outputs',
         *map(str, outputs)], check=False)
    single = measure(ROOT, args.single_out)
    metrics = [{'offset': offset, **measure(base, out)}
               for offset, base, out in zip(gate['holdout_offsets'], bases[1:],
                                             outputs[1:], strict=True)]
    summary = summarize(metrics, gate)
    single_passed = (
        not single.get('failed', False) and
        len(single['entities']) >= gate['dev_required_full_entities'] and
        all(e['fully_covered'] and e['correct_label_fully_covered']
            for e in single['entities']) and
        single['nonsensitive_ink_redacted_fraction'] <=
        gate['per_image_nonpii_ink_false_mask_max'])
    summary.update(config_sha256=CONFIG_HASH,
                   gate_sha256=frozen['gate_sha256'], single=single,
                   single_gate_passed=single_passed,
                   eligible_for_new_holdout=summary['gate_passed'] and single_passed,
                   subprocess_returncode=result.returncode, new_holdout=False,
                   failed_records=sum(m.get('failed', False) for m in metrics))
    if hashlib.sha256((ROOT / 'quality_gate.json').read_bytes()).hexdigest() != frozen['gate_sha256']:
        raise ValueError('Gate changed')
    (args.destination / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {'records', 'per_label', 'single'}}, indent=2))


if __name__ == '__main__':
    main()