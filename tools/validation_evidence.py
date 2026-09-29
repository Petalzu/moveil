"""Export compact GPU benchmark evidence, without OCR text or image contents."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    parser.add_argument('single', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.cohort / 'summary.json').read_text())
    paths = [args.single / 'audit.json', *sorted(args.cohort.glob('*/run/audit.json'))]
    if len(paths) != 11:
        raise ValueError('Expected root plus ten development audits')
    records = []
    for path in paths:
        audit = json.loads(path.read_text())
        if audit['status'] == 'failed' or audit['worker']['actual_device'] != 'cuda:0':
            raise ValueError('Expected successful real CUDA inference')
        records.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'status': audit['status'], 'seconds': audit['elapsed_seconds'],
                        'ocr_seconds': audit['ocr_seconds'],
                        'detections': len(audit['detections'])})
    evidence = {'summary': {k: v for k, v in summary.items() if k not in {'records', 'single'}},
                'root_metrics': {k: v for k, v in summary['single'].items() if k != 'entities'},
                'worker': json.loads(paths[0].read_text())['worker'],
                'records': records,
                'total_image_seconds': sum(r['seconds'] for r in records),
                'total_ocr_seconds': sum(r['ocr_seconds'] for r in records),
                'scope': 'synthetic, source-overlapping development data; not production approval'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps({k: evidence[k] for k in ('total_image_seconds', 'total_ocr_seconds')}))


if __name__ == '__main__':
    main()