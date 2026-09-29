"""Pixel coverage metrics for the sample review interface."""
import hashlib
import json
from collections import defaultdict

import numpy as np
from PIL import Image


def mask_boxes(size, boxes):
    mask = np.zeros(size, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return mask


def evaluate(root, run):
    truth = json.loads((root / "private/truth.json").read_text())
    audit = json.loads((run / "audit.json").read_text())
    if audit["status"] not in {"completed_unassessed", "no_detections_unverified"}:
        raise ValueError("Failed agent run is not evaluable as success")
    input_path, output_path = root / "inputs/sample.png", run / "redacted.png"
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != audit["input_sha256"]:
        raise ValueError("Input hash mismatch")
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != audit["output_sha256"]:
        raise ValueError("Output hash mismatch")
    with Image.open(input_path) as im:
        original = np.array(im.convert("RGB"))
    with Image.open(output_path) as im:
        if im.info or im.mode != 'RGB':
            raise ValueError("Output must be a metadata-free RGB PNG")
        output = np.array(im.convert("RGB"))
    if original.shape != output.shape:
        raise ValueError("Output dimensions mismatch")
    shape = original.shape[:2]
    black = np.all(output == 0, axis=2)
    ink = np.any(original < 250, axis=2)
    applied = mask_boxes(shape, [box for d in audit["detections"] for box in d["boxes"]])
    if not black[applied].all():
        raise ValueError("Mask pixels are not solid black")
    sensitive = mask_boxes(shape, [b for e in truth["entities"] for b in e["boxes"]]) & ink
    results = []
    for entity in truth["entities"]:
        target = mask_boxes(shape, entity["boxes"]) & ink
        covered = int((target & applied & black).sum())
        total = int(target.sum())
        same_label = mask_boxes(shape, [b for d in audit["detections"] if d["label"] == entity["label"] for b in d["boxes"]])
        results.append({"id": entity["id"], "label": entity["label"], "ink_pixels": total,
                        "covered_pixels": covered, "coverage": covered / total if total else 0,
                        "fully_covered": total > 0 and covered == total,
                        "correct_label_fully_covered": total > 0 and int((target & same_label & black).sum()) == total})
    false_detections = sum(not (mask_boxes(shape, d["boxes"]) & sensitive).any() for d in audit["detections"])
    missed = [e for e in results if not e["fully_covered"]]
    nonsensitive = ink & ~sensitive
    metrics = {"uid": truth["uid"], "revision": truth["revision"], "entities": results,
               "entity_count": len(results), "fully_covered_entities": len(results) - len(missed),
               "missed_entity_ids": [e["id"] for e in missed], "missed_labels": sorted({e["label"] for e in missed}),
               "sensitive_ink_coverage": float((sensitive & applied & black).sum() / sensitive.sum()),
               "false_positive_detections_no_sensitive_overlap": int(false_detections),
               "nonsensitive_ink_redacted_fraction": float((nonsensitive & applied).sum() / nonsensitive.sum()),
               "black_rectangle_pixels_verified": int(applied.sum()), "metadata_empty": True,
               "result": "single_image_all_entities_covered" if not missed else "single_image_incomplete",
               "limitation": "One rendered synthetic intl record, not a dataset benchmark. Geometry matching is not semantic precision. No guarantee for unseen labels or scans."}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def measure(base, out):
    audit = out / 'audit.json'
    if audit.exists() and json.loads(audit.read_text())['status'] != 'failed':
        return evaluate(base, out)
    truth = json.loads((base / 'private/truth.json').read_text())
    return {'entities': [{'id': e['id'], 'label': e['label'], 'fully_covered': False,
                          'correct_label_fully_covered': False} for e in truth['entities']],
            'nonsensitive_ink_redacted_fraction': 0, 'failed': True}


def summarize(metrics, gate):
    entities = [e for m in metrics for e in m['entities']]
    labels = defaultdict(list)
    for entity in entities:
        labels[entity['label']].append(entity)
    per_label = {label: {'n': len(items), 'full': sum(e['fully_covered'] for e in items),
                         'correct': sum(e['correct_label_fully_covered'] for e in items)}
                 for label, items in labels.items()}
    overall = sum(e['fully_covered'] for e in entities) / len(entities) if entities else 0
    correct = sum(e['correct_label_fully_covered'] for e in entities) / len(entities) if entities else 0
    zero = sum(bool(m['entities']) and all(e['fully_covered'] for e in m['entities'])
               for m in metrics) / len(metrics) if metrics else 0
    maximum = max((m['nonsensitive_ink_redacted_fraction'] for m in metrics), default=1)
    passed = (len(metrics) >= gate['minimum_records'] and bool(entities)
              and not any(m.get('failed', False) for m in metrics)
              and overall > gate['overall_entity_full_coverage_min']
              and correct > gate['correct_label_full_coverage_min'])
    return {'gate_passed': passed, 'gate_version': gate['version'],
            'n': len(metrics), 'entity_count': len(entities), 'overall': overall,
            'correct_label': correct, 'zero_missed_images': zero,
            'max_nonpii_ink_mask': maximum, 'per_label': per_label, 'records': metrics,
            'scope': 'Fixed synthetic development cohort'}