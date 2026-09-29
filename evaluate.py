"""Independent evaluator: annotations are accessible here, never to agent.py."""
import argparse
import hashlib
import json
from pathlib import Path

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
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == audit["input_sha256"]
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == audit["output_sha256"]
    with Image.open(input_path) as im:
        original = np.array(im.convert("RGB"))
    with Image.open(output_path) as im:
        assert not im.info
        output = np.array(im.convert("RGB"))
    assert original.shape == output.shape
    shape = original.shape[:2]
    black = np.all(output == 0, axis=2)
    ink = np.any(original < 250, axis=2)
    applied = mask_boxes(shape, [box for d in audit["detections"] for box in d["boxes"]])
    assert black[applied].all()
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
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evaluate(Path(__file__).resolve().parent, args.run)