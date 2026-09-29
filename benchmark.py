"""Fixed smoke cohort. Annotation handling lives here, never in inference."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent


def prepare(destination, gate):
    destination.mkdir(parents=True, exist_ok=False)
    response = requests.get("https://datasets-server.huggingface.co/rows", params={
        "dataset": "nvidia/Nemotron-PII", "config": "default", "split": "train",
        "offset": gate["holdout_offsets"][0], "length": len(gate["holdout_offsets"])}, timeout=90)
    response.raise_for_status()
    if response.headers.get("x-revision") != gate["revision"]:
        raise ValueError("Unpinned dataset response")
    rows = response.json()["rows"]
    render_rows(destination, gate, rows)


def render_rows(destination, gate, rows, font_path=None):
    """Render already validated records; never invoke inference."""
    if [r["row_idx"] for r in rows] != gate["holdout_offsets"]:
        raise ValueError("Wrong cohort")
    font = ImageFont.truetype(str(font_path or "C:/Windows/Fonts/consola.ttf"), 30)
    cell = round(font.getlength("M"))
    manifest = []
    for item in rows:
        row = item["row"]
        if item["truncated_cells"] or row["locale"] != "intl":
            raise ValueError("Invalid record")
        text, spans = row["text"], ast.literal_eval(row["spans"])
        repairs = []
        for span in spans:
            if not isinstance(span["text"], str):
                span["text"] = str(span["text"])
            if text[span["start"]:span["end"]] != span["text"]:
                if not span["text"] or text.count(span["text"]) != 1:
                    raise ValueError(f"Ambiguous source span at row {item['row_idx']}: {span['label']}")
                start = text.index(span["text"])
                repairs.append({"label": span["label"], "old": [span["start"],span["end"]], "new": [start,start+len(span["text"])]})
                span["start"], span["end"] = start, start+len(span["text"])
        base = destination / str(item["row_idx"])
        (base / "inputs").mkdir(parents=True)
        (base / "private").mkdir()
        positions, x, y = [], 40, 40
        for char in text:
            if char == "\n" or x + cell > 40 + cell*72:
                x, y = 40, y+48
            positions.append((x,y))
            if char != "\n":
                x += cell
        image = Image.new("RGB", (80+cell*72,y+88), "white")
        draw, boxes = ImageDraw.Draw(image), []
        for char, pos in zip(text, positions):
            draw.text(pos,char,font=font,fill="black",anchor="la")
            boxes.append(list(draw.textbbox(pos,char,font=font,anchor="la")))
        image.save(base / "inputs/sample.png")
        truth = {"uid": row["uid"], "revision": gate["revision"], "entities": [
            {"id": i, "label": s["label"], "boxes": [boxes[j] for j in range(s["start"],s["end"]) if not text[j].isspace()]}
            for i,s in enumerate(spans)]}
        (base / "private/truth.json").write_text(json.dumps(truth), encoding="utf-8")
        manifest.append({"offset": item["row_idx"], "uid": row["uid"], "annotation_repairs": repairs, "input_sha256": hashlib.sha256((base / "inputs/sample.png").read_bytes()).hexdigest()})
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def summarize(metrics, gate):
    entities = [e for m in metrics for e in m["entities"]]
    labels = defaultdict(list)
    for e in entities:
        labels[e["label"]].append(e)
    per_label = {k: {"n": len(v), "full": sum(e["fully_covered"] for e in v),
                    "correct": sum(e["correct_label_fully_covered"] for e in v)} for k,v in labels.items()}
    overall = sum(e["fully_covered"] for e in entities)/len(entities) if entities else 0
    correct = sum(e["correct_label_fully_covered"] for e in entities)/len(entities) if entities else 0
    zero = sum(bool(m["entities"]) and all(e["fully_covered"] for e in m["entities"]) for m in metrics)/len(metrics) if metrics else 0
    maximum = max((m["nonsensitive_ink_redacted_fraction"] for m in metrics), default=1)
    passed = (len(metrics) >= gate["minimum_records"] and overall >= gate["overall_entity_full_coverage_min"]
              and correct >= gate["correct_label_full_coverage_min"] and zero >= gate["zero_missed_image_fraction_min"]
              and maximum <= gate["per_image_nonpii_ink_false_mask_max"]
              and all(v["full"]/v["n"] >= gate["per_label_entity_full_coverage_min"] for v in per_label.values()))
    return {"gate_passed": passed, "n": len(metrics), "entity_count": len(entities), "overall": overall,
            "correct_label": correct, "zero_missed_images": zero, "max_nonpii_ink_mask": maximum,
            "per_label": per_label, "records": metrics, "scope": "Fixed synthetic smoke cohort only, not production assurance"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument('--from-cohort', type=Path, help='Copy only development inputs/truth, never prior run outputs')
    args = parser.parse_args()
    gate_bytes = (ROOT / "quality_gate.json").read_bytes()
    gate = json.loads(gate_bytes)
    if args.from_cohort:
        if args.prepare:
            raise ValueError('Choose prepare or development copy')
        args.destination.mkdir(parents=True, exist_ok=False)
        for offset in gate['holdout_offsets']:
            for folder in ('inputs', 'private'):
                shutil.copytree(args.from_cohort / str(offset) / folder, args.destination / str(offset) / folder)
        shutil.copyfile(args.from_cohort / 'manifest.json', args.destination / 'manifest.json')
    config = {'model': 'pcb-vision', 'temperature': 0, 'max_tokens': 1400,
              'scope': 'development regression' if args.from_cohort else 'fixed cohort',
              'gate_sha256': hashlib.sha256(gate_bytes).hexdigest(),
              'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('agent.py', 'pii_rules.py', 'benchmark.py')}}
    if args.prepare:
        prepare(args.destination, gate)
    (args.destination / 'inference_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    results = []
    for offset in gate["holdout_offsets"]:
        base = args.destination / str(offset)
        out = base / "run"
        if out.exists():
            raise ValueError("Refusing reused cohort results")
        process = subprocess.run([sys.executable, str(ROOT / "agent.py"), str(base / "inputs/sample.png"), "--out", str(out)], check=False)
        if process.returncode == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                metric = evaluate(base, out)
        else:
            truth = json.loads((base / "private/truth.json").read_text())
            metric = {"entities": [{"id": e["id"], "label": e["label"], "fully_covered": False, "correct_label_fully_covered": False} for e in truth["entities"]],
                      "nonsensitive_ink_redacted_fraction": 0, "failed": True}
        results.append({"offset": offset, **metric})
        print(json.dumps({"offset": offset, "full": sum(e["fully_covered"] for e in metric["entities"]), "n": len(metric["entities"])}), flush=True)
    summary = summarize(results, gate)
    summary["gate_sha256"] = hashlib.sha256(gate_bytes).hexdigest()
    summary["agent_sha256"] = hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest()
    (args.destination / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"records","per_label"}},indent=2))


if __name__ == "__main__":
    main()