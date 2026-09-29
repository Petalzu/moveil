"""Offline fixture producer only. Never imported by the agent."""
import ast
import hashlib
import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
DATASET = "nvidia/Nemotron-PII"
URL = "https://datasets-server.huggingface.co/rows"


def main():
    private = ROOT / "private"
    inputs = ROOT / "inputs"
    private.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)
    params = dict(dataset=DATASET, config="default", split="train", offset=50000, length=1)
    before = requests.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60)
    before.raise_for_status()
    assert before.json()["sha"] == REVISION, "Dataset main changed; refusing unpinned rows"
    response = requests.get(URL, params=params, timeout=90)
    response.raise_for_status()
    # datasets-server serves a cached revision; require explicit matching evidence.
    served_revision = response.headers.get("x-revision")
    assert served_revision == REVISION, f"Unverified rows revision: {served_revision}"
    payload = response.json()
    item = payload["rows"][0]
    assert not item["truncated_cells"]
    row = item["row"]
    render_record(row, ROOT, served_revision, response.url)


def render_record(row, destination, served_revision, source_url, font_path=None):
    """Render a validated source row without downloading or running inference."""
    private = destination / "private"
    inputs = destination / "inputs"
    private.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)
    assert row["locale"] == "intl"
    spans = ast.literal_eval(row["spans"])
    text = row["text"]
    for i, span in enumerate(spans):
        assert text[span["start"]:span["end"]] == span["text"], f"Invalid source span {i}"
    (private / "record.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    font_path = Path(font_path) if font_path is not None else Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 30)
    cell = int(round(font.getlength("M")))
    positions = []
    x, y = 40, 40
    # Rendering uses only text, not entity locations. Fixed wrapping at 72 characters.
    for char in text:
        if char == "\n" or x + cell > 40 + cell * 72:
            x, y = 40, y + 48
        positions.append((x, y))
        if char != "\n":
            x += cell
    image = Image.new("RGB", (80 + cell * 72, y + 88), "white")
    draw = ImageDraw.Draw(image)
    boxes = []
    for char, pos in zip(text, positions):
        draw.text(pos, char, font=font, fill="black", anchor="la")
        boxes.append(list(draw.textbbox(pos, char, font=font, anchor="la")))
    image.save(inputs / "sample.png")
    truth = {"uid": row["uid"], "locale": row["locale"], "revision": REVISION,
             "entities": [{"id": i, "label": s["label"], "start": s["start"], "end": s["end"],
                           "boxes": [boxes[j] for j in range(s["start"], s["end"]) if not text[j].isspace()]}
                          for i, s in enumerate(spans)],
             "characters": [{"box": boxes[i], "sensitive": any(s["start"] <= i < s["end"] for s in spans)}
                            for i, c in enumerate(text) if not c.isspace()]}
    (private / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    source = {"dataset": DATASET, "revision": REVISION, "served_revision": served_revision,
              "url": source_url, "split": "train", "row_idx": 50000, "uid": row["uid"], "locale": "intl",
              "labels": sorted({s["label"] for s in spans}), "record_sha256": hashlib.sha256((private / "record.json").read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256((inputs / "sample.png").read_bytes()).hexdigest(),
              "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
              "authors": "Amy Steier, Andre Manoel, Alexa Haushalter, Maarten Van Segbroeck; NVIDIA (2025)",
              "modifications": "Text rendered into PNG; independent character/pixel annotations added. Agent output redacted.",
              "size_discrepancy": "Card: 100000 total, 50000 per split; viewer: 200000 total; rows API: 100000 train. Not resolved.",
              "font": str(font_path), "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()}
    (destination / "SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps({"fixture": "ready", "uid": row["uid"], "labels": source["labels"]}))


if __name__ == "__main__":
    main()