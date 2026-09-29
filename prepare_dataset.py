"""Prepare the fixed synthetic fixture only; no model download or inference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import requests

import benchmark
import fixture

ROOT = Path(__file__).resolve().parent
COHORT = "nvidia-gliner-pii-dev-015"
FONT = Path("C:/Windows/Fonts/consola.ttf")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expectations():
    source = read_json(ROOT / "SOURCE.json")
    gate = read_json(ROOT / "quality_gate.json")
    manifest = read_json(ROOT / "cohorts" / COHORT / "manifest.json")
    if (source["revision"] != fixture.REVISION or gate["revision"] != fixture.REVISION
            or source["row_idx"] != 50000
            or gate["holdout_offsets"] != list(range(50001, 50011))
            or [r["offset"] for r in manifest] != gate["holdout_offsets"]):
        raise ValueError("Unexpected source/gate configuration")
    expected = {50000: source}
    expected.update({r["offset"]: r for r in manifest})
    return source, gate, expected


def validate_rows(rows, expected):
    if not isinstance(rows, list) or [r.get("row_idx") for r in rows] != list(expected):
        raise ValueError("Wrong row offsets/order/count")
    for item in rows:
        row = item["row"]
        if item.get("truncated_cells") != [] or row.get("locale") != "intl":
            raise ValueError("Truncated record or wrong locale")
        if row.get("uid") != expected[item["row_idx"]]["uid"]:
            raise ValueError("Wrong source UID")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Missing source text")
        spans = ast.literal_eval(row["spans"])
        if not isinstance(spans, list) or not spans:
            raise ValueError("Missing source spans")
        for span in spans:
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int
                    or not isinstance(span["label"], str) or not span["label"]):
                raise ValueError("Invalid span fields")
            value = span["text"] if item["row_idx"] == 50000 else str(span["text"])
            if not isinstance(value, str) or not value:
                raise ValueError("Invalid span text")
            aligned = 0 <= start < end <= len(text) and text[start:end] == value
            # Keep benchmark's unique-text offset repair, but reject ambiguity.
            if not aligned and (item["row_idx"] == 50000 or text.count(value) != 1):
                raise ValueError("Invalid or ambiguous source span")
    return rows


def download_rows(expected):
    """The rows API has no pinned-revision promise: fail closed, never fall back."""
    metadata = requests.get(f"https://huggingface.co/api/datasets/{fixture.DATASET}", timeout=60)
    metadata.raise_for_status()
    if metadata.json().get("sha") != fixture.REVISION:
        raise ValueError("Dataset main changed; refusing unpinned rows. Use verified offline records.")
    offsets = list(expected)
    response = requests.get(fixture.URL, params={
        "dataset": fixture.DATASET, "config": "default", "split": "train",
        "offset": offsets[0], "length": len(offsets)}, timeout=90)
    response.raise_for_status()
    if response.headers.get("x-revision") != fixture.REVISION:
        raise ValueError("Missing or mismatched x-revision; refusing latest data")
    rows = validate_rows(response.json()["rows"], expected)
    return {"dataset": fixture.DATASET, "revision": fixture.REVISION,
            "served_revision": fixture.REVISION, "config": "default", "split": "train",
            "source_url": response.url, "rows": rows}


def validate_bundle(bundle, expected):
    for key, value in {"dataset": fixture.DATASET, "revision": fixture.REVISION,
                       "served_revision": fixture.REVISION, "config": "default", "split": "train"}.items():
        if bundle.get(key) != value:
            raise ValueError(f"Wrong offline provenance: {key}")
    return validate_rows(bundle["rows"], expected)


def resolve_font(font=None):
    path = Path(font).expanduser() if font is not None else FONT
    if not path.is_file():
        raise ValueError("Provide --font /path/to/consola.ttf, legally obtained and matching SOURCE.json")
    return path


def prepare(output, records=None, font=None):
    output = output.absolute()
    # Require a NEW destination (even an empty existing directory is refused).
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a new directory")
    source, gate, expected = expectations()
    font_path = resolve_font(font)
    if digest(font_path) != source["font_sha256"]:
        raise ValueError("Exact Consolas font required; SOURCE.json font SHA256 mismatch")
    bundle = read_json(records) if records else download_rows(expected)
    rows = validate_bundle(bundle, expected)
    # Match original record serialization before rendering.
    record_bytes = json.dumps(rows[0]["row"], ensure_ascii=False, indent=2).encode("utf-8")
    if hashlib.sha256(record_bytes).hexdigest() != source["record_sha256"]:
        raise ValueError("Root source record SHA256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-dataset-", dir=output.parent) as directory:
        stage = Path(directory) / "prepared"
        stage.mkdir()
        fixture.render_record(rows[0]["row"], stage, fixture.REVISION, fixture.URL, font_path)
        cohort = stage / "cohorts" / COHORT
        cohort.mkdir(parents=True)
        benchmark.render_rows(cohort, gate, rows[1:], font_path)
        for offset, reference in expected.items():
            base = stage if offset == 50000 else cohort / str(offset)
            if digest(base / "inputs/sample.png") != reference["input_sha256"]:
                raise ValueError(f"Rendered image SHA256 mismatch at {offset}")
            if offset != 50000:
                original = ROOT / "cohorts" / COHORT / str(offset) / "private/truth.json"
            else:
                original = ROOT / "private/truth.json"
            if read_json(base / "private/truth.json") != read_json(original):
                raise ValueError(f"Truth geometry mismatch at {offset}")
        (stage / "private/source-records.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "quality_gate.json").write_bytes((ROOT / "quality_gate.json").read_bytes())
        files = {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()}
        (stage / "manifest.json").write_text(json.dumps({
            "dataset": fixture.DATASET, "revision": fixture.REVISION,
            "scope": "Synthetic development data; training-source overlap, not holdout",
            "font_sha256": digest(font_path), "files": files}, indent=2), encoding="utf-8")
        stage.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data-prepared")
    parser.add_argument("--records", type=Path, help="Offline source-records.json from a verified preparation")
    parser.add_argument("--font", type=Path, help="Legally obtained Consolas TTF matching SOURCE.json (required on Linux)")
    args = parser.parse_args()
    try:
        result = prepare(args.output, args.records, args.font)
    except (ValueError, KeyError, TypeError, SyntaxError, OSError, requests.RequestException) as exc:
        parser.exit(1, f"Dataset preparation refused: {exc}\n")
    print(f"Prepared 11 synthetic images, truth and SHA256 manifest: {result}")


if __name__ == "__main__":
    main()