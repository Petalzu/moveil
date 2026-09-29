"""Synthetic-only tests: no network, weights or inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import ImageFont
import requests

import prepare_dataset as p


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"row_idx": n, "truncated_cells": [], "row": {
            "uid": f"synthetic-{n}", "locale": "intl", "text": "Alice works here.",
            "spans": repr([{"start": 0, "end": 5, "text": "Alice", "label": "first_name"}])}}
            for n in range(50000, 50011)]
        self.expected = {r["row_idx"]: {"uid": r["row"]["uid"]} for r in self.rows}
        self.bundle = {"dataset": p.fixture.DATASET, "revision": p.fixture.REVISION,
                       "served_revision": p.fixture.REVISION, "config": "default",
                       "split": "train", "rows": self.rows}

    def test_rows_valid(self):
        self.assertEqual(p.validate_bundle(self.bundle, self.expected), self.rows)

    def test_font_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            font = Path(directory) / 'consola.ttf'
            with patch.object(p, 'FONT', font), self.assertRaisesRegex(ValueError, '--font'):
                p.resolve_font()
            font.write_bytes(b'font')
            self.assertEqual(p.resolve_font(font), font)
            with patch.object(p, 'FONT', font):
                self.assertEqual(p.resolve_font(), font)

    def test_wrong_font_hash_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            font = Path(directory) / 'consola.ttf'
            font.write_bytes(b'wrong font')
            with patch.object(p, 'download_rows') as download, self.assertRaisesRegex(ValueError, 'font SHA256'):
                p.prepare(Path(directory) / 'output', font=font)
            download.assert_not_called()

    def test_bad_provenance(self):
        for key in ("dataset", "revision", "served_revision", "config", "split"):
            with self.subTest(key=key):
                bundle = dict(self.bundle, **{key: "wrong"})
                with self.assertRaises(ValueError):
                    p.validate_bundle(bundle, self.expected)

    def test_bad_rows(self):
        for field, value in (("uid", "wrong"), ("locale", "en"), ("text", ""),
                             ("spans", "[]")):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0]["row"][field] = value
                with self.assertRaises(ValueError):
                    p.validate_rows(rows, self.expected)
        for rows in (self.rows[:-1], self.rows[::-1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError):
                p.validate_rows(rows, self.expected)
        rows = copy.deepcopy(self.rows)
        rows[0]["truncated_cells"] = ["text"]
        with self.assertRaises(ValueError):
            p.validate_rows(rows, self.expected)

    def test_span_repair_and_ambiguity(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["row"]["spans"] = repr([{
            "start": 1, "end": 6, "text": "Alice", "label": "first_name"}])
        p.validate_rows(rows, self.expected)
        rows[1]["row"]["text"] = "Alice Alice"
        with self.assertRaises(ValueError):
            p.validate_rows(rows, self.expected)
        rows[0]["row"]["spans"] = rows[1]["row"]["spans"]
        with self.assertRaises(ValueError):
            p.validate_rows(rows, self.expected)

    def responses(self):
        metadata = Mock()
        metadata.json.return_value = {"sha": p.fixture.REVISION}
        response = Mock(headers={"x-revision": p.fixture.REVISION}, url=p.fixture.URL)
        response.json.return_value = {"rows": self.rows}
        return metadata, response

    def test_download_verified(self):
        with patch.object(p.requests, "get", side_effect=self.responses()) as get:
            p.validate_bundle(p.download_rows(self.expected), self.expected)
        self.assertEqual([c.kwargs["timeout"] for c in get.call_args_list], [60, 90])
        self.assertEqual(get.call_args.kwargs["params"]["length"], 11)
        self.assertNotIn("revision", get.call_args.kwargs["params"])

    def test_main_changed(self):
        metadata, _ = self.responses()
        metadata.json.return_value = {"sha": "new-main"}
        with patch.object(p.requests, "get", return_value=metadata) as get:
            with self.assertRaises(ValueError):
                p.download_rows(self.expected)
        self.assertEqual(get.call_count, 1)

    def test_missing_or_wrong_revision(self):
        for revision in (None, "wrong"):
            metadata, response = self.responses()
            response.headers = {} if revision is None else {"x-revision": revision}
            with patch.object(p.requests, "get", side_effect=[metadata, response]):
                with self.assertRaises(ValueError):
                    p.download_rows(self.expected)

    def test_http_and_timeout(self):
        for index in (0, 1):
            responses = self.responses()
            responses[index].raise_for_status.side_effect = requests.HTTPError("HTTP failure")
            with patch.object(p.requests, "get", side_effect=responses):
                with self.assertRaises(requests.HTTPError):
                    p.download_rows(self.expected)
        with patch.object(p.requests, "get", side_effect=requests.Timeout):
            with self.assertRaises(requests.Timeout):
                p.download_rows(self.expected)

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for nonempty in (False, True):
                if nonempty:
                    (output / "keep.txt").write_text("keep")
                with patch.object(p, "download_rows") as download:
                    with self.assertRaises(ValueError):
                        p.prepare(output)
                    download.assert_not_called()
            self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_offline_render_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            font = root / "font.ttf"
            font.write_bytes(b"test font identity")
            gate = {"revision": p.fixture.REVISION, "holdout_offsets": list(range(50001, 50011))}
            source = {"font_sha256": p.digest(font), "record_sha256": hashlib.sha256(
                json.dumps(self.rows[0]["row"], ensure_ascii=False, indent=2).encode()).hexdigest()}
            records = root / "records.json"
            records.write_text(json.dumps(self.bundle), encoding="utf-8")
            reference = root / "reference"
            reference.mkdir()
            # Same renderer, platform-independent test font; production requires Consolas SHA.
            test_font = ImageFont.load_default(size=30)
            with patch.object(ImageFont, "truetype", return_value=test_font):
                p.fixture.render_record(self.rows[0]["row"], reference, p.fixture.REVISION, p.fixture.URL, font)
                cohort = reference / "cohorts" / p.COHORT
                cohort.mkdir(parents=True)
                p.benchmark.render_rows(cohort, gate, self.rows[1:], font)
                for offset in self.expected:
                    base = reference if offset == 50000 else cohort / str(offset)
                    self.expected[offset]["input_sha256"] = p.digest(base / "inputs/sample.png")
                (reference / "quality_gate.json").write_text(json.dumps(gate))
                with patch.object(p, "ROOT", reference), patch.object(p, "FONT", font), patch.object(
                        p, "expectations", return_value=(source, gate, self.expected)), patch.object(
                        p.requests, "get", side_effect=AssertionError("Network forbidden")):
                    output = p.prepare(root / "prepared", records, font=font)
                    self.assertEqual(len(list(output.rglob("sample.png"))), 11)
                    manifest = p.read_json(output / "manifest.json")
                    for name, sha in manifest["files"].items():
                        self.assertEqual(p.digest(output / name), sha)
                    self.assertEqual(p.read_json(output / "private/source-records.json"), self.bundle)
                    self.expected[50000]["input_sha256"] = "wrong"
                    with self.assertRaisesRegex(ValueError, "image SHA256"):
                        p.prepare(root / "bad", records)
                    self.assertFalse((root / "bad").exists())
                    self.assertFalse(list(root.glob(".prepare-dataset-*")))


if __name__ == "__main__":
    unittest.main()