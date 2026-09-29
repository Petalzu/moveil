import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from agent import expand_action, redact, legacy_run as run, tokens_from_lines, validate, exact_spans
from unittest.mock import patch


class SafetyTests(unittest.TestCase):
    def test_exact_cross_fragment_and_repeated(self):
        lines = [{"text": "Alice works as software"}, {"text": "engineer. Alice agrees."}]
        spans = exact_spans([{"label": "occupation", "text": "software engineer"}, {"label": "first_name", "text": "Alice"}], lines)
        self.assertEqual(len(spans), 4)
        validate({"action": "redact", "entities": spans}, 1, lines)

    def test_exact_rejects_invented_and_partial_words(self):
        for text in ["retired", "work", "", "Works"]:
            with self.assertRaises(ValueError):
                exact_spans([{"label": "occupation", "text": text}], [{"text": "works"}])

    def test_wrapped_email_preserves_inverse_coordinates(self):
        lines = [{"text": "Contact a@exam"}, {"text": "ple.org."}]
        spans = exact_spans([{"label": "email", "text": "a@example.org"}], lines)
        self.assertEqual([(s["line"],s["start"],s["end"]) for s in spans], [(0,8,14),(1,0,7)])

    def test_token_ids_do_not_bridge_unselected_text(self):
        lines = [{"text": "Alice public Bob"}]
        tokens = tokens_from_lines(lines)
        action = expand_action({"action": "redact", "entities": [{"label": "first_name", "matches": [{"token_id": 0, "evidence": "Alice"}, {"token_id": 2, "evidence": "Bob"}]}]}, tokens)
        validate(action, 1, lines)
        self.assertEqual([(e["start"], e["end"]) for e in action["entities"]], [(0, 5), (13, 16)])

    def test_invalid_token_ids(self):
        tokens = tokens_from_lines([{"text": "Alice"}])
        for ids in [[True], [-1], [1], ["0"], []]:
            with self.assertRaises(ValueError):
                expand_action({"action": "redact", "entities": [{"label": "first_name", "matches": [{"token_id": i, "evidence": "Alice"} for i in ids]}]}, tokens)

    def test_evidence_mismatch_and_ambiguity(self):
        tokens = tokens_from_lines([{"text": "works retired banana"}])
        for idx, evidence in [(0, "retired"), (1, "Retired"), (1, ""), (1, None), (2, "ana")]:
            with self.assertRaises(ValueError):
                expand_action({"action": "redact", "entities": [{"label": "employment_status", "matches": [{"token_id": idx, "evidence": evidence}]}]}, tokens)

    def test_evidence_exact_substring_offsets(self):
        tokens = tokens_from_lines([{"text": "Born(2001-02-03)."}])
        action = expand_action({"action": "redact", "entities": [{"label": "date_of_birth", "matches": [{"token_id": 0, "evidence": "2001-02-03"}]}]}, tokens)
        self.assertEqual(action["entities"][0], {"label": "date_of_birth", "line": 0, "start": 5, "end": 15})

    def test_mismatch_retries_then_fails_closed(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.png"
            Image.new("RGB", (20, 20)).save(source)
            out = Path(directory) / "run"
            def reply(action):
                response = Mock()
                response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(action)}}]}
                return response
            bad = {"action": "redact", "entities": [{"label": "employment_status", "matches": [{"token_id": 0, "evidence": "SECRET"}]}]}
            with patch("agent.ocr", return_value=[{"text": "public"}]), patch("agent.requests.post", side_effect=[reply({"action": "ocr"}), reply(bad), reply(bad), reply(bad)]) as post:
                self.assertFalse(run(source, out, "http://127.0.0.1", "mock"))
                self.assertEqual(post.call_count, 4)
            audit = (out / "audit.json").read_text()
            self.assertNotIn("SECRET", audit)
            self.assertEqual(sum(not t["validated"] for t in json.loads(audit)["trace"]), 3)
            self.assertFalse((out / "redacted.png").exists())

    def test_no_fixture_imports(self):
        import ast
        import agent
        tree = ast.parse(Path(agent.__file__).read_text(encoding="utf-8"))
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn("fixture", modules)
        self.assertNotIn("evaluate", modules)

    def test_reject_bad_actions(self):
        for action in [{"action": "shell"}, {"action": "ocr", "path": "private/truth.json"}, [], {"action": "finish"}]:
            with self.assertRaises(ValueError):
                validate(action, 0, [])

    def test_validate_offsets_and_labels(self):
        lines = [{"text": "Some text"}]
        base = {"label": "first_name", "line": 0, "start": 0, "end": 4}
        validate({"action": "redact", "entities": [base]}, 1, lines)
        for delta in [{"start": -1}, {"end": 100}, {"line": True}, {"label": "shell"}, {"path": "x"}]:
            with self.assertRaises(ValueError):
                validate({"action": "redact", "entities": [{**base, **delta}]}, 1, lines)

    def test_solid_new_png_no_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.png", Path(directory) / "out.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("private", "must disappear")
            Image.new("RGB", (40, 40), "white").save(source, pnginfo=info)
            lines = [{"text": "John", "units": [{"start": 0, "end": 4, "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]}]}]
            redact(source, output, [{"line": 0, "start": 0, "end": 4, "label": "first_name"}], lines)
            with Image.open(output) as im:
                self.assertFalse(im.info)
                self.assertEqual(im.getpixel((15, 15)), (0, 0, 0))
                self.assertEqual(im.getpixel((0, 0)), (255, 255, 255))

    def test_failure_not_success_or_sensitive_log(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.png"
            Image.new("RGB", (20, 20)).save(source)
            out = Path(directory) / "run"
            with patch("agent.requests.post", side_effect=RuntimeError("SECRET")):
                self.assertFalse(run(source, out, "http://127.0.0.1", "mock"))
            content = (out / "audit.json").read_text()
            self.assertNotIn("SECRET", content)
            self.assertEqual(json.loads(content)["status"], "failed")
            self.assertFalse((out / "redacted.png").exists())


if __name__ == "__main__":
    unittest.main()