import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from agent import run, projection_ocr
from benchmark import summarize


class PipelineTests(unittest.TestCase):
    def test_new_pipeline_no_truth_and_verified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory)/"image.png", Path(directory)/"run"
            Image.new("RGB",(50,30),"white").save(source)
            lines = [{"text":"Alice", "units":[{"start":0,"end":5,"polygon":[[10,10],[30,10],[30,20],[10,20]]}]}]
            reply = Mock()
            def respond(*args, **kwargs):
                labels = kwargs['json']['response_format']['json_schema']['schema']['properties']['entities']['items']['properties']['label']['enum']
                entities = [{'label': 'first_name', 'text': 'Alice'}] if 'first_name' in labels else []
                reply.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'entities': entities})}}]}
                return reply
            with patch("agent.projection_ocr",return_value=lines), patch("agent.requests.post",side_effect=respond):
                self.assertTrue(run(source,out,"http://127.0.0.1","mock"))
            audit = json.loads((out/"audit.json").read_text())
            self.assertEqual(audit["pipeline"], "exact-text-v1")
            self.assertEqual(len(audit["trace"]), 4)
            self.assertEqual(len(audit["detections"]),1)
            self.assertNotIn("Alice",(out/"audit.json").read_text())
            with Image.open(out/"redacted.png") as image:
                self.assertEqual(image.getpixel((15,15)),(0,0,0))
                self.assertEqual(image.getpixel((0,0)),(255,255,255))
            with self.assertRaises(FileExistsError):
                run(source,out,"http://127.0.0.1","mock")

    def test_new_pipeline_bad_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory)/"image.png", Path(directory)/"run"
            Image.new("RGB",(50,30),"white").save(source)
            reply = Mock()
            reply.json.return_value = {"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"entities":[{"label":"first_name","text":"SECRET"}]})}}]}
            with patch("agent.projection_ocr",return_value=[{"text":"public"}]), patch("agent.requests.post",return_value=reply) as post:
                self.assertFalse(run(source,out,"http://127.0.0.1","mock"))
                self.assertEqual(post.call_count,3)
            self.assertFalse((out/"redacted.png").exists())
            self.assertNotIn("SECRET",(out/"audit.json").read_text())

    def test_blank_and_dark_images_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"image.png"
            for color in ["white","black"]:
                Image.new("RGB",(40,40),color).save(source)
                with self.assertRaises(ValueError):
                    projection_ocr(source)

    def test_frozen_gate_rejects_missed_label_and_overmask(self):
        gate=json.loads(Path(__file__).with_name("quality_gate.json").read_text())
        entity={"label":"email","fully_covered":True,"correct_label_fully_covered":True}
        records=[{"entities":[dict(entity)],"nonsensitive_ink_redacted_fraction":0} for _ in range(10)]
        self.assertTrue(summarize(records,gate)["gate_passed"])
        self.assertFalse(summarize(records[:9],gate)["gate_passed"])
        records[0]["nonsensitive_ink_redacted_fraction"]=0.11
        self.assertFalse(summarize(records,gate)["gate_passed"])
        records[0]["nonsensitive_ink_redacted_fraction"]=0
        records[0]["entities"][0]["correct_label_fully_covered"]=False
        self.assertFalse(summarize(records,gate)["gate_passed"])
        records[0]["entities"][0].update(label="rare",fully_covered=False)
        self.assertFalse(summarize(records,gate)["gate_passed"])


if __name__ == "__main__":
    unittest.main()