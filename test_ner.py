import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from agent import LABELS
from ner_config import CONFIG, LABEL_MAP, POLICY
from ner_pipeline import map_spans, run
from ner_worker import predict, windows


class NerTests(unittest.TestCase):
    def test_policy_not_filtered_to_dev_labels(self):
        self.assertEqual(set(POLICY), LABELS)
        self.assertTrue(set(LABEL_MAP.values()) <= LABELS)
        self.assertEqual(LABEL_MAP['postcode'], 'postal_code')
        self.assertEqual(LABEL_MAP['credit_card_number'], 'credit_debit_card')
        self.assertIn('occupation', LABEL_MAP)

    def test_offsets_do_not_expand_repeated_text(self):
        lines = [{'text': 'Mira'}, {'text': 'and Mira'}]
        entities = [{'start': 9, 'end': 13, 'label': 'first_name', 'score': .9}]
        self.assertEqual(map_spans(entities, lines), [{'line': 1, 'start': 4, 'end': 8, 'label': 'first_name'}])

    def test_cross_line_span(self):
        result = map_spans([{'start': 0, 'end': 8, 'label': 'occupation', 'score': .7}],
                           [{'text': 'art'}, {'text': 'lead'}])
        self.assertEqual([(e['line'], e['start'], e['end']) for e in result], [(0, 0, 3), (1, 0, 4)])

    def test_bad_protocol_rejected(self):
        valid = {'start': 0, 'end': 4, 'label': 'first_name', 'score': .8}
        for changed in ({'start': True}, {'end': 9}, {'score': float('nan')}, {'score': .49},
                        {'label': 'invented'}, {'text': 'Mira'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                map_spans([{**valid, **changed}], [{'text': 'Mira'}])

    def test_window_coverage_and_budget(self):
        import re
        model = Mock()
        model.data_processor.words_splitter.side_effect = lambda s: [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', s)]
        model.data_processor.transformer_tokenizer.side_effect = lambda s, **kw: {'input_ids': list(range(len(s.split()) * 6 + 2))}
        text = ' '.join(['word'] * 240)
        parts = windows(model, text)
        covered = set()
        for start, part in parts:
            covered.update(range(start, start+len(part)))
            self.assertLessEqual(len(part.split())*6+2, CONFIG['max_subtokens'])
        self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))
        self.assertGreater(len(parts), 1)

    def test_predict_exact_evidence_and_overlap(self):
        model = Mock()
        model.inference.return_value = [[{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'first_name', 'score': .9}],
                                       [{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'last_name', 'score': .7}]]
        with patch('ner_worker.windows', return_value=[(0, 'Mira'), (0, 'Mira')]):
            entities, count = predict(model, 'Mira')
            self.assertEqual(count, 2)
            self.assertEqual(len(entities), 1)
            self.assertEqual(entities[0]['label'], 'first_name')
            self.assertNotIn('text', entities[0])
            model.inference.return_value[0][0]['text'] = 'other'
            with self.assertRaises(ValueError):
                predict(model, 'Mira')

    def test_image_run_no_plaintext_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            image, out = Path(directory)/'image.png', Path(directory)/'run'
            Image.new('RGB', (50, 40), 'white').save(image)
            lines = [{'text': 'Mira', 'units': [{'start': 0, 'end': 4, 'polygon': [[10,10],[30,10],[30,20],[10,20]]}]}]
            worker = Mock(metadata={'status': 'ready'})
            worker.predict.return_value = {'entities': [{'start': 0, 'end': 4, 'label': 'first_name', 'score': .9}], 'windows': 1}
            with patch('ner_pipeline.projection_ocr', return_value=lines):
                self.assertTrue(run(image, out, worker))
                self.assertNotIn('Mira', (out/'audit.json').read_text())
                self.assertFalse(json.loads((out/'audit.json').read_text())['openclaw_host_verified'])
                worker.predict.side_effect = ValueError('PRIVATE TEXT')
                failed = Path(directory)/'failed'
                self.assertFalse(run(image, failed, worker))
                self.assertFalse((failed/'redacted.png').exists())
                self.assertNotIn('PRIVATE TEXT', (failed/'audit.json').read_text())


if __name__ == '__main__':
    unittest.main()