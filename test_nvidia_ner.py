import unittest
from unittest.mock import Mock, patch

from nvidia_ner_config import CONFIG, CONFIG_HASH, LABEL_GROUPS, PROMPTS
from nvidia_ner_pipeline import validate_ready
from nvidia_ner_worker import predict


class NvidiaNerTests(unittest.TestCase):
    def test_cuda_handshake_rejects_cpu_fallback(self):
        with patch.dict(CONFIG, device='cuda'):
            with self.assertRaisesRegex(ValueError, 'device mismatch'):
                validate_ready({'status': 'ready', 'config_sha256': CONFIG_HASH,
                                'actual_device': 'cpu'})
            validate_ready({'status': 'ready', 'config_sha256': CONFIG_HASH,
                            'actual_device': 'cuda:0'})

    def test_handshake_rejects_wrong_config(self):
        with self.assertRaises(ValueError):
            validate_ready({'status': 'ready', 'config_sha256': 'wrong',
                            'actual_device': 'cuda:0'})

    def test_all_prompts_are_in_safe_size_groups(self):
        flattened = [label for group in LABEL_GROUPS for label in group]
        self.assertEqual(flattened, PROMPTS)
        self.assertTrue(all(0 < len(group) <= CONFIG['model_max_types']
                            for group in LABEL_GROUPS))

    def test_predict_queries_each_group_and_validates_evidence(self):
        model = Mock()
        model.inference.side_effect = [
            [[{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'first_name', 'score': .91}]],
            [[]],
            [[]],
        ]
        with patch('nvidia_ner_worker.windows', return_value=[(0, 'Mira')]):
            entities, count = predict(model, 'Mira')
        self.assertEqual(count, 1)
        self.assertEqual(len(model.inference.call_args_list), len(LABEL_GROUPS))
        self.assertEqual(entities, [{'start': 0, 'end': 4, 'label': 'first_name', 'score': .91}])
        self.assertTrue(all(call.kwargs['threshold'] == CONFIG['threshold']
                            for call in model.inference.call_args_list))

    def test_overlapping_labels_use_deterministic_highest_score(self):
        model = Mock()
        model.inference.side_effect = [
            [[{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'first_name', 'score': .7}]],
            [[{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'last_name', 'score': .9}]],
            [[]],
        ]
        with patch('nvidia_ner_worker.windows', return_value=[(0, 'Mira')]):
            entities, _ = predict(model, 'Mira')
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['label'], 'last_name')

    def test_model_span_with_wrong_evidence_is_rejected(self):
        model = Mock()
        model.inference.side_effect = [
            [[{'start': 0, 'end': 4, 'text': 'Mira', 'label': 'first_name', 'score': .9}]],
            [[]],
            [[{'start': 0, 'end': 4, 'text': 'Lira', 'label': 'first_name', 'score': .9}]],
        ]
        with patch('nvidia_ner_worker.windows', return_value=[(0, 'Mira')]):
            with self.assertRaises(ValueError):
                predict(model, 'Mira')


if __name__ == '__main__':
    unittest.main()