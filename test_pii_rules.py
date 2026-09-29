import unittest
from pii_rules import chunks, recognize, luhn


class RuleTests(unittest.TestCase):
    def test_formats_and_context(self):
        text = 'Born on 1991-12-09. Event 2026-11-03T12:45:30. Email a.b@example.org. IP 192.0.2.1. Phone +44 20 7946 0958.'
        entities = recognize(text)
        for label, value in [('date_of_birth', '1991-12-09'), ('date_time', '2026-11-03T12:45:30'), ('email', 'a.b@example.org'), ('ip_address', '192.0.2.1'), ('phone_number', '+44 20 7946 0958')]:
            self.assertIn({'label': label, 'text': value}, entities)

    def test_reject_invalid_formats(self):
        self.assertEqual(recognize('2026-99-35 999.2.3.4 1111111111111111'), [])
        self.assertTrue(luhn('4111 1111 1111 1111'))
        self.assertFalse(luhn('4111 1111 1111 1112'))

    def test_chunks_bounded_overlap_and_coverage(self):
        lines = [{'text': str(i) * 10} for i in range(100)]
        parts = chunks(lines, 150, 40)
        self.assertTrue(all(len(' '.join(x['text'] for x in part)) <= 150 for part in parts))
        self.assertTrue(all(any(line in part for part in parts) for line in lines))
        self.assertTrue(all(any(x in b for x in a) for a, b in zip(parts, parts[1:])))
        with self.assertRaises(ValueError):
            chunks([{'text': 'x' * 1000}])


if __name__ == '__main__':
    unittest.main()