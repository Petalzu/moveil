"""Portable geometry regressions; historical Paddle evidence is archived."""
import unittest
from hybrid_geometry import align_geometry, require_geometry


def line(text, x=0):
    return {'text': text, 'confidence': .99, 'units': [
        {'start': 0, 'end': len(text), 'polygon': [[x,0],[x+20,0],[x+20,20],[x,20]]}]}


class HybridGeometryTests(unittest.TestCase):
    def test_ctc_activation_is_not_glyph_extent(self):
        native = line('Ethan Connelly', 100)
        words = [line('Ethan', 40), line('Connelly', 80)]
        result = align_geometry([native], words)
        self.assertEqual(result[0]['text'], native['text'])
        self.assertEqual(result[0]['units'][0]['polygon'], words[0]['units'][0]['polygon'])
        self.assertEqual([(u['start'], u['end']) for u in result[0]['units']], [(0,5),(6,14)])

    def test_repeat_and_wrap(self):
        result = align_geometry([line('Test Test'), line('test')],
                                [line('Test', 0), line('Test', 30), line('test', 60)])
        self.assertEqual(result[0]['units'][1]['polygon'][0][0], 30)
        self.assertEqual(result[1]['units'][0]['polygon'][0][0], 60)

    def test_substitution_does_not_borrow_neighbors(self):
        result = align_geometry([line('code O123 end')],
                                [line('code'), line('0123', 30), line('end', 60)])
        with self.assertRaises(ValueError):
            require_geometry(result, [{'line':0,'start':5,'end':9}])
        require_geometry(result, [{'line':0,'start':10,'end':13}])

    def test_partial_overlap_cannot_hide_missing_geometry(self):
        result = align_geometry([line('Alpha Beta')], [line('Alpha'), line('Zeta')])
        with self.assertRaises(ValueError):
            require_geometry(result, [{'line':0,'start':0,'end':10}])

    def test_whitespace_only_normalization(self):
        result = align_geometry([line('a @ b')], [line('a@b')])
        require_geometry(result, [{'line':0,'start':0,'end':5}])
        self.assertEqual(result[0]['text'], 'a @ b')