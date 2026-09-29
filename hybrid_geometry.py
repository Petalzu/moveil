"""Paddle text + Rapid image-projection geometry. No truth or native-box snapping."""
import copy
from difflib import SequenceMatcher


def align_geometry(paddle_lines, rapid_words):
    result = copy.deepcopy(paddle_lines)
    a, positions = [], []
    for i, line in enumerate(result):
        line['units'] = []
        line['geometry_unmapped'] = []
        for j, char in enumerate(line['text']):
            if not char.isspace():
                a.append(char)
                positions.append((i, j))
    b, owners = [], []
    for i, word in enumerate(rapid_words):
        for char in word['text']:
            if not char.isspace():
                b.append(char)
                owners.append(i)
    a, b = ''.join(a), ''.join(b)
    mapping = {}
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        # Entire document equality is unambiguous even with repeated words.
        # Otherwise accept only uniquely anchored exact blocks in both streams.
        text = a[block.a:block.a+block.size]
        if not text or (a != b and (a.count(text) != 1 or b.count(text) != 1)):
            continue
        for k in range(block.size):
            mapping[block.b+k] = block.a+k
    for owner, word in enumerate(rapid_words):
        indices = [i for i, value in enumerate(owners) if value == owner]
        if not indices or any(i not in mapping for i in indices):
            continue
        targets = [mapping[i] for i in indices]
        if targets != list(range(targets[0], targets[-1]+1)):
            continue
        grouped = {}
        for target in targets:
            row, offset = positions[target]
            grouped.setdefault(row, []).append(offset)
        # A projection word must not cross physical OCR lines.
        if len(grouped) != 1:
            continue
        row, offsets = next(iter(grouped.items()))
        result[row]['units'].append({'start': min(offsets), 'end': max(offsets)+1,
            'polygon': copy.deepcopy(word['units'][0]['polygon'])})
    for line in result:
        line['units'].sort(key=lambda u: u['start'])
        line['geometry_unmapped'] = [i for i,c in enumerate(line['text']) if not c.isspace()
            and not any(u['start'] <= i < u['end'] for u in line['units'])]
    return result


def require_geometry(lines, spans):
    for span in spans:
        line = lines[span['line']]
        if any(span['start'] <= i < span['end'] for i in line.get('geometry_unmapped', [])):
            raise ValueError('Unaligned entity geometry')


class HybridGeometry:
    def __init__(self, paddle):
        from rapidocr_onnxruntime import RapidOCR
        from importlib.metadata import version
        self.paddle, self.rapid = paddle, RapidOCR()
        self.metadata = dict(paddle.metadata, backend='paddle-rapid-strict-hybrid-v1',
            geometry='Rapid projection words; uniquely anchored exact character alignment; whitespace only',
            rapid_version=version('rapidocr-onnxruntime'))

    def __call__(self, image):
        from agent import projection_ocr
        return align_geometry(self.paddle(image), projection_ocr(image, self.rapid))