"""Conservative OCR-only format recognizers; no dataset or renderer imports."""
import ipaddress
import re
from datetime import datetime


def chunks(lines, limit=900, overlap=160):
    """Whole OCR words, bounded overlapping windows; geometry stays unchanged."""
    result, start = [], 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and size + len(lines[end]['text']) + 1 <= limit:
            size += len(lines[end]['text']) + 1
            end += 1
        if end == start:
            raise ValueError('OCR token exceeds chunk limit')
        result.append(lines[start:end])
        if end == len(lines):
            break
        next_start, size = end, 0
        while next_start > start + 1 and size < overlap:
            next_start -= 1
            size += len(lines[next_start]['text']) + 1
        start = next_start
    return result


def luhn(text):
    digits = [int(c) for c in text if c.isdecimal() and c.isascii()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) < 2:
        return False
    return sum((d * 2 - 9 if d > 4 else d * 2) if i % 2 else d
               for i, d in enumerate(reversed(digits))) % 10 == 0


def recognize(document):
    found = []
    def add(label, match):
        item = {'label': label, 'text': match.group().rstrip('.,;')}
        if item not in found:
            found.append(item)
    for m in re.finditer(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+', document):
        add('email', m)
    for m in re.finditer(r'\bhttps?://[^\s<>]+', document):
        add('url', m)
    for m in re.finditer(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)', document):
        try:
            ipaddress.ip_address(m.group())
            add('ip_address', m)
        except ValueError:
            pass
    for m in re.finditer(r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b', document):
        try:
            datetime.fromisoformat(m.group())
        except ValueError:
            continue
        context = document[max(0, m.start()-45):m.start()].lower()
        label = 'date_time' if 'T' in m.group() else ('date_of_birth' if re.search(r'\bborn\b|\bbirth\b|\bdob\b', context) else 'date')
        add(label, m)
    for m in re.finditer(r'(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])', document):
        if luhn(m.group()):
            add('credit_debit_card', m)
    for m in re.finditer(r'(?<![\w-])\+?\d(?:[ ()-]*\d){6,14}(?![\w-])', document):
        context = document[max(0, m.start()-40):m.start()].lower()
        if re.search(r'\b(?:phone|tel|telephone|mobile|contact)\b', context) and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', m.group()):
            add('phone_number', m)
    return found