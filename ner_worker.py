"""Shared tokenizer-bounded text windows for NVIDIA GLiNER-PII inference."""


def windows(model, text, config):
    words = list(model.data_processor.words_splitter(text))
    result, start = [], 0
    while start < len(words):
        end = min(start + config['window_words'], len(words))
        while end > start:
            part = text[words[start][1]:words[end-1][2]]
            count = len(model.data_processor.transformer_tokenizer(
                part, add_special_tokens=True, truncation=False)['input_ids'])
            if count <= config['max_subtokens']:
                break
            end -= 1
        if end <= start:
            raise ValueError('Oversized token')
        result.append((words[start][1], part))
        if end == len(words):
            break
        start = max(start + 1, end - config['overlap_words'])
    return result
