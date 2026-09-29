"""Pinned experimental configuration for NVIDIA GLiNER-PII."""
import hashlib
import json
import os

from ner_config import LABEL_MAP

MODEL = 'nvidia/gliner-PII'
REVISION = 'bd23e8ef4425fd04e34c5204ab49ffaa706eae79'
WEIGHT_SHA256 = 'a4dfd0dcbd718acc86dca65fa99f1097d2796c8d7a681e1bc42f40f946c03802'
PROMPTS = sorted(LABEL_MAP)
LABEL_GROUP_SIZE = 24  # model config max_types=25; leave room for model markers
LABEL_GROUPS = [PROMPTS[i:i + LABEL_GROUP_SIZE]
                for i in range(0, len(PROMPTS), LABEL_GROUP_SIZE)]
CONFIG = {
    'model': MODEL,
    'revision': REVISION,
    'weight_sha256': WEIGHT_SHA256,
    'threshold': 0.3,
    'threshold_source': 'NVIDIA model card recommendation',
    'label_groups': LABEL_GROUPS,
    'label_group_size': LABEL_GROUP_SIZE,
    'model_max_types': 25,
    'window_words': 72,
    'overlap_words': 18,
    'max_subtokens': 352,
    'model_max_len': 384,
    'batch_size': 1,
    'seed': 0,
    'threads': 4,
    'device': os.environ.get('MOVEIL_DEVICE', 'cpu'),
    'label_mapping': LABEL_MAP,
    'label_support': 'prompts from existing policy; class-level support remains unverified',
    'geometry': 'same Rapid projection OCR and strict character-to-box mapping as baseline',
    'merge': 'highest-score label for identical spans; preserve partial-overlap coverage',
    'training_overlap': 'Nemotron-PII used during training; this cohort is development only',
}
if CONFIG['device'] not in {'cpu', 'cuda', 'cuda:0'}:
    raise ValueError('MOVEIL_DEVICE must be cpu, cuda or cuda:0')
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()