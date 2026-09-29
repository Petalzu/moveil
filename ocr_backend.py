"""Pinned PaddleOCR recognition models, RapidAI conversion, local Torch CUDA execution."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

MODELS = {
    'ppocrv5-torch': {
        'version': 'PP-OCRv5', 'type': 'server',
        'weights': 'ch_PP-OCRv5_rec_server.pth',
        'weights_sha256': '4767ddc90c1532ec01d881a980dae0a0b92679f4f82f88c4e9f92563de69e740',
        'dictionary': 'ppocrv5_dict.txt',
        'dictionary_sha256': 'd1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b',
    },
    'ppocrv6-torch': {
        'version': 'PP-OCRv6', 'type': 'medium',
        'weights': 'PP-OCRv6_rec_medium.pth',
        'weights_sha256': 'c2a2a7aa73892e1efc90cccd2aecc7e670ce01610490bbef7626abb3a3df52aa',
        'dictionary': 'ppocrv6_dict.txt',
        'dictionary_sha256': 'b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d',
    },
}
BACKEND = os.environ.get('MOVEIL_OCR_BACKEND', 'rapid-cpu')
BATCH_SIZE = int(os.environ.get('MOVEIL_OCR_BATCH_SIZE', '16'))
if BACKEND not in {'rapid-cpu', *MODELS} or not 1 <= BATCH_SIZE <= 64:
    raise ValueError('Invalid OCR backend or batch size (1..64)')
OCR_CONFIG = {'backend': BACKEND, 'geometry': 'original-resolution projection word boxes'}
if BACKEND in MODELS:
    OCR_CONFIG.update(MODELS[BACKEND], rapidocr='3.9.2', asset_revision='v3.9.2',
                      batch_size=BATCH_SIZE, device='cuda:0', dtype='float32')


class TorchOCR:
    """One serial owner; no automatic downloads or CPU fallback during inference."""
    def __init__(self):
        started = time.monotonic()
        if importlib.metadata.version('rapidocr') != '3.9.2':
            raise ValueError('GPU OCR requires rapidocr==3.9.2')
        model = MODELS[BACKEND]
        root = Path(os.environ.get('MOVEIL_OCR_MODEL_DIR',
                    str(Path(__file__).resolve().parent / 'models' / 'ppocr'))).resolve()
        for name, expected in ((model['weights'], model['weights_sha256']),
                               (model['dictionary'], model['dictionary_sha256'])):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError('OCR asset hash mismatch')
        import torch
        import rapidocr
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.utils.parse_parameters import ParseParams
        from rapidocr.utils.typings import EngineType, OCRVersion, ModelType
        if not torch.cuda.is_available():
            raise RuntimeError('GPU OCR requires CUDA; refusing CPU fallback')
        torch.set_num_threads(4)
        cfg = ParseParams.load(Path(rapidocr.__file__).parent / 'config.yaml')
        cfg = ParseParams.update_batch(cfg, {
            'Rec.engine_type': EngineType.TORCH,
            'Rec.ocr_version': OCRVersion(model['version']),
            'Rec.model_type': ModelType(model['type']),
            'Rec.model_path': str(root / model['weights']),
            'Rec.rec_keys_path': str(root / model['dictionary']),
            'Rec.rec_batch_num': BATCH_SIZE, 'EngineConfig.torch.use_cuda': True,
        })
        cfg.Rec.engine_cfg = cfg.EngineConfig.torch
        cfg.Rec.font_path = None
        cfg.Rec.model_root_dir = str(root)
        self.recognizer = TextRecognizer(cfg.Rec)
        parameters = tuple(self.recognizer.session.predictor.parameters())
        if not parameters or any(str(p.device) != 'cuda:0' or p.dtype != torch.float32 for p in parameters):
            raise RuntimeError('Unexpected OCR device or dtype')
        self.metadata = {**OCR_CONFIG, 'actual_device': 'cuda:0',
                         'gpu_name': torch.cuda.get_device_name(0),
                         'torch': torch.__version__, 'cuda': torch.version.cuda,
                         'load_seconds': time.monotonic() - started}

    def recognize(self, crops):
        from rapidocr.ch_ppocr_rec import TextRecInput
        # Paddle recognition expects BGR, whereas projection geometry uses RGB.
        result = self.recognizer(TextRecInput(img=[crop[:, :, ::-1].copy() for crop in crops]))
        return zip(result.txts, result.scores, strict=True)

    def __call__(self, image):
        from agent import projection_ocr
        return projection_ocr(image, recognize_batch=self.recognize)


def make_ocr_provider():
    return None if BACKEND == 'rapid-cpu' else TorchOCR()