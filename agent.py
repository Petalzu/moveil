"""Image-only agent. No fixture/ground-truth imports, arbitrary tools or paths."""
import math
from functools import lru_cache

from PIL import Image, ImageDraw

# Broad static privacy policy, not derived from the test record's labels.
LABELS = set("first_name last_name full_name date_of_birth age gender race_ethnicity sexuality religion language nationality country city state county street_address postal_code phone_number email url ip_address mac_address username password company_name occupation employment_status salary income date time ssn passport_number national_id driver_license medical_record_number blood_type medical_condition medication health_insurance_number bank_account_number bank_routing_number swift_bic iban credit_debit_card credit_card_pin vehicle_identifier license_plate biometric other_pii".split())
LABELS.update("education_level religious_belief biometric_identifier tax_id date_time pin health_plan_beneficiary_number".split())


def validate(action, phase, lines):
    if not isinstance(action, dict):
        raise ValueError("Object required")
    expected = ("ocr", "redact", "finish")[phase]
    if action.get("action") != expected:
        raise ValueError("Invalid action for state")
    if expected != "redact":
        if set(action) != {"action"}:
            raise ValueError("Unexpected fields")
        return
    if set(action) != {"action", "entities"} or not isinstance(action["entities"], list):
        raise ValueError("Invalid entities")
    if len(action["entities"]) > 128:
        raise ValueError("Too many entities")
    for entity in action["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"label", "line", "start", "end"}:
            raise ValueError("Invalid entity fields")
        if not isinstance(entity["label"], str) or entity["label"] not in LABELS:
            raise ValueError("Unknown label")
        if any(type(entity[k]) is not int for k in ("line", "start", "end")):
            raise ValueError("Integer indices required")
        i, start, end = entity["line"], entity["start"], entity["end"]
        if not 0 <= i < len(lines) or not 0 <= start < end <= len(lines[i]["text"]):
            raise ValueError("Out of bounds")


def redact(image_path, output, entities, lines):
    with Image.open(image_path) as original:
        original.load()
        image = Image.new("RGB", original.size)
        image.paste(original.convert("RGB"))
    draw = ImageDraw.Draw(image)
    detections = []
    for entity in entities:
        boxes = []
        for unit in lines[entity["line"]]["units"]:
            if unit["start"] < entity["end"] and unit["end"] > entity["start"]:
                xs, ys = zip(*unit["polygon"])
                box = [max(0, math.floor(min(xs)) - 4), max(0, math.floor(min(ys)) - 4),
                       min(image.width, math.ceil(max(xs)) + 5), min(image.height, math.ceil(max(ys)) + 5)]
                draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=(0, 0, 0))
                boxes.append(box)
        if not boxes:
            raise ValueError("Entity has no OCR geometry")
        detections.append({**entity, "boxes": boxes})
    image.save(output, format="PNG")
    with Image.open(output) as check:
        if check.info or check.mode != "RGB":
            raise ValueError("Unexpected output metadata")
        for det in detections:
            for box in det["boxes"]:
                if check.crop(box).getextrema() != ((0, 0), (0, 0), (0, 0)):
                    raise ValueError("Pixel verification failed")
    return detections


@lru_cache(maxsize=1)
def projection_engine():
    """Reuse CPU ONNX sessions across the serial image batch."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def projection_ocr(image, engine=None, *, recognize_batch=None):
    """Whitespace word boxes for clean horizontal dark-on-light documents only."""
    import numpy as np
    with Image.open(image) as source:
        pixels = np.array(source.convert("RGB"))
    ink = np.any(pixels < 180, axis=2)
    ys = np.where(ink.any(axis=1))[0]
    if not len(ys) or ink.mean() > 0.25:
        raise ValueError("Unsupported layout")
    rows = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
    crops, polygons = [], []
    for row in rows:
        y0, y1 = int(row[0]), int(row[-1]) + 1
        if y1-y0 < 10:
            raise ValueError("Unsupported layout")
        xs = np.where(ink[y0:y1].any(axis=0))[0]
        words = np.split(xs, np.where(np.diff(xs) > (y1-y0)*0.45)[0] + 1)
        for word in words:
            x0, x1 = int(word[0]), int(word[-1]) + 1
            crop = pixels[max(0,y0-8):y1+8, max(0,x0-8):x1+8]
            crops.append(crop)
            polygons.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            if len(crops) > 512:
                raise ValueError("Document exceeds validated scope")
    if recognize_batch is None:
        engine = engine if engine is not None else projection_engine()
        results = []
        for crop in crops:
            result, _ = engine(crop, use_det=False, use_cls=False)
            if not result:
                raise ValueError("OCR returned no text")
            results.append(result[0])
    else:
        results = list(recognize_batch(crops))
    if len(results) != len(polygons):
        raise ValueError("OCR result count mismatch")
    lines = []
    for (text, score), polygon in zip(results, polygons, strict=True):
        if not isinstance(text, str) or not text.strip() or len(text) > 6000:
            raise ValueError("OCR returned invalid text")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("OCR returned invalid confidence")
        text = text.strip()
        lines.append({"text": text, "confidence": score, "units": [
            {"start": 0, "end": len(text), "polygon": polygon}]})
    return lines
