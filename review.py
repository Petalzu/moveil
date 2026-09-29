"""Loopback-only image review and live local NVIDIA inference."""
import argparse
import hashlib
import io
import json
import re
import secrets
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw
from live_batch import LiveBatch, MAX_UPLOAD_BYTES, UPLOAD_TYPES

ROOT = Path(__file__).resolve().parent


def allowed_path(root, value, area):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("Invalid path")
    parts = value.split("/")
    if not parts or parts[0] != area or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Invalid path")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to((root / area).resolve()) or any(p.is_symlink() or p.is_junction() if hasattr(p, "is_junction") else p.is_symlink() for p in [path, *path.parents] if p != root):
        raise ValueError("Invalid path")
    return path


def clean_png(path):
    with Image.open(path) as source:
        result = Image.new("RGB", source.size)
        result.paste(source.convert("RGB"))
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


class Review:
    def __init__(self, root, run, source):
        self.root, self.source_name = root, source
        self.source = allowed_path(root, source, "inputs")
        self.load(run)

    def load(self, name):
        directory = allowed_path(self.root, name, "runs")
        audit_path = allowed_path(self.root, name + "/audit.json", "runs")
        output = allowed_path(self.root, name + "/redacted.png", "runs")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["status"] not in {"completed_unassessed", "no_detections_unverified", "pending_human_review"}:
            raise ValueError("Failed run")
        for path, key in [(self.source, "input_sha256"), (output, "output_sha256")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != audit[key]:
                raise ValueError("Hash mismatch")
        with Image.open(self.source) as original, Image.open(output) as image:
            if original.size != image.size:
                raise ValueError("Size mismatch")
            self.size = image.size
        self.name, self.directory, self.output, self.audit = name, directory, output, audit

    def state(self):
        metric_path = self.directory / "metrics.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.is_file() else None
        return {"run": self.name, "width": self.size[0], "height": self.size[1],
            "model": {key: self.audit.get('worker', {}).get(key) for key in
                      ('actual_device', 'requested_device', 'gpu_name', 'versions')},
            "evaluation_status": "manual" if self.audit.get("parent_run") else "evaluated" if metrics is not None else "unannotated",
            "cohort_metrics": None,
            "entity_matches": [] if metrics is None else metrics.get('entities', []),
            "status": "待人工复核 · 未批准", "detections": self.audit["detections"],
            "automatic_metrics": None if metrics is None else {k: metrics[k] for k in ["entity_count", "fully_covered_entities", "nonsensitive_ink_redacted_fraction"]}}

    def save(self, boxes):
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= 128:
            raise ValueError("Invalid rectangles")
        w, h = self.size
        for b in boxes:
            if not isinstance(b, list) or len(b) != 4 or any(type(n) is not int for n in b):
                raise ValueError("Invalid rectangle")
            if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                raise ValueError("Invalid rectangle")
        # Recheck disk hashes immediately before deriving a version.
        self.load(self.name)
        with Image.open(self.output) as old:
            image = Image.new("RGB", old.size)
            image.paste(old.convert("RGB"))
        draw = ImageDraw.Draw(image)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill="black")
        name = "runs/review-" + secrets.token_hex(8)
        directory = allowed_path(self.root, name, "runs")
        directory.mkdir(exist_ok=False)
        try:
            output = directory / "redacted.png"
            image.save(output, format="PNG")
            with Image.open(output) as check, Image.open(self.output) as old:
                if check.info or check.mode != "RGB":
                    raise ValueError("Metadata verification failed")
                # Every formerly black pixel must remain black, including prior manual masks.
                if any(a == (0, 0, 0) and b != a for a, b in zip(old.convert("RGB").get_flattened_data(), check.get_flattened_data())):
                    raise ValueError("Mask verification failed")
                for b in boxes:
                    if check.crop(b).getextrema() != ((0, 0),) * 3:
                        raise ValueError("Mask verification failed")
            audit = {"status": "pending_human_review", "approved": False, "parent_run": self.name,
                     "input_sha256": self.audit["input_sha256"], "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                     "detections": self.audit["detections"] + [{"label": "manual", "boxes": [b]} for b in boxes]}
            (directory / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(directory)
            raise
        self.load(name)
        return self.state()


def make_server(review, port=18196):
    token = secrets.token_urlsafe(32)
    batch = LiveBatch(review.root, Review)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log URLs, request bodies, OCR text, or tokens.

        def respond(self, status, content, kind="application/json", download=False):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="redacted-unapproved.png"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def trusted(self):
            host = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == host
                    and self.headers.get("Origin") in {None, "http://" + host}
                    and self.headers.get("Sec-Fetch-Site") in {None, "none", "same-origin"})

        def do_GET(self):
            if not self.trusted():
                return self.respond(403, b'{"error":"Forbidden"}')
            route = urlsplit(self.path)
            if route.query:
                return self.respond(400, b'{"error":"No query paths"}')
            try:
                if route.path == "/api/state":
                    return self.respond(200, json.dumps({**review.state(), "token": token}).encode())
                if route.path == '/api/batch':
                    return self.respond(200, json.dumps(batch.snapshot()).encode())
                match = re.fullmatch(r'/api/items/(live-[a-f0-9]{16}-[0-9]+)/(state|original.png|redacted.png|download)', route.path)
                if match:
                    item = batch.get_review(match[1])
                    if match[2] == 'state':
                        return self.respond(200, json.dumps(item.state()).encode())
                    return self.respond(200, clean_png(item.source if match[2] == 'original.png' else item.output), 'image/png', match[2] == 'download')
                if route.path in {"/original.png", "/redacted.png", "/download"}:
                    return self.respond(200, clean_png(review.source if route.path == "/original.png" else review.output), "image/png", route.path == "/download")
                assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
                if route.path not in assets:
                    return self.respond(404, b'{"error":"Not found"}')
                file, kind = assets[route.path]
                self.respond(200, (ROOT / "web" / file).read_bytes(), kind)
            except Exception:
                self.respond(400, b'{"error":"Read failed"}')

        def do_POST(self):
            if not self.trusted() or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}" or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), token):
                return self.respond(403, b'{"error":"Forbidden"}')
            if self.path not in {"/api/save", "/api/batch", "/api/upload"}:
                return self.respond(404, b'{"error":"Not found"}')
            if self.path == '/api/upload':
                try:
                    lengths = self.headers.get_all('Content-Length', [])
                    if self.headers.get('Transfer-Encoding') is not None or len(lengths) != 1 or not re.fullmatch(r'[0-9]+', lengths[0]):
                        return self.respond(400, b'{"error":"Invalid content length"}')
                    size = int(lengths[0])
                    if size > MAX_UPLOAD_BYTES:
                        return self.respond(413, b'{"error":"Image too large"}')
                    kind = self.headers.get('Content-Type')
                    if kind not in UPLOAD_TYPES:
                        return self.respond(415, b'{"error":"Unsupported image type"}')
                    if size == 0:
                        return self.respond(400, b'{"error":"Empty image"}')
                    payload = self.rfile.read(size)
                    if len(payload) != size:
                        raise ValueError('Incomplete image')
                    result = batch.start_upload(payload, kind)
                    return self.respond(202, json.dumps(result).encode())
                except RuntimeError:
                    return self.respond(409, b'{"error":"A job is already running"}')
                except Exception:
                    return self.respond(400, b'{"error":"Upload rejected"}')
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Invalid request")
                data = json.loads(self.rfile.read(size))
                if self.path == '/api/batch':
                    if set(data) != {'count'}:
                        raise ValueError('Invalid request')
                    return self.respond(202, json.dumps(batch.start(data['count'])).encode())
                item = batch.get_review(data['item']) if 'item' in data else review
                if set(data) not in ({"boxes", "run"}, {"boxes", "run", "item"}) or data["run"] != item.name:
                    raise ValueError("Stale version")
                result = item.save(data["boxes"])
                self.respond(200, json.dumps(result).encode())
            except Exception:
                self.respond(400, b'{"error":"Save rejected; refresh and retry"}')

    return HTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Project-relative runs/name")
    parser.add_argument("--input", default="inputs/sample.png")
    parser.add_argument("--port", type=int, default=18196)
    args = parser.parse_args()
    server = make_server(Review(ROOT, args.run, args.input), args.port)
    print(f"Review: http://127.0.0.1:{server.server_port} (unapproved)", flush=True)
    server.serve_forever()