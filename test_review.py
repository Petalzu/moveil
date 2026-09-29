import hashlib
import http.client
import io
import json
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from PIL import Image, ImageDraw, PngImagePlugin

from review import Review, allowed_path, clean_png, make_server
from live_batch import LiveBatch, MAX_UPLOAD_BYTES, normalize_upload


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "inputs").mkdir()
        directory = self.root / "runs/base"
        directory.mkdir(parents=True)
        source = self.root / "inputs/sample.png"
        image = Image.new("RGB", (40, 40), "white")
        info = PngImagePlugin.PngInfo()
        info.add_text("secret", "not exported")
        image.save(source, pnginfo=info)
        ImageDraw.Draw(image).rectangle((2, 2, 9, 9), fill="black")
        output = directory / "redacted.png"
        image.save(output)
        self.audit = {"status": "completed_unassessed", "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "detections": [{"label": "first_name", "boxes": [[2, 2, 10, 10]]}]}
        (directory / "audit.json").write_text(json.dumps(self.audit))
        self.review = Review(self.root, "runs/base", "inputs/sample.png")

    def test_paths_reject_traversal_absolute_and_private(self):
        for path in ["runs/../private", "runs/%2e%2e/private", "runs\\base", "C:/runs/base", "/runs/base", "private/truth.json", "runs//base", "runs/./base"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                allowed_path(self.root, path, "runs")

    def test_save_preserves_old_masks_and_source(self):
        original = (self.root / "inputs/sample.png").read_bytes()
        old = (self.root / "runs/base/redacted.png").read_bytes()
        first = self.review.save([[15, 15, 20, 20]])
        second = self.review.save([[25, 25, 30, 30]])
        self.assertNotEqual(first["run"], second["run"])
        self.assertEqual(original, (self.root / "inputs/sample.png").read_bytes())
        self.assertEqual(old, (self.root / "runs/base/redacted.png").read_bytes())
        with Image.open(self.review.output) as image:
            self.assertFalse(image.info)
            for box in [[2, 2, 10, 10], [15, 15, 20, 20], [25, 25, 30, 30]]:
                self.assertEqual(image.crop(box).getextrema(), ((0, 0),) * 3)
        self.assertFalse(self.review.audit["approved"])

    def test_bad_rectangles_write_nothing(self):
        for boxes in [[], [[-1, 0, 5, 5]], [[0, 0, 41, 20]], [[True, 0, 5, 5]], [[2, 2, 2, 3]], [[0.5, 0, 3, 3]], "bad"]:
            with self.assertRaises(ValueError):
                self.review.save(boxes)
        self.assertEqual(len(list((self.root / "runs").iterdir())), 1)

    def test_save_failure_cleans_new_version(self):
        with patch("review.Image.Image.save", side_effect=OSError("SECRET")):
            with self.assertRaises(OSError):
                self.review.save([[20, 20, 25, 25]])
        self.assertEqual(len(list((self.root / "runs").iterdir())), 1)
        self.assertEqual(self.review.name, "runs/base")

    def test_reject_failed_or_tampered_run(self):
        path = self.root / "runs/base/audit.json"
        for delta in [{"status": "failed"}, {"output_sha256": "wrong"}, {"input_sha256": "wrong"}]:
            path.write_text(json.dumps({**self.audit, **delta}))
            with self.assertRaises(ValueError):
                self.review.load("runs/base")

    def test_export_no_metadata(self):
        import io
        with Image.open(io.BytesIO(clean_png(self.review.source))) as image:
            self.assertFalse(image.info)
            self.assertEqual(image.mode, "RGB")

    def test_live_batch_validation_and_snapshot(self):
        batch = LiveBatch(self.root, Review)
        self.assertEqual(batch.snapshot()['status'], 'idle')
        for count in [0, 11, True, 1.5, '2', None]:
            with self.assertRaises(ValueError):
                batch.start(count)
        with patch('live_batch.threading.Thread') as thread:
            first = batch.start(2)
            thread.return_value.start.assert_called_once()
            self.assertEqual(first['total'], 2)
            self.assertEqual(first['status'], 'running')
            with self.assertRaises(RuntimeError):
                batch.start(1)
            first['items'].append({'id': 'fake'})
            self.assertEqual(batch.snapshot()['items'], [])
        with self.assertRaises(KeyError):
            batch.get_review('../private')

    def test_metrics_do_not_load_historical_summary(self):
        self.assertIsNone(self.review.state()['cohort_metrics'])
        self.assertEqual(self.review.state()['entity_matches'], [])

    def test_state_exposes_only_safe_model_metadata(self):
        self.review.audit['worker'] = {'actual_device': 'cuda:0', 'gpu_name': 'NVIDIA GB10',
                                      'model_files': {'private/path': 'hash'}}
        model = self.review.state()['model']
        self.assertEqual(model['actual_device'], 'cuda:0')
        self.assertNotIn('model_files', model)

    @staticmethod
    def image_bytes(image=None, format='PNG', **kwargs):
        buffer = io.BytesIO()
        (image if image is not None else Image.new('RGB', (8, 6), 'white')).save(buffer, format=format, **kwargs)
        return buffer.getvalue()

    def fake_pipeline(self):
        def run(source, out, worker, **kwargs):
            out.mkdir(parents=True)
            shutil.copyfile(source, out / 'redacted.png')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (out / 'audit.json').write_text(json.dumps({
                'status': 'no_detections_unverified', 'input_sha256': digest,
                'output_sha256': digest, 'detections': []}))
            return True
        return types.SimpleNamespace(NvidiaNerWorker=MagicMock(), run=MagicMock(side_effect=run),
                                     CONFIG={'threshold': 0.5}, CONFIG_HASH='test')

    def test_upload_normalization(self):
        info = PngImagePlugin.PngInfo()
        info.add_text('secret', 'remove me')
        payload = self.image_bytes(Image.new('RGBA', (8, 6), (255, 0, 0, 0)), pnginfo=info)
        with Image.open(io.BytesIO(normalize_upload(payload, 'image/png'))) as image:
            self.assertEqual(image.mode, 'RGB')
            self.assertFalse(image.info)
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = 'private author'
        payload = self.image_bytes(format='JPEG', exif=exif)
        with Image.open(io.BytesIO(normalize_upload(payload, 'image/jpeg'))) as image:
            self.assertEqual(image.size, (6, 8))
            self.assertFalse(image.info)
            self.assertFalse(image.getexif())
        with Image.open(io.BytesIO(normalize_upload(self.image_bytes(format='WEBP'), 'image/webp'))) as image:
            self.assertEqual(image.mode, 'RGB')

    def test_upload_invalid_images_leave_no_files(self):
        batch = LiveBatch(self.root, Review)
        animated = self.image_bytes(save_all=True, append_images=[Image.new('RGB', (8, 6), 'black')], duration=100)
        invalid = [(b'', 'image/png'), (b'not an image', 'image/png'),
                   (b'x' * (MAX_UPLOAD_BYTES + 1), 'image/png'),
                   (self.image_bytes(), 'image/jpeg'), (self.image_bytes(format='GIF'), 'image/png'),
                   (animated, 'image/png'), (self.image_bytes()[:40], 'image/png'),
                   (self.image_bytes(Image.new('RGB', (8193, 1))), 'image/png'),
                   (self.image_bytes(Image.new('RGB', (4001, 4000))), 'image/png')]
        with patch('live_batch.threading.Thread') as thread:
            for payload, kind in invalid:
                with self.subTest(kind=kind, size=len(payload)), self.assertRaises(Exception):
                    batch.start_upload(payload, kind)
                self.assertEqual(list((self.root / 'inputs').iterdir()), [self.review.source])
                self.assertEqual(batch.snapshot()['status'], 'idle')
            thread.assert_not_called()

    def test_upload_concurrency_and_start_failure_cleanup(self):
        batch = LiveBatch(self.root, Review)
        with patch('live_batch.threading.Thread') as thread:
            thread.return_value.start.side_effect = RuntimeError('start failed')
            with self.assertRaises(RuntimeError):
                batch.start_upload(self.image_bytes())
        self.assertEqual(batch.snapshot()['status'], 'idle')
        self.assertEqual(list((self.root / 'inputs').iterdir()), [self.review.source])
        with patch('live_batch.threading.Thread'):
            first = batch.start_upload(self.image_bytes())
            self.assertEqual(first['mode'], 'upload')
            before = set((self.root / 'inputs').rglob('*'))
            with patch('live_batch.normalize_upload') as normalize:
                with self.assertRaises(RuntimeError):
                    batch.start_upload(b'invalid')
                normalize.assert_not_called()
            with self.assertRaises(RuntimeError):
                batch.start(1)
            self.assertEqual(before, set((self.root / 'inputs').rglob('*')))
        other = LiveBatch(self.root, Review)
        with patch('live_batch.threading.Thread'):
            self.assertEqual(other.start(1)['mode'], 'samples')
            with self.assertRaises(RuntimeError):
                other.start_upload(self.image_bytes())

    def test_upload_pipeline_persistence_and_no_metrics(self):
        batch = LiveBatch(self.root, Review)
        with patch('live_batch.threading.Thread') as thread:
            job = batch.start_upload(self.image_bytes())
            args = thread.call_args.kwargs['args']
        pipeline = self.fake_pipeline()
        pipeline.NvidiaNerWorker.return_value.__enter__.return_value.metadata = {
            'actual_device': 'cuda:0', 'requested_device': 'cuda'}
        with patch.dict(sys.modules, {'nvidia_ner_pipeline': pipeline}):
            batch._run(*args)
        pipeline.run.assert_called_once()
        pipeline.NvidiaNerWorker.assert_called_once()
        state = batch.snapshot()
        self.assertEqual(state['model']['actual_device'], 'cuda:0')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['completed'], 1)
        self.assertEqual(state['failed'], 0)
        self.assertIsNone(state['metrics'])
        item = state['items'][0]
        self.assertEqual(item['display_name'], '上传图片')
        self.assertEqual(item['offset'], 1)
        self.assertIs(type(item['offset']), int)
        self.assertEqual(item['detection_count'], 0)
        self.assertIsNone(item['full'])
        self.assertIsNone(item['entities'])
        review = batch.get_review(item['id'])
        restored = Review(self.root, review.name, review.source_name)
        self.assertEqual(restored.state()['evaluation_status'], 'unannotated')
        self.assertIsNone(restored.state()['automatic_metrics'])
        self.assertFalse(list((self.root / 'runs' / job['id']).rglob('metrics.json')))
        self.assertFalse((self.root / 'runs' / job['id'] / 'summary.json').exists())
        self.assertEqual(restored.save([[0, 0, 2, 2]])['evaluation_status'], 'manual')

    def test_evaluation_status(self):
        self.assertEqual(self.review.state()['evaluation_status'], 'unannotated')
        (self.review.directory / 'metrics.json').write_text(json.dumps({
            'entities': [], 'entity_count': 0, 'fully_covered_entities': 0,
            'nonsensitive_ink_redacted_fraction': 0}))
        self.assertEqual(self.review.state()['evaluation_status'], 'evaluated')
        state = self.review.save([[20, 20, 25, 25]])
        self.assertEqual(state['evaluation_status'], 'manual')
        self.assertIsNone(state['automatic_metrics'])

    def test_http_upload_validation_and_item_routes(self):
        server = make_server(self.review, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        token = session.get(base + '/api/state').json()['token']
        headers = {'Origin': base, 'X-Review-Token': token, 'Content-Type': 'image/png'}

        def header_only(changes, expected):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            try:
                connection.request('POST', '/api/upload', headers={**headers, 'Content-Length': '100', **changes})
                response = connection.getresponse()
                self.assertEqual(response.status, expected)
                response.read()
            finally:
                connection.close()

        # No body sent: these responses prove rejection happens before reading it.
        header_only({'Origin': 'https://evil.example'}, 403)
        header_only({'X-Review-Token': 'wrong'}, 403)
        header_only({'Sec-Fetch-Site': 'cross-site'}, 403)
        header_only({'Host': 'evil.example'}, 403)
        header_only({'Content-Length': str(MAX_UPLOAD_BYTES + 1)}, 413)
        header_only({'Content-Type': 'image/gif'}, 415)
        header_only({'Content-Length': '0'}, 400)
        header_only({'Content-Length': '-1'}, 400)
        header_only({'Transfer-Encoding': 'chunked'}, 400)
        self.assertEqual(session.post(base + '/api/upload', data=b'invalid', headers=headers).status_code, 400)
        self.assertEqual(session.post(base + '/api/upload', data=self.image_bytes()).status_code, 403)
        self.assertEqual(list((self.root / 'inputs').iterdir()), [self.review.source])
        with patch('live_batch.threading.Thread') as thread:
            response = session.post(base + '/api/upload', data=self.image_bytes(), headers=headers)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()['mode'], 'upload')
            call = thread.call_args.kwargs
            self.assertEqual(session.post(base + '/api/upload', data=self.image_bytes(), headers=headers).status_code, 409)
        with patch.dict(sys.modules, {'nvidia_ner_pipeline': self.fake_pipeline()}):
            call['target'](*call['args'])
        item = session.get(base + '/api/batch').json()['items'][0]
        prefix = base + '/api/items/' + item['id']
        state = session.get(prefix + '/state').json()
        self.assertEqual(state['evaluation_status'], 'unannotated')
        self.assertIsNone(state['automatic_metrics'])
        for route in ['/original.png', '/redacted.png', '/download']:
            self.assertEqual(session.get(prefix + route).status_code, 200)
        saved = session.post(base + '/api/save', json={'item': item['id'], 'run': state['run'],
                             'boxes': [[0, 0, 2, 2]]}, headers={**headers, 'Content-Type': 'application/json'})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()['evaluation_status'], 'manual')

    def test_http_origin_token_path_and_save(self):
        server = make_server(self.review, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            session = requests.Session()
            session.trust_env = False
            with session:
                state = session.get(base + "/api/state").json()
                data = {"run": "runs/base", "boxes": [[20, 20, 25, 25]]}
                for headers in [{}, {"Origin": "https://evil.example", "X-Review-Token": state["token"]}, {"Origin": base, "X-Review-Token": "wrong"}]:
                    self.assertEqual(session.post(base + "/api/save", json=data, headers=headers).status_code, 403)
                self.assertEqual(session.get(base + "/private/truth.json").status_code, 404)
                self.assertEqual(session.get(base + "/api/state", headers={"Host": "evil.example"}).status_code, 403)
                self.assertEqual(session.get(base + "/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
                headers = {"Origin": base, "X-Review-Token": state["token"]}
                self.assertEqual(session.get(base + '/api/batch').json()['status'], 'idle')
                self.assertEqual(session.post(base + '/api/batch', json={'count': 2}).status_code, 403)
                for count in [0, 11, True, '2']:
                    self.assertEqual(session.post(base + '/api/batch', json={'count': count}, headers=headers).status_code, 400)
                with patch('live_batch.threading.Thread'):
                    launched = session.post(base + '/api/batch', json={'count': 2}, headers=headers)
                    self.assertEqual(launched.status_code, 202)
                    self.assertEqual(launched.json()['total'], 2)
                    self.assertEqual(session.post(base + '/api/batch', json={'count': 1}, headers=headers).status_code, 400)
                saved = session.post(base + "/api/save", json=data, headers=headers)
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(session.post(base + "/api/save", json=data, headers=headers).status_code, 400)
                downloaded = session.get(base + "/download")
                self.assertEqual(downloaded.content, clean_png(self.review.output))
                self.assertIn("attachment", downloaded.headers["Content-Disposition"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()