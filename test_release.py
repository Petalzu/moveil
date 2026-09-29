"""Release entry points, without model download or external inference."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_model
import run_demo


class ReleaseTests(unittest.TestCase):
    def test_custom_model_path_is_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, MOVEIL_MODEL_DIR=directory)
            result = subprocess.run([sys.executable, '-c',
                'import prepare_model, nvidia_ner_worker; '
                'assert prepare_model.MODEL_DIR == nvidia_ner_worker.MODEL_DIR; '
                'print(prepare_model.MODEL_DIR)'], env=env, check=True,
                capture_output=True, text=True)
            self.assertEqual(Path(result.stdout.strip()), Path(directory))

    def test_weight_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pytorch_model.bin').write_bytes(b'test weight')
            with self.assertRaises(ValueError):
                prepare_model.verify_weight(root)
            with patch.object(prepare_model, 'WEIGHT_SHA256', hashlib.sha256(b'test weight').hexdigest()):
                prepare_model.verify_weight(root)

    def test_bootstrap_uses_real_pipeline(self):
        with patch.object(run_demo.subprocess, 'run') as execute:
            name = run_demo.bootstrap()
        self.assertTrue(name.startswith('runs/bootstrap-'))
        args = execute.call_args
        self.assertTrue(args.kwargs['check'])
        self.assertEqual(args.kwargs['cwd'], run_demo.ROOT)
        self.assertIn(str(run_demo.ROOT / 'nvidia_ner_pipeline.py'), args.args[0])
        self.assertIn(str(run_demo.ROOT / 'inputs/sample.png'), args.args[0])

    def test_failed_bootstrap_stops(self):
        with patch.object(run_demo.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'inference')):
            with self.assertRaises(subprocess.CalledProcessError):
                run_demo.bootstrap()


if __name__ == '__main__':
    unittest.main()