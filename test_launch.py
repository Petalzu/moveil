"""Launcher contracts tested without downloading or loading model weights."""
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import launch_check as launch


class LaunchTests(unittest.TestCase):
    def test_platform_paths(self):
        root = Path('/application')
        self.assertEqual(launch.environment_python(root, '.venv-ner', 'posix'), root / '.venv-ner/bin/python')
        self.assertEqual(launch.environment_python(root, '.venv-ner', 'nt'), root / '.venv-ner/Scripts/python.exe')

    def test_preflight_checks_both_environments_and_weight(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(
                launch.subprocess, 'run') as run, patch.object(launch, 'verify_weight') as verify:
            launch.check()
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
        verify.assert_called_once_with(launch.MODEL_DIR)

    def test_missing_environment(self):
        with patch.object(Path, 'is_file', return_value=False), self.assertRaisesRegex(ValueError, 'environment'):
            launch.check()

    def test_missing_model(self):
        with patch.object(Path, 'is_file', side_effect=[True, False]), patch.object(
                launch.subprocess, 'run'), self.assertRaisesRegex(ValueError, 'prepare_model'):
            launch.check()

    def test_dependency_failure_stops(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(
                launch.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'import')), patch.object(
                launch, 'verify_weight') as verify, self.assertRaises(subprocess.CalledProcessError):
            launch.check()
        verify.assert_not_called()