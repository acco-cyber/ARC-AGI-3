"""Prevent a diagnostic override from consuming a full GPU benchmark budget."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticGuardTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT/'Notebook/arc3-lease2048-diagnostic.ipynb').read_text())
        self.prefix = self.notebook['cells'][1]['source'].split('_arc_runtime =')[0]

    def test_validation_environment_cannot_expand_diagnostic_budget(self):
        scope = {}
        with patch.dict(os.environ, {'ARC40_RUN_MODE':'validation',
                                    'KAGGLE_IS_COMPETITION_RERUN':''}):
            exec(self.prefix, scope)
        self.assertEqual(scope['ARC40_RUN_MODE'], 'smoke')
        self.assertEqual(scope['ARC40_EVAL_SECONDS'], 180)

    def test_competition_marker_rejected_before_gpu_preflight(self):
        with patch.dict(os.environ, {'KAGGLE_IS_COMPETITION_RERUN':'true'}):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic cannot enter'):
                exec(self.prefix, {})

    def test_diagnostic_packaging_is_unconditionally_disabled(self):
        bench = self.notebook['cells'][24]['source']
        node = next(n for n in ast.walk(ast.parse(bench)) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Constant) and n.test.value is False)
        with tempfile.TemporaryDirectory() as directory:
            scope = {'ARC40_RUN_MODE':'validation',
                     '_arc_report':{'local_validation_passed':True},
                     'WORKING_DIR':Path(directory)}
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'packaging-guard', 'exec'), scope)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_executed_archive_matches_gpu_request_and_retrieved_source(self):
        archive = ROOT/'Notebook/archive/arc3-lease2048-diagnostic-v1.ipynb'
        receipt = json.loads((ROOT/'runs/2026-10-17-afternoon/diagnostic-request.json').read_text())
        proof = json.loads((ROOT/'runs/2026-10-17-afternoon/diagnostic-source-proof.json').read_text())
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), receipt['notebook_sha256'])
        self.assertEqual(receipt['notebook_sha256'], proof['local_file_sha256'])
        self.assertTrue(proof['all_cell_sources_equal'])


if __name__ == '__main__':
    unittest.main()
