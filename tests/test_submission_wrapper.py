import ast
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class SubmissionWrapperTests(unittest.TestCase):
    def test_commit_executes_every_code_cell_without_loading_model(self):
        notebook = json.loads((ROOT/'Notebook/arc3-target40-submission.ipynb').read_text())
        pandas = Mock()
        scope = {'__name__':'__main__'}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                'KAGGLE_IS_COMPETITION_RERUN':'false', 'ARC40_WORKING_DIR':folder}), patch.dict(
                sys.modules, {'pandas':pandas}), patch('subprocess.Popen') as spawn, patch('subprocess.run') as run:
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    result = eval(compile(cell['source'], f'cell_{index}', 'exec',
                                  flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), scope)
                    if inspect.isawaitable(result):
                        asyncio.run(result)
            spawn.assert_not_called()
            run.assert_not_called()
            pandas.DataFrame.return_value.to_parquet.assert_called_once()
            manifest = json.loads((Path(folder)/'arc40_submission_manifest.json').read_text())
            self.assertFalse(manifest['is_score_evidence'])
            self.assertFalse(manifest['full_public_validation_passed'])
            self.assertIsNone(manifest['official_score'])

    def test_real_rerun_skips_commit_stub_and_selects_live_implementation(self):
        notebook = json.loads((ROOT/'Notebook/arc3-target40-submission.ipynb').read_text())
        scope = {'__name__':'__main__'}
        pandas = Mock()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                'KAGGLE_IS_COMPETITION_RERUN':'true', 'ARC40_WORKING_DIR':folder}), patch.dict(
                sys.modules, {'pandas':pandas}):
            exec(notebook['cells'][1]['source'], scope)
            self.assertTrue(scope['_ARC40_LIVE_COMPETITION'])
            pandas.DataFrame.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])
        # The live path retains the audited hardware preflight and full wall budget.
        source = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn('ARC40_START_EPOCH + 532 * 60', source)
        self.assertIn('_arc_runtime.preflight_gpu', source)


if __name__ == '__main__':
    unittest.main()
