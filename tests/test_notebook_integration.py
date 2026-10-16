import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import arc40_runtime


ROOT = Path(__file__).resolve().parents[1]


class NotebookIntegrationTests(unittest.TestCase):
    def test_preflight_timeout_leaves_failure_receipt_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / 'preflight.json'
            with patch('arc40_runtime.sys.version_info', (3, 12)), patch(
                    'arc40_runtime.subprocess.run',
                    side_effect=subprocess.TimeoutExpired('nvidia-smi', 10)) as probe:
                with self.assertRaisesRegex(RuntimeError, 'GPU preflight failed'):
                    arc40_runtime.preflight_gpu(report_path=receipt)
            report = json.loads(receipt.read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['model_inference_attempted'])
            self.assertEqual(probe.call_args.kwargs['timeout'], 10)

    def test_readiness_rejects_exited_server_immediately(self):
        proc = Mock()
        proc.poll.return_value = 12
        with self.assertRaisesRegex(RuntimeError, 'exited before readiness'):
            arc40_runtime.wait_for_server_ready(proc, 'http://unused', timeout_s=0)

    def test_competition_releases_benchmark_without_second_readiness_wait(self):
        notebook = json.loads((ROOT/'Notebook/arc3-target40-engineered.ipynb').read_text())
        source = notebook['cells'][14]['source']
        fragment = source[source.rindex('if TRUE_SUBMISSION:'):]
        runtime = Mock()
        scope = {'TRUE_SUBMISSION': True, '_arc_runtime': runtime}
        with redirect_stdout(io.StringIO()):
            exec(fragment, scope)
        runtime.wait_for_server_ready.assert_not_called()

    def test_offline_readiness_failure_releases_owned_server(self):
        notebook = json.loads((ROOT/'Notebook/arc3-target40-engineered.ipynb').read_text())
        source = notebook['cells'][14]['source']
        fragment = source[source.rindex('if TRUE_SUBMISSION:'):]
        runtime = Mock()
        runtime.wait_for_server_ready.side_effect = RuntimeError('not ready')
        cleanup, tail = Mock(), Mock()
        scope = {'TRUE_SUBMISSION': False, '_arc_runtime': runtime,
                 'proc': Mock(), 'url': 'http://unused',
                 '_arc_cleanup_server': cleanup, 'show_log_tail': tail}
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            exec(fragment, scope)
        cleanup.assert_called_once()
        tail.assert_called_once()

    def test_generated_notebook_compiles_with_magics_and_await(self):
        from IPython.core.inputtransformer2 import TransformerManager
        import nbformat
        notebook = nbformat.read(ROOT/'Notebook/arc3-target40-engineered.ipynb', as_version=4)
        nbformat.validate(notebook)
        transformer = TransformerManager()
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(transformer.transform_cell(cell.source), f'cell_{index}',
                        'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


if __name__ == '__main__':
    unittest.main()
