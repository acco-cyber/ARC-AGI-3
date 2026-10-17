"""CPU-only QA of the tokenizer smoke builder's corrected guard and packaging."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[2]/'scripts/build_cpu_tokenizer_smoke.py'
source = BUILDER.read_text()
tree = ast.parse(source)
runner = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'RUNNER' for target in node.targets))
compile(runner, 'embedded-tokenizer-runner', 'exec')
runner_tree = ast.parse(runner)
try_node = next(node for node in runner_tree.body if isinstance(node, ast.Try))
start = next(index for index, node in enumerate(try_node.body) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loaded_frameworks')
fragment = ast.fix_missing_locations(ast.Module(body=try_node.body[start:-1], type_ignores=[]))
results = []
for initialized in (False, True):
    calls = []

    class Cuda:
        def is_initialized(self):
            calls.append('is_initialized')
            return initialized

        def is_available(self):
            raise AssertionError('Device availability must not be queried')

        def init(self):
            raise AssertionError('CUDA must not be initialized')

    report = {}
    raised = False
    with patch.dict(sys.modules, {'torch': types.SimpleNamespace(cuda=Cuda())}):
        try:
            exec(compile(fragment, 'actual-embedded-cuda-guard', 'exec'), {'sys': sys, 'report': report})
        except RuntimeError:
            raised = True
    assert raised == initialized
    assert calls == ['is_initialized']
    assert report['torch_cuda_initialized'] is initialized
    results.append({'test': 'rejects_initialized_cuda' if initialized else 'accepts_incidental_cpu_torch_import',
                    'passed': True, 'cuda_calls': calls})

spec = importlib.util.spec_from_file_location('cpu_smoke_builder_qa', BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
with tempfile.TemporaryDirectory() as folder:
    build = builder.build(output_dir=Path(folder))
    metadata = json.loads(Path(build['metadata']).read_text())
    assert Path(build['kernel_metadata']).read_bytes() == Path(build['metadata']).read_bytes()
    assert metadata['enable_gpu'] is False
    assert 'accelerator' not in metadata and 'machine_shape' not in metadata
    notebook = json.loads(Path(build['notebook']).read_text())
    notebook_code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert "target = Path('/tmp/arc3-cpu-tokenizer-site-packages')" in notebook_code
    results.append({'test': 'cpu_metadata_alias_identical_and_dependencies_disposable', 'passed': True})

result = {'scope': 'AST-extracted runner guard and builder output; no actual tokenizer or GPU execution',
          'tests': results, 'passed': all(test['passed'] for test in results)}
(HERE/'cpu-smoke-guard-qa.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
