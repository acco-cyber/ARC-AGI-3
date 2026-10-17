"""Build a CPU-only Kaggle notebook for the attached tokenizer and probe helper.

Does not upload, run Kaggle, load model weights, start a server or submit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

INSTALLER = r'''from __future__ import annotations
import email.parser
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

try:
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version

root = Path(sys.argv[1])
target = Path(sys.argv[2])
receipt = Path(sys.argv[3])
compatible_tags = set(sys_tags())
forbidden = {'torch', 'torchvision', 'torchaudio', 'tensorflow', 'flax', 'jax', 'jaxlib',
             'sglang', 'triton', 'flashinfer', 'flashinfer-python', 'vllm', 'cuda-python'}

def allowed(name):
    name = canonicalize_name(name)
    return name not in forbidden and not name.startswith(('nvidia-', 'cuda-', 'flashinfer-', 'torch-'))

all_wheels = []
for path in sorted(root.glob('*.whl')):
    try:
        name, version, build, tags = parse_wheel_filename(path.name)
    except Exception:
        continue
    if compatible_tags.intersection(tags):
        all_wheels.append((canonicalize_name(name), version, path))

plan = {}
requirements = []
processed = set()

def choose(name, specifier=None, *, pinned=False):
    name = canonicalize_name(name)
    if not allowed(name):
        raise RuntimeError('Refusing heavyweight/GPU dependency: ' + name)
    candidates = [(version, path) for n, version, path in all_wheels
                  if n == name and (specifier is None or specifier.contains(version, prereleases=True))]
    if not candidates:
        raise RuntimeError(f'No compatible offline wheel for {name}{specifier or ""}')
    if pinned and len(candidates) != 1:
        raise RuntimeError(f'Expected one pinned offline {name} wheel, found {len(candidates)}')
    version, path = max(candidates, key=lambda item: (item[0], item[1].name))
    current = plan.get(name)
    if current is not None and current[0] != version:
        raise RuntimeError(f'Offline requirement conflict for {name}: {current[0]} versus {version}')
    plan[name] = (version, path)
    return path

for package in ('transformers', 'tokenizers', 'huggingface-hub'):
    choose(package, pinned=True)

image_dependencies = {}
while True:
    pending = [(name, value) for name, value in plan.items() if name not in processed]
    if not pending:
        break
    for name, (version, path) in pending:
        processed.add(name)
        with zipfile.ZipFile(path) as wheel:
            names = [member for member in wheel.namelist() if member.endswith('.dist-info/METADATA')]
            if len(names) != 1:
                raise RuntimeError('Expected one wheel metadata file: ' + path.name)
            message = email.parser.Parser().parsestr(wheel.read(names[0]).decode())
        for raw in message.get_all('Requires-Dist', []):
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({'extra': ''}):
                continue
            dep = canonicalize_name(req.name)
            if req.url:
                raise RuntimeError('Direct URL dependency is not permitted: ' + raw)
            requirements.append((name, raw))
            if dep in plan:
                if not req.specifier.contains(plan[dep][0], prereleases=True):
                    raise RuntimeError(f'Pinned wheel requirement conflict: {name} requires {raw}')
                continue
            try:
                installed = metadata.version(dep)
            except metadata.PackageNotFoundError:
                installed = None
            if installed is not None and req.specifier.contains(installed, prereleases=True):
                image_dependencies[dep] = installed
                continue
            choose(dep, req.specifier)

# Verify all collected constraints after the closure is complete.
for owner, raw in requirements:
    req = Requirement(raw)
    dep = canonicalize_name(req.name)
    version = plan[dep][0] if dep in plan else image_dependencies[dep]
    if not req.specifier.contains(version, prereleases=True):
        raise RuntimeError(f'Offline dependency conflict: {owner} requires {raw}; selected {version}')

target.mkdir(parents=True, exist_ok=True)
install = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
           '--disable-pip-version-check', '--no-warn-script-location', '--target', str(target)]
install += [str(value[1]) for name, value in sorted(plan.items())]
subprocess.run(install, check=True, timeout=180)
report = {'purpose': 'CPU tokenizer-only offline dependency preparation',
          'python': sys.version, 'target_directory': str(target),
          'pinned_tokenizer_packages': ['transformers', 'tokenizers', 'huggingface-hub'],
          'installed_wheels': {name: {'version': str(version), 'wheel': path.name}
                               for name, (version, path) in sorted(plan.items())},
          'reused_image_dependencies': dict(sorted(image_dependencies.items())),
          'gpu_packages_installed': False, 'model_weights_loaded': False,
          'server_started': False, 'http_requests_sent': False}
receipt.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
'''

RUNNER = r'''from __future__ import annotations
import collections.abc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

started = time.monotonic()
sys.path.insert(0, sys.argv[1])
model_path, helper_path, output_path = map(Path, sys.argv[2:5])
report = {'scope': 'CPU-only real-tokenizer shape smoke; no model inference or score evidence',
          'passed': False, 'weights_loaded': False, 'server_started': False,
          'http_requests_sent': False, 'gpu_enabled': False, 'official_score': None,
          'authorizes_submission': False,
          'helper_sha256': hashlib.sha256(helper_path.read_bytes()).hexdigest(),
          'model_path': str(model_path), 'python': sys.version, 'cases': []}

def token_shape(value):
    info = {'type': type(value).__module__ + '.' + type(value).__name__}
    if isinstance(value, collections.abc.Mapping):
        info['keys'] = sorted(map(str, value))
        info['mapping_length'] = len(value)
        if 'input_ids' not in value:
            raise RuntimeError('Template mapping has no input_ids')
        value = value['input_ids']
        info['input_ids_type'] = type(value).__module__ + '.' + type(value).__name__
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError('Unsupported token sequence: ' + str(type(value)))
    if len(value) and isinstance(value[0], (list, tuple)):
        info['shape'] = [len(value), len(value[0])]
        if len(value) != 1:
            raise RuntimeError('Expected one rendered sequence')
        value = value[0]
    else:
        info['shape'] = [len(value)]
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise RuntimeError('Rendered input_ids contain noninteger IDs')
    info['token_count'] = len(value)
    return info

try:
    from transformers import AutoTokenizer
    import transformers
    import tokenizers
    import huggingface_hub
    report['package_versions'] = {name: metadata.version(name) for name in
        ('transformers', 'tokenizers', 'huggingface-hub', 'jinja2')}
    report['package_paths'] = {name: module.__file__ for name, module in
        [('transformers', transformers), ('tokenizers', tokenizers), ('huggingface_hub', huggingface_hub)]}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    report['tokenizer_class'] = type(tokenizer).__module__ + '.' + type(tokenizer).__name__
    report['tokenizer_is_fast'] = bool(getattr(tokenizer, 'is_fast', False))
    report['chat_template_type'] = type(getattr(tokenizer, 'chat_template', None)).__name__
    spec = importlib.util.spec_from_file_location('cpu_tokenizer_probe_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    for index, target in enumerate((2048, 8192, 16384)):
        case = {'case': index, 'target_input_tokens': target, 'max_output_tokens': 64, 'seed': 20261017 + index}
        payload, helper_count = helper.build_payload(tokenizer, case, 0, 'flashnext', 25)
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
        default_shape = token_shape(tokenizer.apply_chat_template(payload['messages'], **kwargs))
        flat_shape = token_shape(tokenizer.apply_chat_template(payload['messages'], return_dict=False, **kwargs))
        actual = flat_shape['token_count']
        valid = (helper_count == actual == default_shape['token_count'] and abs(actual-target) <= 8)
        row = {'target_input_tokens': target, 'helper_token_count': helper_count,
               'default_template': default_shape, 'explicit_flat_template': flat_shape,
               'max_output_tokens': payload['max_tokens'], 'passed': valid}
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
        if not valid:
            raise RuntimeError(f'Real tokenizer input-length mismatch for {target}: helper={helper_count}, actual={actual}')
    loaded_frameworks = sorted(name for name in ('torch', 'tensorflow', 'jax', 'flax') if name in sys.modules)
    report['loaded_ml_frameworks'] = loaded_frameworks
    # A tokenizer dependency may import an existing framework incidentally.
    # Reading this flag does not initialize CUDA or probe device availability.
    torch_module = sys.modules.get('torch')
    torch_cuda = getattr(torch_module, 'cuda', None)
    report['torch_cuda_initialized'] = bool(torch_cuda.is_initialized()) if torch_cuda is not None else False
    if report['torch_cuda_initialized']:
        raise RuntimeError('CPU tokenizer smoke observed an initialized CUDA runtime')
    report['passed'] = len(report['cases']) == 3 and all(row['passed'] for row in report['cases'])
except BaseException as exc:
    report['error_type'] = type(exc).__name__
    report['error'] = str(exc)
    raise
finally:
    report['elapsed_seconds'] = time.monotonic() - started
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, indent=2), flush=True)
'''


def build(output_dir=REPO/'launch/arc3-cpu-tokenizer-shape-smoke', helper_path=REPO/'scripts/load_parity_probe.py'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    helper_path = Path(helper_path)
    helper = helper_path.read_text()
    helper_sha = hashlib.sha256(helper.encode()).hexdigest()
    slug = 'arc3-cpu-tokenizer-shape-smoke'
    code = '''import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1', 'true'}:
    raise RuntimeError('CPU tokenizer diagnostic must not run as a competition submission')
work = Path('/kaggle/working/cpu_tokenizer_smoke')
work.mkdir(parents=True, exist_ok=True)
model_candidates = [
    Path('/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'),
    Path('/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'),
]
models = list(dict.fromkeys(path.resolve() for path in model_candidates if (path/'tokenizer.json').is_file()))
if len(models) != 1:
    raise RuntimeError(f'Expected exactly one attached model tokenizer, found {models}')
wheel_roots = []
for root in [Path('/kaggle/input/datasets/dfranzen/pennyroyal-v253'), Path('/kaggle/input/pennyroyal-v253')]:
    if root.is_dir():
        for parent in [root, *root.rglob('wheels')]:
            if parent.is_dir() and list(parent.glob('transformers-*.whl')):
                wheel_roots.append(parent.resolve())
wheel_roots = list(dict.fromkeys(wheel_roots))
if len(wheel_roots) != 1:
    raise RuntimeError(f'Expected exactly one pinned Pennyroyal wheel directory, found {wheel_roots}')
installer = work/'install_cpu_tokenizer.py'
runner = work/'run_cpu_tokenizer_smoke.py'
helper_path = work/'load_parity_probe.py'
installer.write_text(INSTALLER_PLACEHOLDER)
runner.write_text(RUNNER_PLACEHOLDER)
helper_path.write_text(HELPER_PLACEHOLDER)
assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == HELPER_SHA_PLACEHOLDER
runtime_env = dict(os.environ)
runtime_env.update({'USE_TORCH':'0', 'USE_TF':'0', 'USE_FLAX':'0',
    'CUDA_VISIBLE_DEVICES':'', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
    'HF_DATASETS_OFFLINE':'1', 'PIP_NO_INDEX':'1', 'PIP_DISABLE_PIP_VERSION_CHECK':'1',
    'TOKENIZERS_PARALLELISM':'false', 'PYTHONNOUSERSITE':'1'})
runtime_env.pop('PYTHONPATH', None)
# Installed dependencies are disposable setup, not notebook output artifacts.
target = Path('/tmp/arc3-cpu-tokenizer-site-packages')
commands = [
    ('dependency-preparation', [sys.executable, '-u', str(installer), str(wheel_roots[0]), str(target), str(work/'dependencies.json')], 210),
    ('real-tokenizer-smoke', [sys.executable, '-u', str(runner), str(target), str(models[0]), str(helper_path), str(work/'tokenizer_smoke.json')], 120),
]
for label, command, timeout in commands:
    print(label, flush=True)
    result = subprocess.run(command, env=runtime_env, text=True, capture_output=True, timeout=timeout)
    (work/(label+'.log')).write_text(result.stdout+'\\n'+result.stderr)
    print(result.stdout[-16000:], flush=True)
    if result.returncode:
        print(result.stderr[-6000:], flush=True)
        raise RuntimeError(label+' failed; inspect saved CPU evidence')
report = json.loads((work/'tokenizer_smoke.json').read_text())
assert report['passed']
print('CPU tokenizer shape smoke passed. No GPU, server, HTTP or model weights used.', flush=True)
'''.replace('INSTALLER_PLACEHOLDER', repr(INSTALLER)).replace('RUNNER_PLACEHOLDER', repr(RUNNER)).replace('HELPER_PLACEHOLDER', repr(helper)).replace('HELPER_SHA_PLACEHOLDER', repr(helper_sha))
    header = '''# ARC3 CPU tokenizer shape smoke

Uses the actual attached AutoTokenizer and pinned Pennyroyal tokenizer wheels to
check the transport probe's 2K, 8K and 16K payload counts. It records default
chat-template return types, mapping keys, input-ID shape, actual counts, package
versions and helper SHA256. Explicit flat-token rendering independently checks
the helper's count against the real tokenizer.

CPU only; no inference server, HTTP call, game execution, model weights or
submission artifact. This validates tokenizer compatibility, not gameplay score
or GPU transport performance. Required light dependencies reuse the CPU image
or compatible attached offline wheels. Heavy inference packages are refused.
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.12'},
                     'arc3_cpu_tokenizer_smoke': {'helper_sha256': helper_sha, 'gpu_enabled': False,
                         'weights_loaded': False, 'official_score': None}},
        'cells': [
            {'id': 'intro', 'cell_type': 'markdown', 'metadata': {}, 'source': header},
            {'id': 'cpu-smoke', 'cell_type': 'code', 'metadata': {}, 'source': code, 'outputs': [], 'execution_count': None},
        ]}
    for source in (INSTALLER, RUNNER, code):
        ast.parse(source)
        compile(source, '<cpu-tokenizer-smoke>', 'exec')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    path = output_dir/(slug+'.ipynb')
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    metadata = json.loads((REPO/'Notebook/arc40-kernel-metadata.json').read_text())
    metadata.update(id='koushikrudra/'+slug, title='ARC3 CPU Tokenizer Shape Smoke',
        code_file=path.name, enable_gpu=False, enable_tpu=False, enable_internet=False,
        dataset_sources=['dfranzen/pennyroyal-v253'], competition_sources=[],
        model_sources=['dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1'])
    metadata.pop('machine_shape', None)
    metadata.pop('accelerator', None)
    metadata_path = output_dir/'cpu-tokenizer-kernel-metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    standard_metadata_path = output_dir/'kernel-metadata.json'
    standard_metadata_path.write_bytes(metadata_path.read_bytes())
    result = {'notebook': str(path), 'metadata': str(metadata_path),
              'kernel_metadata': str(standard_metadata_path),
              'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'helper_sha256': helper_sha, 'nbformat_validated': True,
              'gpu_enabled': False, 'machine_shape_present': 'machine_shape' in metadata,
              'actual_tokenizer_executed_locally': False}
    (output_dir/'cpu-tokenizer-build.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, default=REPO/'launch/arc3-cpu-tokenizer-shape-smoke')
    parser.add_argument('--helper-path', type=Path, default=REPO/'scripts/load_parity_probe.py')
    args = parser.parse_args()
    print(json.dumps(build(args.output_directory, args.helper_path), indent=2))
