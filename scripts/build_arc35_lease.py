"""Freeze a conservative candidate and a cheaper functional diagnostic.

The diagnostic deliberately uses a different threshold to exercise the hook;
its result is never score evidence for the65,536-token candidate.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

from build_no_action_candidate import build as build_lease

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'Notebook/arc3-target40-engineered.ipynb'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'Expected one source anchor: {before[:80]}')
    return source.replace(before, after, 1)


def build(*, diagnostic=False):
    quantum = 2048 if diagnostic else 65536
    slug = 'arc3-lease2048-diagnostic' if diagnostic else 'arc3-lease65536-validation'
    path = ROOT/'Notebook'/f'{slug}.ipynb'
    build_lease(BASELINE, path, token_quantum=quantum)
    notebook = json.loads(path.read_text())
    mode = 'smoke' if diagnostic else 'validation'
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = ''.join(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['cells'][0]['source'] = f'''# ARC3: no-action admission lease ({quantum:,} tokens)

This is an unproven scheduling experiment built from the completed public56.35
baseline, whose observed official result is30.59. No official35/40, rank, score
gain or GPU-time reduction is established. Working prompts, retained history,
Python state, model and serving settings remain unchanged.

At complete pre-request boundaries, voluntarily requeue a game after{quantum:,}
generated tokens without an executed action. The existing D-prime comparator
still chooses the next owner; immediate reacquisition is possible. An existing
queue wait is not cancellable once entered. No response is interrupted.

Default mode: **{mode}**. Every run records levels completed, reached and unseen,
per-level action counts, and no-action tail tokens. Reaching a game does not mean
all its levels were solved. The full candidate evaluates all25 public games for
7,254.545 solver seconds. The diagnostic uses three games for180 solver seconds,
a lower2,048-token threshold, and one bounded synthetic25/110-state probe on the
same loaded server. It cannot measure the full candidate's gameplay score.

The diagnostic refuses competition reruns and emits no submission packaging.
The public validation gate is a public diagnostic, never a hidden-score bound.
All baseline attributions and third-party licenses apply.
'''
    setup = notebook['cells'][1]['source']
    if diagnostic:
        setup = replace_once(setup,
            "ARC40_RUN_MODE = 'competition' if ARC40_TRUE_SUBMISSION else _arc_os.environ.get('ARC40_RUN_MODE', 'validation').strip().lower()",
            "ARC40_RUN_MODE = 'smoke'  # Fixed diagnostic budget; ignore mode overrides.")
        setup = replace_once(setup, "ARC40_EVAL_SECONDS =",
            "if ARC40_TRUE_SUBMISSION:\n    raise RuntimeError('Functional diagnostic cannot enter the competition')\nARC40_EVAL_SECONDS =")
    coverage_module = (ROOT/'scripts/arc_level_coverage.py').read_text()
    setup += ("\n_arc35_coverage_module = _arc_types.ModuleType('arc_level_coverage')\n"
              f"exec({coverage_module!r}, _arc35_coverage_module.__dict__)\n")
    notebook['cells'][1]['source'] = setup
    bench = notebook['cells'][23]['source']
    anchor = '        bm._save_json()'
    bench = replace_once(bench, anchor, anchor + '''
        _arc35_coverage = _arc35_coverage_module.assess_level_coverage(
            _arc_json.loads((WORKING_DIR/'benchmark.json').read_text()),
            expected_game_ids=_arc_expected_ids, levels_are_monotonic=True,
            configured_limits={'solver_wall_seconds': ARC40_EVAL_SECONDS,
                'concurrent_active_streams':10, 'no_action_token_quantum':QUANTUM})
        (WORKING_DIR/'arc35_level_coverage.json').write_text(
            _arc_json.dumps(_arc35_coverage, indent=2, allow_nan=False))
        (WORKING_DIR/'arc35_handover.json').write_text(_arc_json.dumps(
            _arc35_handover_stats.snapshot(), indent=2, allow_nan=False))
        print('Level coverage:', _arc35_coverage['summary'], flush=True)
        print('No-action handover:', _arc35_handover_stats.snapshot(), flush=True)
        if not _arc35_coverage['valid_evidence']:
            raise RuntimeError('Level ledger evidence is inconsistent')
'''.replace('QUANTUM', str(quantum)))
    bench = replace_once(bench,
        "'candidate': 'Franzen + Affectify D-prime, engineering revision2',",
        f"'candidate': 'Franzen + Affectify D-prime, no-action lease {quantum}',\n            'no_action_token_quantum': {quantum},\n            'diagnostic_only': {diagnostic!r},\n            'candidate_baseline_notebook_sha256': {hashlib.sha256(BASELINE.read_bytes()).hexdigest()!r},")
    if diagnostic:
        bench = replace_once(bench,
            "    if ARC40_RUN_MODE == 'validation' and _arc_report['local_validation_passed']:",
            "    if False:  # A diagnostic never emits competition packaging.")
        helper = (ROOT/'scripts/load_parity_probe.py').read_text()
        probe = '''# One bounded synthetic probe reuses the loaded server.
# Failed probes remain diagnostic evidence; gameplay integration still runs.
import subprocess as _probe_subprocess
_probe_dir = WORKING_DIR/'runtime_load_probe'
_probe_dir.mkdir(parents=True, exist_ok=True)
_probe_script = WORKING_DIR/'arc3_load_parity_probe.py'
_probe_script.write_text(HELPER)
try:
    _probe_result = _probe_subprocess.run([
        PYTHON, '-u', str(_probe_script), '--base-url', SERVER_BASE_URL,
        '--model', SERVED_MODEL_NAME, '--tokenizer', MODEL_DIR,
        '--output', str(_probe_dir)], env=env, text=True,
        capture_output=True, timeout=390)
    (_probe_dir/'probe-client.log').write_text(_probe_result.stdout+'\\n'+_probe_result.stderr)
    print('Transport probe return code:', _probe_result.returncode, flush=True)
    print(_probe_result.stdout[-6000:], flush=True)
except _probe_subprocess.TimeoutExpired:
    (_probe_dir/'hard_timeout.json').write_text(_arc_json.dumps({
        'timeout_seconds':390, 'valid_comparison':False,
        'official_score':None, 'authorizes_submission':False}, indent=2))
    # Do not overlap gameplay with timed-out requests still running server-side.
    return
'''.replace('HELPER', repr(helper))
        probe = 'def _arc35_run_probe():\n' + '\n'.join('    '+line if line else '' for line in probe.splitlines()) + '\n'
        anchor = '        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)'
        bench = replace_once(bench, anchor, anchor + '\n        _arc35_run_probe()')
        notebook['cells'].insert(17, dict(cell_type='code', metadata={},
            source=probe, execution_count=None, outputs=[]))
    notebook['cells'][24 if diagnostic else 23]['source'] = bench
    hook_meta = notebook['metadata']['arc35_no_action_handover']
    hook_meta['baseline_changed_cell_indices'] = [22]
    hook_meta['changed_cell_indices'] = [23 if diagnostic else 22]
    notebook['metadata']['arc40']['default_run_mode'] = mode
    notebook['metadata']['arc35_lease_candidate'] = {
        'build_revision':2, 'diagnostic_only': diagnostic, 'token_quantum': quantum,
        'baseline_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        'coverage_module_sha256': hashlib.sha256(coverage_module.encode()).hexdigest(),
        'official_score':None, 'score_gain_established':False,
        'speed_gain_established':False, 'automatic_submission':False}
    from IPython.core.inputtransformer2 import TransformerManager
    transformer = TransformerManager()
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(transformer.transform_cell(cell['source']), f'lease_cell{index}',
                    'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    metadata = json.loads((ROOT/'Notebook/arc40-kernel-metadata.json').read_text())
    metadata.update(id='koushikrudra/'+slug, title='ARC3 '+slug.replace('-', ' ').title(),
                    code_file=path.name)
    directory = ROOT/'launch'/slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory/path.name).write_bytes(path.read_bytes())
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (ROOT/'Notebook'/f'{slug}-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return path


if __name__ == '__main__':
    print(build())
    print(build(diagnostic=True))
