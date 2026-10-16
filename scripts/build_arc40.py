"""Build the auditable, self-contained ARC40 engineering candidate."""
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Notebook/arc3-target35-dprime-gpu-validation.ipynb"
OUT = ROOT / "Notebook/arc3-target40-engineered.ipynb"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one source anchor: {before[:90]!r}")
    return text.replace(before, after, 1)


def build():
    n = json.loads(BASE.read_text())
    for i, c in enumerate(n["cells"]):
        s = "".join(c["source"])
        if i != 4:
            s = s.replace("ARC35", "ARC40").replace("arc35", "arc40")
        c["source"] = s
        c.get("metadata", {}).pop("papermill", None)
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    n["metadata"].pop("papermill", None)
    old = n["metadata"].pop("arc35", {})
    n["metadata"]["arc40"] = {
        "target_exclusive": 40.0, "official_score": None,
        "inference_validated": False, "default_run_mode": "smoke",
        "base_notebook_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "harness_patch_sha256": hashlib.sha256(n["cells"][4]["source"].encode()).hexdigest(),
        "automatic_submission": False,
    }
    n["cells"][0]["source"] = """# ARC3 — engineered harness, target above 40

40 is an experimental target. No official score or GPU success is claimed.

The model, prompts, quantization, ten-stream cache configuration, MTP3, and D-prime
priority are preserved from the attributed Franzen/Affectify baseline below.
This revision fixes startup warmup, public evaluation coverage, and owned-process
cleanup. It does not assert a measured score or speed gain.

Default Save & Run is **smoke**: three deterministically selected public games,
180 seconds of gameplay after startup, a diagnostic report, and no packaging file.
Set ARC40_RUN_MODE=validation for all 25 public games, approximately121 minutes of
solver time, and a strict public gate above40. All games need inference evidence,
exact manifest identities, finite0-100 scores and acceptable terminal states.
The public split is a diagnostic partition of already public games, not unseen data.
The game-bootstrap bound describes those public games only.

Real Kaggle competition reruns select the live gateway and original532-minute
solver budget automatically. Smoke diagnostics cannot authorize submission.
GPU inference, runtime improvement and official score must be measured separately.

Sources:
- https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution
- https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub
- https://github.com/da-fr/arc-agi-3-solution
- https://github.com/Tufalabs/duck-harness

Third-party code and model licenses remain in force.
"""
    helpers = (ROOT / "scripts/arc40_runtime.py").read_text()
    evaluation = (ROOT / "scripts/arc40_evaluation.py").read_text()
    # Embed each module in its own namespace to preserve future imports/docstrings.
    n["cells"][1]["source"] = """import hashlib as _arc_hashlib
import json as _arc_json
import os as _arc_os
import time as _arc_time
from pathlib import Path as _ArcPath
import types as _arc_types
ARC40_START_EPOCH = _arc_time.time()
# Preserve the driver search path from the historically successful GPU run.
_arc_driver_paths = ['/usr/local/nvidia/lib64', '/usr/local/nvidia/lib']
_arc_driver_paths += [p for p in _arc_os.environ.get('LD_LIBRARY_PATH', '').split(':')
                      if p and p not in _arc_driver_paths]
_arc_os.environ['LD_LIBRARY_PATH'] = ':'.join(_arc_driver_paths)
ARC40_TARGET = 40.0
ARC40_EXPECTED_PUBLIC_GAMES = 25
ARC40_REFERENCE_COMPETITION_GAMES = 110
ARC40_PUBLIC_SECONDS = 532 * 60 * 25 / 110
ARC40_SPLIT_SEED = 'urad-arc3-target40-v2'
ARC40_TRUE_SUBMISSION = _arc_os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1','true'}
ARC40_RUN_MODE = 'competition' if ARC40_TRUE_SUBMISSION else _arc_os.environ.get('ARC40_RUN_MODE', 'smoke').strip().lower()
if ARC40_RUN_MODE not in {'smoke', 'validation', 'competition'}:
    raise RuntimeError('ARC40_RUN_MODE must be smoke or validation outside competition')
if ARC40_RUN_MODE == 'competition' and not ARC40_TRUE_SUBMISSION:
    raise RuntimeError('Competition mode requires the Kaggle competition rerun marker')
ARC40_EVAL_SECONDS = 180.0 if ARC40_RUN_MODE == 'smoke' else ARC40_PUBLIC_SECONDS
_arc_runtime = _arc_types.ModuleType('arc40_runtime')
_arc_evaluation = _arc_types.ModuleType('arc40_evaluation')
""" + "exec(" + repr(helpers) + ", _arc_runtime.__dict__)\n" + "exec(" + repr(evaluation) + ", _arc_evaluation.__dict__)\n" + """_arc_preflight_path = _ArcPath('/kaggle/working/arc40_preflight.json')
_arc_preflight_path.parent.mkdir(parents=True, exist_ok=True)
_arc_gpu_report = _arc_runtime.preflight_gpu(report_path=_arc_preflight_path)
print('GPU preflight:', _arc_gpu_report, flush=True)
print('Mode:', ARC40_RUN_MODE, '| target above', ARC40_TARGET, '| official score UNKNOWN', flush=True)
# Remove stale packaging and validation artifacts when rerunning an interactive kernel.
if not ARC40_TRUE_SUBMISSION:
    for _name in ('submission.parquet', 'arc40_validation.json', 'arc40_smoke.json'):
        (_arc_preflight_path.parent / _name).unlink(missing_ok=True)
"""
    launch = n["cells"][14]["source"]
    anchor = 'print(f"pid {proc.pid} -> {LOG}")'
    launch = replace_once(launch, anchor, anchor + "\n_arc_cleanup_server = _arc_runtime.register_owned_server(proc)")
    pos = launch.index('_arc_health_deadline = time.monotonic() + 300.0')
    launch = launch[:pos] + """if TRUE_SUBMISSION:
    # Release after the original startup deadline so the harness can RESET the
    # live gateway while HTTP retries wait for the model to finish loading.
    print('Competition: benchmark warmup released; backend may still be loading.', flush=True)
else:
    try:
        _arc_runtime.wait_for_server_ready(proc, url, timeout_s=300.0)
    except BaseException:
        show_log_tail()
        _arc_cleanup_server()
        raise
    print('Offline inference server verified ready.', flush=True)
"""
    n["cells"][14]["source"] = launch
    n["cells"][6]["source"] = replace_once(
        n["cells"][6]["source"], 'NOTEBOOK_START_TIME = time.time()',
        'NOTEBOOK_START_TIME = ARC40_START_EPOCH')
    accounting_patch = (ROOT / 'scripts/patches/token_accounting.patch').read_text()
    n['cells'][6]['source'] = replace_once(
        n['cells'][6]['source'], "print('harness patch applied successfully')",
        "_arc_accounting_patch = " + repr(accounting_patch) + "\n" +
        "_arc_accounting_path = Path('/kaggle/arc40-accounting.patch')\n" +
        "_arc_accounting_path.write_text(_arc_accounting_patch)\n" +
        "subprocess.run(['git','apply','--check',str(_arc_accounting_path)], cwd=BUNDLE_DIR/'src', check=True)\n" +
        "subprocess.run(['git','apply',str(_arc_accounting_path)], cwd=BUNDLE_DIR/'src', check=True)\n" +
        "print('Baseline and final-token accounting patches applied successfully')")
    config = n["cells"][18]["source"].replace('ARC40_PUBLIC_SECONDS', 'ARC40_EVAL_SECONDS')
    n["cells"][18]["source"] = config
    bench = n["cells"][23]["source"]
    start = bench.index('if not TRUE_SUBMISSION:\n    if len(bm.games)')
    end = bench.index('# Outside a real submission', start)
    manifest = """if not TRUE_SUBMISSION:
    if len(bm.games) != ARC40_EXPECTED_PUBLIC_GAMES:
        raise RuntimeError(f'Expected all25 public environments; found {len(bm.games)}')
    _arc_all_games = {str(g.env_name): g for g in bm.games}
    if len(_arc_all_games) != ARC40_EXPECTED_PUBLIC_GAMES:
        raise RuntimeError('Duplicate public environment identity')
    _arc_ordered_ids = sorted(_arc_all_games, key=lambda gid: _arc_hashlib.sha256(
        (ARC40_SPLIT_SEED + '/' + gid.split('-')[0]).encode()).hexdigest())
    if ARC40_RUN_MODE == 'smoke':
        bm.games = [_arc_all_games[gid] for gid in _arc_ordered_ids[:3]]
    _arc_expected_ids = [str(g.env_name) for g in bm.games]
    _arc_public_split = {gid: ('heldout' if i % 2 else 'development')
                         for i, gid in enumerate(_arc_ordered_ids)}
    (WORKING_DIR / 'arc40_eval_manifest.json').write_text(_arc_json.dumps({
        'mode': ARC40_RUN_MODE, 'all_public_ids': sorted(_arc_all_games),
        'expected_game_ids': _arc_expected_ids, 'split': _arc_public_split,
        'split_seed': ARC40_SPLIT_SEED, 'solver_budget_seconds': ARC40_EVAL_SECONDS,
        'official_score': None}, indent=2))

"""
    bench = bench[:start] + manifest + bench[end:]
    bench = bench.replace('timedelta(seconds=ARC40_PUBLIC_SECONDS)', 'timedelta(seconds=ARC40_EVAL_SECONDS)')
    bench = replace_once(bench, 'soft_end = None', """# Count startup against the real nine-hour notebook wall limit.
soft_end = datetime.fromtimestamp(ARC40_START_EPOCH + 532 * 60) if TRUE_SUBMISSION else None
if TRUE_SUBMISSION and _arc_time.time() >= ARC40_START_EPOCH + 532 * 60:
    raise RuntimeError('Competition wall budget exhausted during startup')""")
    anchor = 'await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)'
    bench = replace_once(bench, anchor, """try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
finally:
    # Release GPU before CPU aggregation or a failed validation assertion.
    if '_stop_monitor' in globals():
        _stop_monitor.set()
    _arc_cleanup_server()
""")
    start = bench.index('if not TRUE_SUBMISSION:\n    import math')
    bench = bench[:start] + """if not TRUE_SUBMISSION:
    _arc_rows = []
    for _run in bm.game_runs:
        _arc_rows.append({
            'game_id': str(_run.game_id),
            'score': None if _run.final_score is None else float(_run.final_score),
            'state': str(_run.state),
            'levels_completed': int(_run.levels_completed),
            'total_levels': int(_run.number_of_levels),
            'actions': int(sum(_run.actions_per_level)),
            'generated_tokens': int(sum(r.generated_tokens for r in _run.history) + _run.final_generated_tokens),
            'actions_per_level': list(_run.actions_per_level),
        })
    _arc_report = _arc_evaluation.assess_public_run(
        _arc_rows, expected_game_ids=_arc_expected_ids, target=ARC40_TARGET,
        evaluation_kind='smoke' if ARC40_RUN_MODE == 'smoke' else 'full_public',
        elapsed_seconds=_arc_time.time()-ARC40_START_EPOCH,
        budget_seconds=ARC40_EVAL_SECONDS+1200.0,
        split_seed=ARC40_SPLIT_SEED)
    _arc_report.update({
        'candidate': 'Franzen + Affectify D-prime, engineering revision2',
        'solver_budget_seconds': ARC40_EVAL_SECONDS,
        'elapsed_seconds': _arc_time.time() - ARC40_START_EPOCH,
        'base_notebook_sha256': 'BASE_HASH_PLACEHOLDER',
    })
    _arc_report_path = WORKING_DIR / ('arc40_smoke.json' if ARC40_RUN_MODE == 'smoke' else 'arc40_validation.json')
    _arc_report_path.write_text(_arc_json.dumps(_arc_report, indent=2, allow_nan=False))
    print(_arc_json.dumps({k:v for k,v in _arc_report.items() if k != 'games'}, indent=2), flush=True)
    if ARC40_RUN_MODE == 'validation' and not _arc_report['local_validation_passed']:
        raise RuntimeError('Public target above40 did not pass. Inspect arc40_validation.json.')
    # No placeholder packaging parquet is created by smoke/validation. Submission
    # requires a separately reviewed artifact from a completed competition run.
"""
    bench = bench.replace('BASE_HASH_PLACEHOLDER', n['metadata']['arc40']['base_notebook_sha256'])
    # Cleanup also covers game-list/manifest failures before bm.run is entered.
    preamble, body = bench.split("if TRUE_SUBMISSION:\n    # Real submission", 1)
    body = "if TRUE_SUBMISSION:\n    # Real submission" + body
    bench = preamble + "try:\n" + "\n".join("    " + line if line else "" for line in body.splitlines()) + "\nfinally:\n    _arc_cleanup_server()\n"
    n["cells"][23]["source"] = bench
    n["metadata"]["kaggle"]["isGpuEnabled"] = True
    n["metadata"]["kaggle"].pop("dockerImageVersionId", None)
    n["metadata"]["kaggle"]["accelerator"] = "nvidiaRtxPro6000"
    # Compile actual notebook code, including cell magics/top-level await.
    from IPython.core.inputtransformer2 import TransformerManager
    tm = TransformerManager()
    for i,c in enumerate(n['cells']):
        if c['cell_type'] == 'code':
            compile(tm.transform_cell(c['source']), f'cell_{i}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    OUT.write_text(json.dumps(n, indent=1) + "\n")
    m = json.loads((ROOT / 'Notebook/kernel-metadata.json').read_text())
    m.update(id='koushikrudra/arc3-target40-engineered-harness', title='ARC3 Target40 Engineered Harness',
             code_file=OUT.name, enable_gpu=True, machine_shape='NvidiaRtxPro6000')
    # Reuse the upstream notebook's current documented image, read by the API.
    source_meta = json.loads((ROOT / 'scripts/upstream-kernel-metadata.json').read_text())
    m['docker_image'] = source_meta['docker_image']
    m['docker_image_pinning_type'] = 'original'
    run_dir = ROOT / 'launch/arc40'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + "\n")
    (ROOT / 'Notebook/arc40-kernel-metadata.json').write_text(json.dumps(m, indent=2) + "\n")
    (run_dir / OUT.name).write_bytes(OUT.read_bytes())
    return OUT

if __name__ == '__main__':
    print(build())
