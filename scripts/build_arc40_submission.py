"""Package the engineered live gateway notebook after a completed GPU smoke.

Commit output is the official starter schema stub, never a fabricated score.
This builder does not assert or require a hidden score before it can be measured.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    source_path = ROOT/'Notebook/arc3-target40-engineered.ipynb'
    candidate = json.loads(source_path.read_text())
    evidence_dir = ROOT/'runs/2026-10-16T1225KST'
    proof = json.loads((evidence_dir/'technical-smoke-proof.json').read_text())
    report = json.loads((evidence_dir/'smoke-v1/arc40_smoke.json').read_text())
    preflight = json.loads((evidence_dir/'smoke-v1/arc40_preflight.json').read_text())
    if proof['kernel_status']['status'] != 'COMPLETE' or not preflight['passed']:
        raise RuntimeError('A completed GPU smoke and hardware preflight are required')
    if not report['valid_evaluation'] or not report['checks']['all_games_have_inference']:
        raise RuntimeError('Every sampled game must have real inference and valid termination')
    if not report['checks']['all_games_terminal_without_crash']:
        raise RuntimeError('GPU smoke has a crashed or unfinished game')
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        'purpose':'Official starter packaging for a live competition gateway rerun',
        'target_official_score_exclusive':40, 'official_score':None,
        'is_score_evidence':False, 'full_public_validation_passed':False,
        'technical_smoke_completed':True,
        'technical_smoke_notebook_sha256':proof['request']['notebook_sha256'],
        'technical_smoke_elapsed_seconds':report['elapsed_seconds'],
        'technical_smoke_public_mean':report['mean_public_score'],
        'engineered_source_sha256':source_sha,
        'scope':'GPU integration passed on the first source revision; later changes are diagnostic persistence and packaging, covered by CPU tests',
    }
    header = """# ARC3 — target above 40, official gateway submission

The Save & Run stage creates the official starter schema stub and its provenance
manifest. Its score field is packaging data, not a measured or predicted score.
A separate Kaggle competition rerun executes the full engineered agent against
the live gateway and determines the real score.

The first source revision completed a genuine RTX PRO 6000 smoke: all three
sampled games had inference and valid termination. Its three-minute public mean
was 2.328, which does not establish an eventual hidden score. An older full-public
run had mean42.830, partition35.166 and lowerbound31.272. No official40 is claimed.

The implementation below is guarded by KAGGLE_IS_COMPETITION_RERUN so packaging
never reloads the model or repeats a two-hour public benchmark. The official
rerun uses the full budget; startup, accounting and cleanup fixes remain active.
"""
    commit_source = """import os as _arc40_commit_os
import json as _arc40_commit_json
from pathlib import Path as _Arc40CommitPath
_ARC40_LIVE_COMPETITION = _arc40_commit_os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1','true'}
if not _ARC40_LIVE_COMPETITION:
    import pandas as _arc40_commit_pd
    _arc40_commit_dir = _Arc40CommitPath(_arc40_commit_os.environ.get('ARC40_WORKING_DIR','/kaggle/working'))
    _arc40_commit_dir.mkdir(parents=True, exist_ok=True)
    _arc40_commit_pd.DataFrame([['1_0','1',True,1]], columns=['row_id','game_id','end_of_game','score']).to_parquet(_arc40_commit_dir/'submission.parquet', index=False)
    _arc40_commit_manifest = MANIFEST_PLACEHOLDER
    (_arc40_commit_dir/'arc40_submission_manifest.json').write_text(_arc40_commit_json.dumps(_arc40_commit_manifest, indent=2))
    print('Official starter packaging created. Official score UNKNOWN; target above40. No model loaded.')
""".replace('MANIFEST_PLACEHOLDER', repr(manifest))
    cells = [dict(cell_type='markdown',metadata={},source=header),
             dict(cell_type='code',metadata={},source=commit_source,outputs=[],execution_count=None)]
    for cell in candidate['cells']:
        cell = copy.deepcopy(cell)
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.startswith('%%writefile '):
                first, content = source.split('\n',1)
                filename = first.split(None,1)[1]
                source = 'from pathlib import Path\nPath(' + repr(filename) + ').write_text(' + repr(content) + ')\n'
            cell['source'] = 'if _ARC40_LIVE_COMPETITION:\n' + '\n'.join('    '+line if line else '' for line in source.splitlines()) + '\n'
            cell['outputs'] = []
            cell['execution_count'] = None
        cells.append(cell)
    candidate['cells'] = cells
    candidate['metadata']['arc40_submission'] = manifest
    path = ROOT/'Notebook/arc3-target40-submission.ipynb'
    from IPython.core.inputtransformer2 import TransformerManager
    transformer = TransformerManager()
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            compile(transformer.transform_cell(cell['source']), f'production_cell_{index}',
                    'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    path.write_text(json.dumps(candidate,indent=1)+'\n')
    metadata = json.loads((ROOT/'Notebook/arc40-kernel-metadata.json').read_text())
    metadata['code_file'] = path.name
    launch = ROOT/'launch/arc40-submission'
    launch.mkdir(parents=True,exist_ok=True)
    (launch/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (launch/path.name).write_bytes(path.read_bytes())
    (ROOT/'Notebook/arc40-submission-kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path

if __name__ == '__main__':
    print(build())
