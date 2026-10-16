"""Extract logged historic evidence without executing the notebook."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'Notebook/arc3-target35-dprime-validation (1).ipynb'
notebook = json.loads(SOURCE.read_text())
text = '\n'.join(''.join(o.get('text', [])) for o in notebook['cells'][23].get('outputs', []))
start = text.rfind('{\n  "candidate":')
report, _ = json.JSONDecoder().raw_decode(text[start:])
rows = []
pattern = r'^\[finished\] ([\w-]+) state=(\w+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) per-level=([^\n]+)'
for match in re.finditer(pattern, text, re.M):
    gid, state, completed, levels, score, actions, tokens, suffix = match.groups()
    completed, levels = int(completed), int(levels)
    display_score = float(score)
    completion_cap = completed*(completed+1)/(levels*(levels+1))*100
    # The notebook persisted summary metrics, but printed game scores to 2dp.
    # Clamp only displayed rounding above the theoretical level cap; never
    # present these rounded per-game reconstructions as original exact floats.
    conservative_score = min(display_score, completion_cap)
    note = re.search(r'note="tokens=(\d+)"', suffix)
    rows.append({'game_id':gid, 'state':state, 'levels_completed':completed,
                 'total_levels':levels, 'score':conservative_score,
                 'printed_score_2dp':display_score, 'actions':int(actions),
                 'generated_tokens':int(tokens),
                 'final_note_tokens':int(note.group(1)) if note else None,
                 'logged_per_level_action_baselines':suffix.split(' note=',1)[0]})
assert len(rows)==25 and len({r['game_id'] for r in rows})==25
sys.path.insert(0, str(ROOT/'scripts'))
from arc40_evaluation import assess_public_run
reevaluated = assess_public_run(rows, [r['game_id'] for r in rows], target=40.0,
    evaluation_kind='full_public', elapsed_seconds=report['elapsed_seconds'],
    budget_seconds=9*60*60, split_seed=report['split_seed'],
    bootstrap_seed=20261015)
combined = '\n\n'.join(''.join(c['source']) for c in notebook['cells'])
result = {
    'evidence_kind':'Recovered notebook stream outputs; historical public offline run only',
    'source_path':str(SOURCE),
    'source_url':'https://github.com/acco-cyber/ARC-AGI-3/blob/main/Notebook/arc3-target35-dprime-validation%20(1).ipynb',
    'notebook_file_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'all_cell_sources_joined_sha256':hashlib.sha256(combined.encode()).hexdigest(),
    'harness_patch_cell_sha256':hashlib.sha256(''.join(notebook['cells'][4]['source']).encode()).hexdigest(),
    'original_upstream_source_sha256_from_metadata':notebook['metadata']['arc35']['base_source_sha256'],
    'timestamps':{
        'notebook_start_utc':notebook['metadata']['papermill']['start_time'],
        'notebook_end_utc':notebook['metadata']['papermill']['end_time'],
        'notebook_start_kst':'2026-10-15T05:16:04.835430+09:00',
        'notebook_end_kst':'2026-10-15T07:28:10.611917+09:00',
        'benchmark_started_utc':'2026-10-14T20:25:56+00:00',
        'benchmark_ended_utc':'2026-10-14T22:28:05+00:00',
    },
    'run_mode':'Offline all-25 public validation; not a Kaggle competition rerun',
    'papermill_exception':notebook['metadata']['papermill']['exception'],
    'exception_reason':'RuntimeError: Local target-35 validation did not pass. See arc35_validation.json; do not submit.',
    'authoritative_summary_printed_at_end':report,
    'runtime_evidence':{
        'sglang':'0.5.19+gd00d88efc8d6','torch':'2.13.0+cu130',
        'python':'3.12.13','gpu':'NVIDIA RTX PRO 6000 Blackwell Server Edition',
        'gpu_memory_mib':97887,'capability':[12,0], 'ready_after_seconds':541,
        'target_weight_shards':38,'draft_weight_shards':2,
        'stale_notebook_is_gpu_enabled_field':notebook['metadata']['kaggle']['isGpuEnabled'],
        'stale_metadata_warning':'Actual nvidia-smi, torch GPU and /health outputs show GPU use despite stale isGpuEnabled=false.',
    },
    'recovered_rows_quality':{
        'games':len(rows),'per_game_score_precision':'Only printed 2-decimal scores available; exact aggregate metrics are preserved above.',
        'all_have_inference':all(r['generated_tokens']>0 for r in rows),
        'all_normalized_terminal_without_crash':all(r['state'] in {'won','cancelled'} for r in rows),
        'won':sum(r['state']=='won' for r in rows),'deadline_cancelled':sum(r['state']=='cancelled' for r in rows),
        'logged_generated_tokens_sum':sum(r['generated_tokens'] for r in rows),
        'last_request_note_tokens_sum':sum(r['final_note_tokens'] or 0 for r in rows),
    },
    'reassessment_above_40_using_reconstructed_rounded_rows':reevaluated,
    'interpretation':[
        'The historic public mean was above40, but the predeclared heldout mean35.166 and bootstrap lower31.272 did not support even the earlier35 gate.',
        'This is evidence that the pinned model/runtime performed real GPU inference; it does not validate the newly engineered candidate or its score.',
        'Old printed per-game token fields omit some cancelled analyzer tokens, as shown by larger final note totals; accounting should be fixed before fresh comparison.',
        'The stdlib reassessment bootstrap differs from the original NumPy RNG. Only the original printed aggregate report is exact historic evidence.',
        'No official40 score, official ranking, speed improvement or automatic submission authorization follows from this recovery.',
    ],
    'reconstructed_game_rows':rows,
}
OUT=ROOT/'runs/2026-10-16T1225KST/historical-public-evidence.json'
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(OUT),'original_metrics':report,'recovered_quality':result['recovered_rows_quality'],
                 'new_checks':reevaluated['checks'],'reassessment_invalid_reasons':reevaluated['invalid_reasons']},indent=2))
