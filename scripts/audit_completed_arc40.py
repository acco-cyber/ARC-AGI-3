"""Summarize measured ARC3 artifacts on CPU; never launch or submit."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/2026-10-17'


def main():
    validation = json.loads((RUN/'full-validation/arc40_validation.json').read_text())
    benchmark = json.loads((RUN/'full-validation/benchmark.json').read_text())
    submissions = json.loads((RUN/'all-submissions.json').read_text())
    prior = json.loads((ROOT/'runs/2026-10-16T1225KST/historical-public-evidence.json').read_text())
    games = validation['games']
    total_tokens = sum(g['generated_tokens'] for g in games)
    score_sum = sum(g['score'] for g in games)
    duration = (datetime.fromisoformat(benchmark['end_time'])-datetime.fromisoformat(benchmark['start_time'])).total_seconds()
    finished = Counter(g['state'] for g in games)
    groups = {}
    for label, selected in [('won',[g for g in games if g['state']=='won']),
                            ('cancelled',[g for g in games if g['state']=='cancelled']),
                            ('below15',[g for g in games if g['score']<15])]:
        tokens=sum(g['generated_tokens'] for g in selected)
        score=sum(g['score'] for g in selected)
        groups[label]={'games':len(selected),'tokens':tokens,'token_fraction':tokens/total_tokens,
                       'summed_score':score,'score_fraction':score/score_sum,
                       'tokens_per_summed_score_point':tokens/score if score else None}
    official=[g for g in submissions if g.get('publicScore') not in (None,'') and g.get('status','').lower()=='complete']
    best=max(official,key=lambda g:float(g['publicScore']))
    old=prior['authoritative_summary_printed_at_end']
    summary={
        'official_best_score':float(best['publicScore']),'official_submission_ref':best['ref'],
        'official_submission_date_utc':best['date'],'official_url':best['url'],
        'improvement_over_previously_observed_official_27_22':float(best['publicScore'])-27.22,
        'audited_submission_count':len(submissions),
        'submission_states':dict(Counter(g['status'].lower() for g in submissions)),
        'public_mean':validation['mean_public_score'],
        'public_partition_mean':validation['heldout_mean_public_score'],
        'public_bootstrap_lower':validation['public_game_bootstrap_lower_95'],
        'public_gate_passed':validation['local_validation_passed'],
        'official_35_confirmed':float(best['publicScore'])>=35,
        'official_40_confirmed':float(best['publicScore'])>40,
        'state_counts':dict(finished),'generated_tokens':total_tokens,
        'unbilled_tail_tokens_recorded':sum(g['final_generated_tokens'] for g in benchmark['game_runs']),
        'benchmark_seconds':duration,'whole_run_seconds':validation['elapsed_seconds'],
        'aggregate_benchmark_tokens_per_second':total_tokens/duration,
        'aggregate_including_startup_tokens_per_second':total_tokens/validation['elapsed_seconds'],
        'throughput_scope':'All concurrent public games; not single-request decoding speed',
        'public_mean_difference_from_prior':validation['mean_public_score']-old['mean_public_score'],
        'wall_time_difference_seconds':validation['elapsed_seconds']-old['elapsed_seconds'],
        'public_split_changed':validation['split_seed']!=old['split_seed'],
        'no_causal_quality_or_speed_gain_claim':True,
        'token_groups':groups,
        'submission_limits':json.loads((RUN/'submission-limits.json').read_text()),
        'new_gpu_runs_this_check':0,'new_official_submission_attempts_this_check':0,
        'next_hypothesis':'Game-independent cap on consecutive generated tokens without an emitted action; evaluate only with paired fixed-budget baseline, not per-game IDs or stored answers',
        'hypothesis_is_validated':False,
        'validation_source_sha256':hashlib.sha256((RUN/'full-validation/arc40_validation.json').read_bytes()).hexdigest(),
    }
    (RUN/'monitoring-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
