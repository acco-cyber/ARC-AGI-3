"""Bounded transport/cache probe; never starts a server or touches a game gateway.

This measures 25 versus 110 pending client states behind ten HTTP slots. It does
not reproduce D-prime scheduling, gameplay, image preprocessing, or hidden score.
Run only in an isolated subprocess, against the already loaded local server.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProbeConfig:
    populations: tuple[int, ...] = (25, 110, 110, 25)
    active_slots: int = 10
    requests_per_phase: int = 64
    input_tokens: tuple[int, ...] = (2048, 8192, 16384)
    max_output_tokens: int = 64
    phase_seconds: float = 60.0
    request_timeout_seconds: float = 20.0
    seed: int = 20261017

    def validate(self):
        if self.populations != (25, 110, 110, 25):
            raise ValueError('Use the fixed ABBA population order for this diagnostic')
        if self.active_slots != 10:
            raise ValueError('Preserve the existing ten running requests')
        if not 61 <= self.requests_per_phase <= 128:
            raise ValueError('Use61..128 requests to cross the pinned60-state Mamba cache')
        if max(self.input_tokens) > 114688 or min(self.input_tokens) < 1024:
            raise ValueError('Probe prompts must stay within the pinned context')
        if not 1 <= self.max_output_tokens <= 1024:
            raise ValueError('Bound output tokens to1..1024')
        if not 0 < self.request_timeout_seconds <= self.phase_seconds <= 90:
            raise ValueError('Request/phase deadlines must be positive and bounded')
        return self


def local_url(base_url):
    parsed = urlsplit(base_url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('The probe may call only the already running loopback HTTP server')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Credentials, queries and fragments are not permitted')
    return base_url.rstrip('/')


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo, hi = int(index), min(len(values) - 1, int(index) + 1)
    return values[lo] * (hi - index) + values[hi] * (index - lo) if lo != hi else values[lo]


def shape_plan(config):
    """The same admitted request lengths/count/output cap in every phase."""
    config.validate()
    return [dict(case=i, target_input_tokens=config.input_tokens[i % len(config.input_tokens)],
                 max_output_tokens=config.max_output_tokens,
                 seed=config.seed + i) for i in range(config.requests_per_phase)]


def chat_template_token_count(rendered):
    """Count one tokenized conversation across Transformers return shapes.

    Recent tokenizers can return a BatchEncoding mapping with ``input_ids``
    and ``attention_mask`` even without requesting tensors. Counting mapping
    fields (or a batch dimension) silently reports two (or one) tokens. Accept
    flat IDs and a single batch, including tensor/array values via ``tolist``;
    reject multiple conversations rather than treating them as token IDs.
    """
    if isinstance(rendered, Mapping):
        if 'input_ids' not in rendered:
            raise ValueError('Tokenized chat-template mapping is missing input_ids')
        rendered = rendered['input_ids']
    if callable(getattr(rendered, 'tolist', None)):
        rendered = rendered.tolist()
    if not isinstance(rendered, (list, tuple)):
        raise TypeError('Tokenized chat template must contain an input_ids sequence')
    if rendered and isinstance(rendered[0], (list, tuple)):
        if len(rendered) != 1:
            raise ValueError('Expected one tokenized conversation, received multiple batches')
        rendered = rendered[0]
    if any(isinstance(token, bool) or not isinstance(token, Integral) for token in rendered):
        raise TypeError('Tokenized chat-template input_ids must be a flat integer sequence')
    return len(rendered)


def build_payload(tokenizer, case, phase, model, population):
    """Unique prefix per phase/case avoids second-phase exact-prefix advantage.

    Match the template-rendered token length, not repeated character length.
    Contents are synthetic and cannot execute anything outside this probe.
    """
    target = case['target_input_tokens']
    nonce = hashlib.sha256(f"arc-load/{phase}/{case['case'] % population}".encode()).hexdigest()
    system = ('Synthetic transport measurement. Read the following numbered board notes, '
              'reason briefly about them, and emit a short summary. Do not call tools. '
              f'Independent case {nonce}.')
    filler = tokenizer.encode(' board row column zero one two three four five six seven eight nine',
                              add_special_tokens=False)
    prefix = tokenizer.encode(f'Independent synthetic state {nonce}.\n', add_special_tokens=False)
    pad = (filler * ((target // len(filler)) + 2))[:target]
    tokens = prefix + pad

    def messages(n):
        return [{'role': 'system', 'content': system},
                {'role': 'user', 'content': tokenizer.decode(tokens[:n])}]

    # Template overhead varies slightly with tokenizer. Remove it once and
    # correct any tokenization boundary changes with a tiny bounded search.
    n = max(1, target - len(prefix) - 100)
    msgs = messages(n)
    rendered = tokenizer.apply_chat_template(msgs, tokenize=True,
        add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
    n = min(len(tokens), max(1, n + target - chat_template_token_count(rendered)))
    for _ in range(8):
        msgs = messages(n)
        actual = chat_template_token_count(tokenizer.apply_chat_template(msgs, tokenize=True,
            add_generation_prompt=True, enable_thinking=True, preserve_thinking=True))
        if actual == target:
            break
        n = min(len(tokens), max(1, n + target - actual))
    if abs(actual - target) > 8:
        raise RuntimeError(f'Tokenizer could not match prompt length: wanted{target}, got{actual}')
    payload = {'model': model, 'messages': msgs, 'max_tokens': case['max_output_tokens'],
               'temperature': 0.7, 'top_p': 0.95, 'top_k': 20, 'seed': case['seed'],
               'stream': False,
               'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}}
    return payload, actual


def run_phase(population, payloads, config, post, *, clock=time.monotonic):
    """No inference starts after the deadline; all waiting states are drained.

    A socket read timeout is not a whole-request wall timeout, so the notebook
    must enforce the separate hard subprocess deadline as well.
    """
    config.validate()
    if len(payloads) != config.requests_per_phase:
        raise ValueError('Incorrect fixed request manifest')
    start_gate = threading.Barrier(population + 1)
    slots = threading.BoundedSemaphore(config.active_slots)
    lock = threading.Lock()
    start_at = None
    deadline = None
    active = 0
    peak_active = 0
    rows = []

    def worker(state):
        nonlocal active, peak_active
        start_gate.wait(timeout=10)
        indices = list(range(state, len(payloads), population))
        # Empty clients still enter the ten-slot waiting population once.
        for index in indices or [None]:
            queued_at = clock()
            acquired = False
            while clock() < deadline:
                if slots.acquire(timeout=min(0.1, max(0.001, deadline-clock()))):
                    acquired = True
                    break
            if not acquired:
                return
            try:
                admitted = clock()
                if admitted >= deadline or index is None:
                    return
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                payload, template_tokens = payloads[index]
                row = {'state': state, 'case': index, 'template_tokens': template_tokens,
                       'queue_seconds': admitted-queued_at, 'http_seconds': None,
                       'completion_tokens': 0, 'prompt_tokens': None, 'ok': False}
                try:
                    timeout = max(0.05, min(config.request_timeout_seconds, deadline-clock()))
                    response = post(payload, timeout)
                    row.update(response)
                    prompt = response.get('prompt_tokens')
                    completion = response.get('completion_tokens')
                    row['usage_valid'] = (isinstance(prompt, int) and prompt > 0
                        and isinstance(completion, int) and 0 < completion <= config.max_output_tokens)
                    row['input_length_matched'] = (isinstance(prompt, int)
                        and abs(prompt-template_tokens) <= 16)
                    row['ok'] = bool(response.get('ok')) and row['usage_valid'] and row['input_length_matched']
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                    row['error'] = str(exc)[:500]
                finally:
                    row['http_seconds'] = clock()-admitted
                    with lock:
                        active -= 1
                        rows.append(row)
            finally:
                slots.release()

    with ThreadPoolExecutor(max_workers=population, thread_name_prefix='load-state') as executor:
        futures = [executor.submit(worker, i) for i in range(population)]
        start_at = clock()
        deadline = start_at + config.phase_seconds
        start_gate.wait(timeout=10)
        wait(futures)
        for future in futures:
            future.result()
    elapsed = clock()-start_at
    successes = [row for row in rows if row['ok']]
    return {'population': population, 'active_slots': config.active_slots,
            'peak_active_http': peak_active, 'requested': len(payloads),
            'attempted': len(rows), 'completed': len(successes),
            'valid_transport_comparison': len(successes) == len(payloads),
            'elapsed_seconds': elapsed,
            'completion_tokens': sum(row.get('completion_tokens') or 0 for row in successes),
            'aggregate_completion_tokens_per_second':
                sum(row.get('completion_tokens') or 0 for row in successes)/max(0.001, elapsed),
            'http_p50_seconds': quantile([r['http_seconds'] for r in successes], 0.5),
            'http_p95_seconds': quantile([r['http_seconds'] for r in successes], 0.95),
            'queue_p95_seconds': quantile([r['queue_seconds'] for r in rows], 0.95),
            'distinct_states_served': len({r['state'] for r in rows}),
            'unserved_states': population-len({r['state'] for r in rows}),
            'note': 'Unserved states are pending-load scaffolding, not failed or solved games',
            'rows': sorted(rows, key=lambda r: r['case'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--model', default='flashnext')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = local_url(args.base_url)
    root = base[:-3] if base.endswith('/v1') else base
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    import requests
    config = ProbeConfig().validate()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=True)
    plan = shape_plan(config)
    # Preparing all inputs happens before phase timing; no model weights load.
    payloads = [[build_payload(tokenizer, case, phase, args.model, config.populations[phase]) for case in plan]
                for phase in range(len(config.populations))]
    manifest = {'scope': __doc__, 'config': asdict(config), 'request_plan': plan,
                'official_score': None, 'authorizes_submission': False,
                'weights_loaded_by_probe': False, 'server_launched_by_probe': False}
    (output/'probe_manifest.json').write_text(json.dumps(manifest, indent=2))
    def snapshot(name):
        try:
            response = requests.get(root+'/metrics', timeout=(2, 3))
            response.raise_for_status()
            (output/(name+'.prom')).write_text(response.text)
        except Exception as exc:
            (output/(name+'.error.txt')).write_text(type(exc).__name__+': '+str(exc))
    def post(payload, timeout):
        response = requests.post(base+'/chat/completions', json=payload,
                                 timeout=(min(2, timeout), timeout))
        response.raise_for_status()
        body = response.json()
        usage = body.get('usage') or {}
        return {'ok': bool(body.get('choices')), 'http_status': response.status_code,
                'completion_tokens': int(usage.get('completion_tokens') or 0),
                'prompt_tokens': int(usage.get('prompt_tokens') or 0),
                'finish_reason': (body.get('choices') or [{}])[0].get('finish_reason')}
    warm_payload, _ = build_payload(tokenizer,
        {'case': 0, 'target_input_tokens': 1024, 'max_output_tokens': 16, 'seed': config.seed},
        -1, args.model, 1)
    warm_start = time.monotonic()
    warm = post(warm_payload, 60.0)
    manifest['warmup'] = {**warm, 'elapsed_seconds': time.monotonic()-warm_start}
    (output/'probe_manifest.json').write_text(json.dumps(manifest, indent=2))
    if not warm['ok']:
        raise RuntimeError('Warmup returned no choices; refusing timed phases')
    reports = []
    started = time.monotonic()
    for i, population in enumerate(config.populations):
        snapshot(f'phase{i}-before')
        report = run_phase(population, payloads[i], config, post)
        report['phase'] = i
        reports.append(report)
        (output/f'phase{i}.json').write_text(json.dumps(report, indent=2))
        snapshot(f'phase{i}-after')
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
        if not report['valid_transport_comparison']:
            # Fail fast: more load would consume GPU without a comparable pair.
            break
    metrics_count = len(list(output.glob('phase*.prom')))
    result = {**manifest, 'phases': reports, 'elapsed_seconds': time.monotonic()-started,
              'raw_metrics_snapshot_count': metrics_count,
              'cache_metrics_available': metrics_count == 2*len(reports),
              'valid_transport_comparison': len(reports) == 4 and all(
                  r['valid_transport_comparison'] for r in reports),
              'limitations': ['Synthetic text; no game policies, image decoding or hidden score',
                             'FIFO ten-slot transport gate; D-prime scheduler is not reproduced',
                             '25-state prefixes are reused;110-state phase admits64 unique states',
                             'Repeated synthetic prefixes isolate working-set cache cost; real game histories differ',
                             'Short2K/8K/16K probe cannot establish near-limit128K or long-run stability',
                             'Compare only phases whose returned prompt/output distributions match']}
    (output/'probe_summary.json').write_text(json.dumps(result, indent=2))
    return 0 if result['valid_transport_comparison'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
