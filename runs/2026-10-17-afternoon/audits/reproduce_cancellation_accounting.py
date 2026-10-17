"""CPU-only reproduction using the pinned solver/TAAF methods, without GPU imports.

Run from the workspace root:
    python runs/2026-10-17-afternoon/audits/reproduce_cancellation_accounting.py
"""
from __future__ import annotations

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'tests'))
import test_accounting as accounting

source = (REPO / 'tests/fixtures/duck-source/ARC3-Inference/inference/framework/solver.py').read_text()
tree = ast.parse(source)
solver_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'HarnessSolver')
method_names = {'_run_games', '_drain_game_tasks', '_finish_remaining'}
methods = [node for node in solver_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names]
module = ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
    *methods,
], type_ignores=[]))
scope = {'asyncio': asyncio, 'functools': functools}
exec(compile(module, 'actual-solver-cancellation-methods', 'exec'), scope)


class Host:
    pass


for method_name in method_names:
    setattr(Host, method_name, scope[method_name])


async def reproduce():
    host = Host()
    host._stop_event = threading.Event()
    host.concurrency = 1
    host.cancel_drain_timeout_s = 0.01  # Speed up the production 120-second branch.
    host._worker_pool = ThreadPoolExecutor(max_workers=1)
    host._local_server_for_game_index = lambda _: None
    subject = accounting.session(generated=150, baseline=100)
    entered, release = threading.Event(), threading.Event()

    def play_one(*args):
        entered.set()
        release.wait(timeout=1)
        accounting.finish_session(subject)

    host._play_one = play_one
    task = asyncio.create_task(host._run_games([subject.game]))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    before = subject.game.game_run.final_generated_tokens
    release.set()
    await asyncio.sleep(0.02)
    host._worker_pool.shutdown(wait=True)
    after = subject.game.game_run.final_generated_tokens
    result = {
        'test': 'Actual solver cancellation drain expires before worker return',
        'drain_timeout_test_seconds': 0.01,
        'production_default_seconds': 120.0,
        'unbilled_analyzer_tail': 50,
        'final_tokens_at_forced_finish': before,
        'final_tokens_after_worker_finish': after,
        'risk_confirmed': before == 0 and after == 0,
    }
    assert result['risk_confirmed']
    (HERE / 'cancellation-repro.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(reproduce())
