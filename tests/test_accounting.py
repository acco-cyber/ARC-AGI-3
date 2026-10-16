"""Exercise actual patched solver and TAAF finish methods without GPU packages.

The two source methods and real GameRun dataclass are loaded by AST so this
accounting regression can run on a CPU before installing the Kaggle wheelhouse.
Set ARC3_ACCOUNTING_SOURCE_ROOT to a patched Duck source tree when necessary.
"""
from __future__ import annotations

import ast
import asyncio
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

SOURCE_ROOT = Path(os.environ.get(
    'ARC3_ACCOUNTING_SOURCE_ROOT', str(Path(__file__).parent / 'fixtures/duck-source')))
SOLVER_PATH = SOURCE_ROOT / 'ARC3-Inference/inference/framework/solver.py'
GAME_PATH = SOURCE_ROOT / 'tufa-arc-agi-framework/src/taaf/game.py'


def _source_definitions(path, top_names=(), class_methods=None):
    tree = ast.parse(path.read_text())
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in top_names:
            selected.append(node)
        if isinstance(node, ast.ClassDef) and class_methods and node.name in class_methods:
            selected.extend(method for method in node.body
                            if isinstance(method, ast.FunctionDef) and method.name in class_methods[node.name])
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = dict(dataclass=dataclass, field=field, asyncio=asyncio, time=time)
    exec(compile(module, str(path), 'exec'), scope)
    return scope


_solver = _source_definitions(SOLVER_PATH, ('_analyzer_reported_tokens',),
    {'_HarnessGameSession': {'_finish_if_needed'}})
_game = _source_definitions(GAME_PATH, ('GameState', 'ActionRecord', 'GameRun'), {'Game': {'finish_game'}})
GameRun = _game['GameRun']
finish_session = _solver['_finish_if_needed']


class MinimalGame:
    """Use real TAAF finalization/scoring with a no-op environment close."""
    finish_game = _game['finish_game']

    def __init__(self):
        self.game_run = GameRun(game_id='cpu-test', number_of_levels=2,
            base_actions_per_level=[1, 1], state='playing')
        self.close_count = 0

    def _finish_game(self):
        self.close_count += 1


def session(*, generated=150, baseline=100, state='playing', stopped=False):
    game = MinimalGame()
    game.game_run.state = state
    stop = threading.Event()
    if stopped:
        stop.set()
    return SimpleNamespace(game=game, analyzer=SimpleNamespace(generated_tokens=generated),
                           token_baseline=baseline, stop_event=stop)


def finish(subject):
    with redirect_stdout(io.StringIO()):
        finish_session(subject)


class FinalTokenAccountingTests(unittest.TestCase):
    def test_only_unbilled_tail_is_recorded(self):
        subject = session(generated=150, baseline=100)
        finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 50)
        self.assertEqual(subject.token_baseline, 150)
        self.assertEqual(subject.game.game_run.state, 'gave_up')
        self.assertEqual(subject.game.close_count, 1)

    def test_finish_is_idempotent_even_if_analyzer_advances(self):
        subject = session()
        finish(subject)
        subject.analyzer.generated_tokens = 250
        finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 50)
        self.assertEqual(subject.token_baseline, 150)
        self.assertEqual(subject.game.close_count, 1)

    def test_cancelled_run_keeps_tail_tokens_and_cancelled_state(self):
        subject = session(stopped=True)
        finish(subject)
        self.assertEqual(subject.game.game_run.state, 'cancelled')
        self.assertEqual(subject.game.game_run.final_generated_tokens, 50)

    def test_crashed_run_keeps_tail_tokens_and_crashed_state(self):
        subject = session(state='crashed')
        finish(subject)
        self.assertEqual(subject.game.game_run.state, 'crashed')
        self.assertEqual(subject.game.game_run.final_generated_tokens, 50)

    def test_counter_reset_never_records_negative_tokens(self):
        subject = session(generated=20, baseline=100)
        finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 0)

    def test_legacy_analyzer_total_tokens_fallback(self):
        subject = session()
        subject.analyzer = SimpleNamespace(total_tokens=170)
        finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 70)

    def test_previously_finished_run_is_not_modified(self):
        subject = session()
        subject.game.game_run.final_score = 0.0
        subject.game.game_run.final_generated_tokens = 17
        finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 17)
        self.assertEqual(subject.game.close_count, 0)

    def test_no_run_is_noop(self):
        subject = session()
        subject.game.game_run = None
        finish(subject)
        self.assertEqual(subject.game.close_count, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
