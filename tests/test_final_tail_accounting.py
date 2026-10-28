"""CPU regressions against source-derived public Duck/TAAF finalizers."""
from __future__ import annotations

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import functools
import io
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from final_tail_accounting import install
import test_accounting as pinned


def module_fixture(*, legacy=True):
    source = pinned.SOLVER_PATH.read_text()
    tree = ast.parse(source)
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if method.name in {"_finish_if_needed", "_finish_remaining", "_run_games", "_drain_game_tasks"}:
                        methods[method.name] = textwrap.dedent(ast.get_source_segment(source, method))
    if legacy:
        lines = methods["_finish_if_needed"].splitlines()
        cut = next(i for i, line in enumerate(lines) if "# Actions bill" in line)
        methods["_finish_if_needed"] = "\n".join(lines[:cut]) + "\n        self.game.finish_game()\n"
    temp = tempfile.TemporaryDirectory(prefix="arc3-final-tail-cpu-")
    file = Path(temp.name) / "public_finalizers.py"
    code = "from __future__ import annotations\n\n" + "\n\n".join(methods.values()) + "\n"
    file.write_text(code)
    scope = dict(asyncio=asyncio, functools=functools,
                 _analyzer_reported_tokens=pinned._solver["_analyzer_reported_tokens"])
    exec(compile(code, str(file), "exec"), scope)
    session_cls = type("_HarnessGameSession", (), {"_finish_if_needed": scope["_finish_if_needed"]})
    solver_cls = type("HarnessSolver", (), {k: v for k, v in scope.items()
                      if k in {"_finish_remaining", "_run_games", "_drain_game_tasks"}})
    module = SimpleNamespace(_HarnessGameSession=session_cls, HarnessSolver=solver_cls,
                             _analyzer_reported_tokens=scope["_analyzer_reported_tokens"])
    module._fixture_temp = temp
    return module


class TailAccountingTests(unittest.TestCase):
    def setUp(self):
        self.module = module_fixture()
        self.addCleanup(self.module._fixture_temp.cleanup)
        self.assertTrue(install(self.module))

    def finish(self, subject):
        with redirect_stdout(io.StringIO()):
            self.module._HarnessGameSession._finish_if_needed(subject)

    def test_normal_tail_and_history_score_preserved(self):
        subject = pinned.session()
        run = subject.game.game_run
        record = pinned._game["ActionRecord"](
            action=SimpleNamespace(id="ACTION1", data={}), generated_tokens=100,
            uncached_input_tokens=0, wallclock_seconds=.1)
        run.history.append(record)
        run.actions_per_level[0] = 1
        run.levels_completed = 1
        history = run.history
        self.finish(subject)
        run = subject.game.game_run
        self.assertEqual(run.final_generated_tokens, 50)
        self.assertIs(run.history, history)
        self.assertEqual(run.actions_per_level, [1, 0])
        self.assertEqual(run.history, [record])
        self.assertEqual(sum(r.generated_tokens for r in run.history) + run.final_generated_tokens, 150)
        self.assertEqual(run.final_score, run._compute_final_score())
        self.assertEqual(subject.game.close_count, 1)

    def test_normal_finish_stays_idempotent_if_counter_advances(self):
        subject = pinned.session()
        self.finish(subject)
        subject.analyzer.generated_tokens = 500
        self.finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 50)
        self.assertEqual(subject.token_baseline, 150)
        self.assertEqual(subject.game.close_count, 1)

    def test_cancelled_and_crashed_normal_finish(self):
        for state, stopped, expected in [("playing", True, "cancelled"), ("crashed", False, "crashed")]:
            subject = pinned.session(state=state, stopped=stopped)
            self.finish(subject)
            self.assertEqual(subject.game.game_run.state, expected)
            self.assertEqual(subject.game.game_run.final_generated_tokens, 50)

    def test_legacy_counter_and_counter_reset(self):
        subject = pinned.session()
        subject.analyzer = SimpleNamespace(total_tokens=170)
        self.finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 70)
        subject = pinned.session(generated=20)
        self.finish(subject)
        self.assertEqual(subject.game.game_run.final_generated_tokens, 0)

    def test_no_run_and_previously_finished_are_not_changed(self):
        subject = pinned.session()
        subject.game.game_run = None
        self.finish(subject)
        self.assertEqual(subject.game.close_count, 0)
        subject = pinned.session(stopped=True)
        run = subject.game.game_run
        run.final_score, run.final_generated_tokens, run.state = 0.0, 17, "cancelled"
        self.finish(subject)
        self.assertEqual(run.final_generated_tokens, 17)
        self.assertEqual(subject.game.close_count, 0)

    def test_external_close_reconciles_once_preserving_score_state_and_close_count(self):
        subject = pinned.session(stopped=True)
        host = self.module.HarnessSolver()
        host._stop_event = subject.stop_event
        with redirect_stdout(io.StringIO()):
            host._finish_remaining([subject.game])
        run = subject.game.game_run
        score, state, history = run.final_score, run.state, list(run.history)
        self.assertTrue(run._arc3_external_tail_pending)
        self.finish(subject)
        self.assertFalse(run._arc3_external_tail_pending)
        self.assertEqual(run.final_generated_tokens, 50)
        self.assertEqual((run.final_score, run.state, run.history), (score, state, history))
        subject.analyzer.generated_tokens = 300
        self.finish(subject)
        self.assertEqual(run.final_generated_tokens, 50)
        self.assertEqual(subject.game.close_count, 1)

    def test_already_finished_run_never_gets_external_marker(self):
        subject = pinned.session(stopped=True)
        self.finish(subject)
        host = self.module.HarnessSolver()
        host._stop_event = subject.stop_event
        host._finish_remaining([subject.game])
        self.assertFalse(getattr(subject.game.game_run, "_arc3_external_tail_pending", False))

    def test_actual_async_cancellation_drain_then_late_worker(self):
        async def scenario():
            host = self.module.HarnessSolver()
            host._stop_event = threading.Event()
            host.concurrency = 1
            host.cancel_drain_timeout_s = .01
            host._worker_pool = ThreadPoolExecutor(max_workers=1)
            host._local_server_for_game_index = lambda _: None
            subject = pinned.session()
            subject.stop_event = host._stop_event
            entered, release = threading.Event(), threading.Event()
            def play_one(*args):
                entered.set()
                release.wait(timeout=2)
                self.finish(subject)
            host._play_one = play_one
            task = asyncio.create_task(host._run_games([subject.game]))
            try:
                for _ in range(1000):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.001)
                self.assertTrue(entered.is_set())
                task.cancel()
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertEqual(subject.game.game_run.final_generated_tokens, 0)
                release.set()
            finally:
                release.set()
                host._worker_pool.shutdown(wait=True)
            self.assertEqual(subject.game.game_run.final_generated_tokens, 50)
            self.assertEqual(subject.game.game_run.state, "cancelled")
            self.assertEqual(subject.game.close_count, 1)
        asyncio.run(scenario())

    def test_fail_closed_and_install_idempotence(self):
        self.assertFalse(install(self.module))
        bad = module_fixture()
        self.addCleanup(bad._fixture_temp.cleanup)
        bad._HarnessGameSession._finish_if_needed = lambda self: None
        with self.assertRaises(RuntimeError):
            install(bad)
        self.assertFalse(getattr(bad, "_arc3_final_tail_accounting_installed", False))

    def test_accepts_existing_basic_tail_patch(self):
        patched = module_fixture(legacy=False)
        self.addCleanup(patched._fixture_temp.cleanup)
        self.assertTrue(install(patched))

    def test_foreign_wraps_does_not_bypass_source_guard(self):
        for owner_name, method_name in [("_HarnessGameSession", "_finish_if_needed"),
                                        ("HarnessSolver", "_finish_remaining")]:
            foreign = module_fixture()
            self.addCleanup(foreign._fixture_temp.cleanup)
            owner = getattr(foreign, owner_name)
            original = getattr(owner, method_name)
            @functools.wraps(original)
            def changed(*args, **kwargs):
                raise AssertionError("foreign behavior")
            setattr(owner, method_name, changed)
            with self.assertRaises(RuntimeError):
                install(foreign)
            self.assertIs(getattr(owner, method_name), changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
