"""CPU checks of lease behavior and the real embedded gate/D-prime code."""
from __future__ import annotations

import copy
import ast
import heapq
import json
import logging
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from tempfile import TemporaryDirectory

from no_action_handover import NoActionLease, install
from build_no_action_candidate import build


HERE = Path(__file__).resolve().parent


def embedded_fixtures():
    """Read the frozen repo notebook; never keep a duplicate upstream fixture."""
    candidates = [HERE.parent / "Notebook/arc3-target40-engineered.ipynb",
                  HERE.parent / "ARC-AGI-3/Notebook/arc3-target40-engineered.ipynb"]
    notebook = next((path for path in candidates if path.exists()), None)
    if notebook is None:
        raise FileNotFoundError("Frozen engineered notebook is required for source-derived tests")
    cells = json.loads(notebook.read_text())["cells"]
    sources = ["".join(cell["source"]) if isinstance(cell.get("source"), list)
               else cell.get("source", "") for cell in cells]
    patch = next(source for source in sources if source.startswith("%%writefile ")
                 and "diff --git a/ARC3-Inference/inference/agent/tool_agent.py " in source)
    parts = patch.split("diff --git ")

    def new_text(name):
        part = next(part for part in parts if part.startswith(
            f"a/ARC3-Inference/inference/agent/{name} b/"))
        return "\n".join(line[1:] for line in part.splitlines()
                         if line.startswith(("+", " ")) and not line.startswith("+++")) + "\n"

    form_cell = next(source for source in sources if "_FP_SOURCE = " in source
                     and "_fp_mod.install_d(_fp_ta, _fp_solver)" in source)
    form_source = next(ast.literal_eval(node.value) for node in ast.parse(form_cell).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "_FP_SOURCE"
                               for target in node.targets))
    fixture_modules = []
    for name, source in (("arc35_fixture_priority", new_text("priority_scheduler.py")),
                         ("arc35_fixture_form", form_source)):
        module = types.ModuleType(name)
        sys.modules[name] = module
        exec(compile(source, str(notebook) + ":" + name, "exec"), module.__dict__)
        fixture_modules.append(module)
    return new_text("tool_agent.py"), *fixture_modules


EMBEDDED_TOOL_SOURCE, _priority_fixture, _form_fixture = embedded_fixtures()
PrioritySnapshot = _priority_fixture.PrioritySnapshot
FormPace, d_priority = _form_fixture.FormPace, _form_fixture.d_priority


def real_gate_namespace():
    # This class body is contiguous in the actual embedded patch. No package
    # import/model/runtime dependencies or rewritten scheduler are needed.
    source = EMBEDDED_TOOL_SOURCE
    start = source.index("class _PriorityGate:")
    end = source.index("\ndef _max_active_streams", start)
    scope = dict(threading=threading, time=time, heapq=heapq,
                 PrioritySnapshot=PrioritySnapshot, priority_value=d_priority,
                 _priority_refresh_enabled=lambda: True,
                 _get_env_bool=lambda key, default=False: False,
                 _endgame_start_minutes=lambda: 0,
                 _priority_human_actions=lambda: 25,
                 _priority_action_scale=lambda: 115,
                 _priority_token_scale=lambda: 62000,
                 _priority_action_weight=lambda: .25,
                 _priority_tail_base=lambda: 2,
                 _priority_tail_cap=lambda: 4,
                 _priority_level_options=lambda: {},
                 _GATE_STATS=dict(enqueued=0, blocked=0, blocked_ms=0, handover_ms=0),
                 log=logging.getLogger("test"))
    exec(compile(source[start:end], "embedded_real_priority_gate.py", "exec"), scope)
    return scope


def module_and_agent(gate=None):
    class Agent:
        def __init__(self):
            self._context_was_trimmed = False
            self._session_generated_tokens = 0
            self._last_step_summary = {"end_action_num": 10, "level": 2}
            self._actions_at_level_start = 5
            self._tokens_at_level_start = 0
            self._progress_pace = FormPace()
            self._progress_pace.record_completion(30000)
            self._priority_current, self._priority_next = 1012345, 1012345
            self._history_messages = [{"role": "assistant", "content": "kept reasoning"}]
            self._kept_functions = {"search": "exact retained definition"}
            self._summarized_knowledge = {"world_model": "exact retained facts"}

        def _maybe_handover(self):
            if self._context_was_trimmed:
                self._context_was_trimmed = False
                if gate is not None:
                    gate.handover(123, PrioritySnapshot(2, 5, 123))
            return "original-return"

        def _priority_total_levels(self):
            return 8

    module = types.SimpleNamespace(
        ToolAgent=Agent, PrioritySnapshot=PrioritySnapshot,
        _priority_gate=lambda: gate,
        _priority_action_count=lambda summary: int(summary.get("end_action_num") or 0),
        _priority_refresh_enabled=lambda: True,
        _priority_level_options=lambda: {"normalize_score": True, "tail_lookup": "remaining"},
        _get_env_bool=lambda key, default=False: key == "ARC3_PRIORITY_PACE")
    return module, Agent()


class FakeGate:
    def __init__(self):
        self.calls = []

    def _snapshot_priority(self, snapshot, now):
        return d_priority(snapshot)

    def handover(self, priority, snapshot):
        self.calls.append((priority, snapshot))


class LeaseTests(unittest.TestCase):
    def test_action_resets_and_yields_do_not(self):
        lease = NoActionLease(32768)
        self.assertFalse(lease.observe(0, 0))
        self.assertFalse(lease.observe(20000, 0))
        self.assertTrue(lease.observe(33000, 0))
        lease.handed_over(33000)
        self.assertFalse(lease.observe(65000, 0))
        self.assertTrue(lease.observe(66000, 0))
        self.assertFalse(lease.observe(70000, 1))
        self.assertFalse(lease.observe(102767, 1))
        self.assertTrue(lease.observe(102768, 1))

    def test_ordinary_trim_and_rollback_rebase(self):
        lease = NoActionLease(32768)
        lease.observe(0, 1)
        self.assertFalse(lease.observe(50000, 1, ordinary_handover=True))
        self.assertFalse(lease.observe(10000, 1))
        self.assertFalse(lease.observe(42767, 1))
        self.assertTrue(lease.observe(42768, 1))

    def test_disabled_is_literal_baseline(self):
        module, agent = module_and_agent()
        before = module.ToolAgent._maybe_handover
        stats = install(module)
        self.assertIs(module.ToolAgent._maybe_handover, before)
        self.assertEqual(agent._maybe_handover(), "original-return")
        self.assertNotIn("_arc35_no_action_lease", vars(agent))
        self.assertFalse(stats.snapshot()["enabled"])

    def test_invalid_config_and_duplicate_install(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                install(types.SimpleNamespace(), token_quantum=value)
        module, _ = module_and_agent()
        stats = install(module, token_quantum=32768)
        self.assertIs(install(module, token_quantum=32768), stats)
        with self.assertRaises(RuntimeError):
            install(module, token_quantum=65536)
        with self.assertRaises(RuntimeError):
            install(module, token_quantum=0)

    def test_only_admission_changes_context_priority_and_scoring_preserved(self):
        gate = FakeGate()
        module, agent = module_and_agent(gate)
        stats = install(module, token_quantum=32768)
        context = copy.deepcopy((agent._history_messages, agent._kept_functions,
                                 agent._summarized_knowledge))
        priorities = (agent._priority_current, agent._priority_next)
        agent._maybe_handover()
        agent._session_generated_tokens = 40000
        self.assertEqual(agent._maybe_handover(), "original-return")
        self.assertEqual(len(gate.calls), 1)
        expected = PrioritySnapshot(2, 5, 40000, 30000, 8)
        self.assertEqual(gate.calls[0], (d_priority(expected), expected))
        self.assertEqual(priorities, (agent._priority_current, agent._priority_next))
        self.assertEqual(context, (agent._history_messages, agent._kept_functions,
                                   agent._summarized_knowledge))
        self.assertEqual(stats.snapshot()["voluntary_handovers"], 1)
        agent._maybe_handover()
        self.assertEqual(len(gate.calls), 1)

    def test_real_trim_is_one_handover_not_two(self):
        gate = FakeGate()
        module, agent = module_and_agent(gate)
        stats = install(module, token_quantum=32768)
        agent._maybe_handover()
        agent._context_was_trimmed = True
        agent._session_generated_tokens = 50000
        agent._maybe_handover()
        self.assertEqual(len(gate.calls), 1)
        self.assertEqual(stats.snapshot()["voluntary_handovers"], 0)
        self.assertEqual(stats.snapshot()["ordinary_handovers_seen"], 1)

    def test_gate_disabled_adds_no_agent_state(self):
        module, agent = module_and_agent()
        install(module, token_quantum=32768)
        agent._maybe_handover()
        self.assertNotIn("_arc35_no_action_lease", vars(agent))

    def test_real_gate_allows_waiting_higher_dprime_and_resumes_without_deadlock(self):
        scope = real_gate_namespace()
        gate = scope["_PriorityGate"](1)
        # Emulate one stalled holder. The real gate compares its refreshed
        # D-prime snapshot to a waiting fresh game, then restores ownership.
        stalled = PrioritySnapshot(2, 100, 225000, 30000, 8)
        fresh = PrioritySnapshot(1, 0, 0, 0, 8)
        self.assertGreater(d_priority(fresh), d_priority(stalled))
        gate.acquire(d_priority(stalled))
        waiter_started = threading.Event()
        completed = threading.Event()
        order = []

        def waiter():
            waiter_started.set()
            with gate._cond:
                gate._enqueue_and_wait(d_priority(fresh), fresh)
            order.append("fresh admitted")
            gate.release()
            completed.set()

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        self.assertTrue(waiter_started.wait(1))
        until = time.monotonic() + 1
        while time.monotonic() < until:
            with gate._cond:
                if gate._waiting:
                    break
            time.sleep(.001)
        else:
            self.fail("Real gate waiter did not enqueue")
        gate.handover(d_priority(stalled), stalled)
        order.append("stalled resumed")
        self.assertTrue(completed.wait(1))
        gate.release()
        thread.join(1)
        self.assertEqual(order, ["fresh admitted", "stalled resumed"])
        self.assertEqual(gate._free, 1)
        self.assertFalse(gate._waiting)

    def test_no_new_handover_after_deadline_or_session_cancellation(self):
        gate = FakeGate()
        module, agent = module_and_agent(gate)
        install(module, token_quantum=16)
        agent._maybe_handover()
        agent._session_generated_tokens = 32
        gate._clock_deadline = time.monotonic() - 1
        agent._maybe_handover()
        self.assertFalse(gate.calls)
        gate._clock_deadline = None
        session = types.SimpleNamespace()
        session.should_stop = lambda: True
        agent._step_env_callback = types.MethodType(lambda self: None, session)
        agent._maybe_handover()
        self.assertFalse(gate.calls)

    def test_concurrent_real_gate_handovers_finish_and_conserve_slots(self):
        scope = real_gate_namespace()
        gate = scope["_PriorityGate"](2)
        outcomes = []
        def worker(index):
            snap = PrioritySnapshot(2, index, 60000, 30000, 8)
            gate.acquire(d_priority(snap))
            try:
                for _ in range(5):
                    gate.handover(d_priority(snap), snap)
                outcomes.append(index)
            finally:
                gate.release()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(sorted(outcomes), list(range(6)))
        self.assertEqual(gate._free, 2)
        self.assertFalse(gate._waiting)

    def test_builder_changes_only_install_cell_and_preserves_frozen_input(self):
        import json
        baseline = {"cells": [
            {"cell_type": "code", "source": "unchanged = 1"},
            {"cell_type": "code", "source": "_fp_mod.install_d(_fp_ta, _fp_solver)\n"},
            {"cell_type": "code", "source": "await bm.run()"}], "metadata": {}}
        with TemporaryDirectory() as directory:
            source = Path(directory) / "baseline.ipynb"
            dest = Path(directory) / "candidate.ipynb"
            source.write_text(json.dumps(baseline))
            frozen = source.read_bytes()
            build(source, dest)
            candidate = json.loads(dest.read_text())
            self.assertEqual(source.read_bytes(), frozen)
            self.assertEqual(candidate["cells"][0], baseline["cells"][0])
            self.assertEqual(candidate["cells"][2], baseline["cells"][2])
            self.assertEqual(candidate["metadata"]["arc35_no_action_handover"]["token_quantum"], 65536)
            with self.assertRaises(ValueError):
                build(source, source)

    def test_110_logical_games_real_gate_and_installer_conserve_ten_slots(self):
        scope = real_gate_namespace()
        gate = scope["_PriorityGate"](10)
        module, _ = module_and_agent(gate)
        stats = install(module, token_quantum=2048)
        start = threading.Event()
        all_launched = threading.Event()
        lock = threading.Lock()
        launched = 0
        completed = []
        failures = []

        def worker(index):
            nonlocal launched
            agent = module.ToolAgent()
            with lock:
                launched += 1
                if launched == 110:
                    all_launched.set()
            gate.acquire(d_priority(PrioritySnapshot(2, 5, 0, 30000, 8)))
            try:
                if not start.wait(3):
                    raise RuntimeError("CPU stress launch barrier timed out")
                agent._maybe_handover()
                for total in (2048, 4096):
                    agent._session_generated_tokens = total
                    agent._maybe_handover()
                with lock:
                    completed.append(index)
            except Exception as error:
                with lock:
                    failures.append(repr(error))
            finally:
                gate.release()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(110)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(all_launched.wait(2))
            until = time.monotonic() + 1
            while time.monotonic() < until:
                with gate._cond:
                    if len(gate._waiting) == 100:
                        break
                time.sleep(.001)
            else:
                self.fail("Expected 100 queued games and ten initial holders")
        finally:
            start.set()
        for thread in threads:
            thread.join(3)
        self.assertFalse(failures)
        self.assertEqual(sorted(completed), list(range(110)))
        self.assertEqual(stats.snapshot()["voluntary_handovers"], 220)
        self.assertEqual(gate._free, 10)
        self.assertFalse(gate._waiting)
        self.assertFalse(gate._admitted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
