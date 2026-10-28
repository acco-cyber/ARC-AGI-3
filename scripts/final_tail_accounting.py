"""Opt-in final-token accounting for two pinned public Duck finalizers.

No model, request, game action, scoring formula or deadline is changed. This is
CPU-tested support code, not a score improvement. Install only before play.
If an externally cancelled worker never returns, its remaining tokens cannot
be recovered here. A later serialized benchmark must be saved after workers
return to include their reconciliation.
"""
from __future__ import annotations

import ast
from functools import wraps
import hashlib
import inspect
import textwrap
import threading
import time

_SESSION_FINGERPRINTS = {
    "12b11f2456edef277324e9f598b6d1aa43f072805988b3ab42ad3ef851dc1a37",
    "6e9d39bc9a46b5cc242a33b418a0d73224d10d98f07855bb377ad80b46ad19e8",
}
_REMAINING_FINGERPRINT = "12b48d351b7e3a756457f9601e5b4a5f041eb21dfd79fe205fb3611da4c84c03"


def _fingerprint(method):
    # inspect.getsource unwraps functools.wraps, which could otherwise hide a
    # foreign wrapper's changed behavior behind the pinned original source.
    if hasattr(method, "__wrapped__"):
        raise RuntimeError("Wrapped finalizer is not an unmodified pinned method")
    try:
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise RuntimeError("Cannot verify pinned finalizer source") from exc
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def install(solver_module):
    """Explicitly install once; refuse unrecognized or already modified methods."""
    if getattr(solver_module, "_arc3_final_tail_accounting_installed", False):
        return False
    session_cls = solver_module._HarnessGameSession
    solver_cls = solver_module.HarnessSolver
    original_session = session_cls._finish_if_needed
    original_remaining = solver_cls._finish_remaining
    if (_fingerprint(original_session) not in _SESSION_FINGERPRINTS
            or _fingerprint(original_remaining) != _REMAINING_FINGERPRINT):
        raise RuntimeError("Unrecognized finalizer source; accounting patch not installed")
    reported_tokens = solver_module._analyzer_reported_tokens
    if not callable(reported_tokens):
        raise RuntimeError("Pinned analyzer token accessor is missing")
    lock = threading.RLock()

    @wraps(original_session)
    def finish_session(self):
        with lock:
            run = self.game.game_run
            if run is None or getattr(self, "_arc3_tail_accounted_run", None) is run:
                return
            if run.final_score is not None:
                # Only our external-cancellation wrapper may grant this path.
                # Ordinary finalization remains idempotent even if a counter
                # changes later. The marker belongs to this exact GameRun.
                if not getattr(run, "_arc3_external_tail_pending", False):
                    return
                current = reported_tokens(self.analyzer)
                tail = max(0, current - self.token_baseline)
                run.final_generated_tokens += tail
                if tail and run.started_at_monotonic is not None:
                    run.final_wallclock_seconds = max(
                        run.final_wallclock_seconds,
                        time.monotonic() - run.started_at_monotonic,
                    )
                self.token_baseline = current
                run._arc3_external_tail_pending = False
                self._arc3_tail_accounted_run = run
                return
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            current = reported_tokens(self.analyzer)
            tail = max(0, current - self.token_baseline)
            self.game.finish_game(generated_tokens=tail)
            self.token_baseline = current
            self._arc3_tail_accounted_run = run

    @wraps(original_remaining)
    def finish_remaining(self, games):
        with lock:
            # Capture only runs which were unfinished on entry. A normally
            # completed run must never become eligible for late accounting.
            pending = [g.game_run for g in games
                       if g.game_run is not None and g.game_run.final_score is None]
            original_remaining(self, games)
            if self._stop_event.is_set():
                for run in pending:
                    if run.final_score is not None and run.state == "cancelled":
                        run._arc3_external_tail_pending = True

    session_cls._finish_if_needed = finish_session
    solver_cls._finish_remaining = finish_remaining
    solver_module._arc3_final_tail_accounting_installed = True
    return True
