"""Optional admission handover after prolonged deliberation without action.

Single experimental change: at an existing pre-request boundary, voluntarily
requeue an intact-prefix game after one token quantum without an executed
action. Generation is never interrupted, its output ceiling is unchanged, and
all conversation, world-model and sandbox state is retained. Existing D-prime
scoring still decides whether another game gets the slot.

Call install(tool_agent, token_quantum=65536) AFTER install_d and BEFORE playing.
The default quantum is zero: no monkeypatch and byte-for-byte baseline behavior.
This is an experiment, not a demonstrated score or runtime improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import threading
import time


@dataclass
class NoActionLease:
    """Per-agent accounting; does not reset on analyzer yield/resumption."""

    quantum: int
    last_actions: int | None = None
    last_tokens: int = 0
    action_anchor: int = 0
    lease_anchor: int = 0

    def observe(self, tokens: int, actions: int, *, ordinary_handover=False) -> bool:
        if tokens < 0 or actions < 0:
            raise ValueError("Token and action counts must be nonnegative")
        if (self.last_actions is None or actions != self.last_actions
                or tokens < self.last_tokens):
            # Action progress, a session reset, or existing request-error token
            # rollback begins a new lease. Never turn a negative delta into a
            # false handover. Tokens spent reaching an action are behind us.
            self.action_anchor = self.lease_anchor = tokens
        self.last_actions = actions
        self.last_tokens = tokens
        if ordinary_handover:
            self.lease_anchor = tokens
        return self.quantum > 0 and tokens - self.lease_anchor >= self.quantum

    def handed_over(self, tokens: int) -> None:
        self.lease_anchor = tokens


class HandoverStats:
    def __init__(self, token_quantum: int):
        self._lock = threading.Lock()
        self._data = {
            "enabled": token_quantum > 0, "token_quantum": token_quantum,
            "voluntary_handovers": 0, "ordinary_handovers_seen": 0,
            "max_tokens_without_action_at_handover": 0,
            "voluntary_handover_wait_seconds": 0.0,
        }

    def note_ordinary(self):
        with self._lock:
            self._data["ordinary_handovers_seen"] += 1

    def note_voluntary(self, no_action_tokens: int, elapsed: float):
        with self._lock:
            self._data["voluntary_handovers"] += 1
            self._data["max_tokens_without_action_at_handover"] = max(
                no_action_tokens, self._data["max_tokens_without_action_at_handover"])
            self._data["voluntary_handover_wait_seconds"] += elapsed

    def snapshot(self):
        with self._lock:
            return dict(self._data)


def install(tool_agent, *, token_quantum: int = 0) -> HandoverStats:
    """Wrap only the established safe pre-request admission hook.

    Assumes the pinned source's refreshed queue snapshot path, as activated by
    install_d. Refuses incompatible source/config instead of silently changing
    another scheduler. Priority bands and model request parameters are untouched.
    """
    if type(token_quantum) is not int or token_quantum < 0:
        raise ValueError("token_quantum must be a nonnegative integer")
    prior_method = getattr(getattr(tool_agent, "ToolAgent", None), "_maybe_handover", None)
    prior_config = getattr(prior_method, "_no_action_handover_config", None)
    if prior_config is not None:
        if prior_config[0] != token_quantum:
            raise RuntimeError("No-action handover already installed with another quantum")
        return prior_config[1]
    stats = HandoverStats(token_quantum)
    if token_quantum == 0:
        return stats
    for name in ("ToolAgent", "PrioritySnapshot", "_priority_gate",
                 "_priority_action_count", "_priority_refresh_enabled",
                 "_priority_level_options", "_get_env_bool"):
        if not hasattr(tool_agent, name):
            raise RuntimeError(f"Pinned scheduler is missing {name}")
    if not tool_agent._priority_refresh_enabled():
        raise RuntimeError("Install after D-prime enables refreshed queue snapshots")
    cls = tool_agent.ToolAgent
    original = getattr(cls, "_maybe_handover", None)
    if not callable(original):
        raise RuntimeError("Pinned agent is missing _maybe_handover")
    @wraps(original)
    def maybe_handover(self):
        ordinary = bool(getattr(self, "_context_was_trimmed", False))
        result = original(self)
        gate = tool_agent._priority_gate()
        if gate is None:
            return result
        # _maybe_handover runs only between complete responses, after all tool
        # results were recorded and after normal context maintenance. No HTTP
        # request or partially emitted tool call is cancelled by this wrapper.
        summary = self._last_step_summary or {}
        total = max(0, int(self._session_generated_tokens))
        action_count = max(0, tool_agent._priority_action_count(summary))
        lease = getattr(self, "_arc35_no_action_lease", None)
        if lease is None:
            lease = NoActionLease(token_quantum)
            self._arc35_no_action_lease = lease
        due = lease.observe(total, action_count, ordinary_handover=ordinary)
        if ordinary:
            stats.note_ordinary()
        if not due:
            return result
        deadline = getattr(gate, "_clock_deadline", None)
        if deadline is not None and time.monotonic() >= deadline:
            return result
        session = getattr(getattr(self, "_step_env_callback", None), "__self__", None)
        should_stop = getattr(session, "should_stop", None)
        if callable(should_stop) and should_stop():
            return result
        # Match the original refreshed-queue snapshot exactly. In particular,
        # FormPace returns mean tokens/cleared level, not the base pace ratio.
        level = max(1, int(summary.get("level") or 1))
        actions = max(0, action_count - self._actions_at_level_start)
        tokens = max(0, total - getattr(self, "_tokens_at_level_start", 0))
        options = tool_agent._priority_level_options()
        total_levels = (self._priority_total_levels()
                        if options["normalize_score"] or options["tail_lookup"]
                        else None)
        pace = (self._progress_pace.cost_multiplier()
                if tool_agent._get_env_bool("ARC3_PRIORITY_PACE", False) else 1.0)
        snapshot = tool_agent.PrioritySnapshot(level, actions, tokens, pace, total_levels)
        estimate = gate._snapshot_priority(snapshot, time.monotonic())
        start = time.monotonic()
        gate.handover(estimate, snapshot)
        lease.handed_over(total)
        stats.note_voluntary(total - lease.action_anchor, time.monotonic() - start)
        return result

    maybe_handover._no_action_handover_config = (token_quantum, stats)
    cls._maybe_handover = maybe_handover
    return stats
