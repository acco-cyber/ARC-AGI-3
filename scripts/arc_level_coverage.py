"""Evidence-only ARC3 per-level coverage, with no solver or routing changes.

GameRun JSON persists per-level action counters, engine completion counters and
chronological action records, but no per-action level. Token attribution to
levels is therefore opt-in and conditional on monotonic level progression.
The deployed ONLY_RESET_LEVELS=true policy is relevant evidence for that
assumption; this module does not silently assume it for arbitrary benchmarks.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real


_STATES = {"not_started", "playing", "won", "gave_up", "cancelled", "crashed"}
_KNOWN_NON_RESET = {f"ACTION{i}" for i in range(1, 8)} | {
    "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "CLICK", "UNDO",
}


def _count(value: object) -> int | None:
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def _seconds(value: object) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return number
    return None


def _state(value: object) -> str:
    state = str(value or "").strip().rsplit(".", 1)[-1].lower()
    return {"win": "won", "canceled": "cancelled"}.get(state, state)


def _canonical(value: object) -> str:
    return str(value or "").strip().lower().split("-", 1)[0]


def _action_kind(record: Mapping) -> str:
    action = record.get("action")
    if isinstance(action, Mapping):
        action = action.get("id", action.get("name"))
    if not isinstance(action, (str, Integral)) or isinstance(action, bool):
        return "unknown"
    name = str(action).strip().rsplit(".", 1)[-1].upper()
    if name in {"RESET", "ACTION0", "0"}:
        return "reset"
    if name in _KNOWN_NON_RESET or name in {str(i) for i in range(1, 8)}:
        return "non_reset"
    return "unknown"


def assess_level_coverage(
    benchmark: Mapping,
    *,
    expected_game_ids: Iterable[str] | None = None,
    levels_are_monotonic: bool = False,
    configured_limits: Mapping | None = None,
) -> dict:
    """Build a JSON-safe ledger; absent/invalid evidence never becomes coverage.

    A reached level is known to have been exposed by game startup or by an
    engine completion. A level can be reached without any executed action.
    Completed comes only from the engine's levels_completed counter. Non-reset
    actions show attempts, not understanding, exhaustive exploration or success.

    Limits are caller-provided run configuration, not invented per-level
    allocations. Wall times include waiting and CPU work, not exclusive GPU time.
    """
    if not isinstance(benchmark, Mapping) or not isinstance(benchmark.get("game_runs"), list):
        raise ValueError("benchmark must contain a game_runs list")
    if not isinstance(levels_are_monotonic, bool):
        raise ValueError("levels_are_monotonic must be an explicit boolean")
    if configured_limits is not None and not isinstance(configured_limits, Mapping):
        raise ValueError("configured_limits must be a mapping or None")

    errors = []
    warnings = []
    expected = None
    if expected_game_ids is not None:
        expected = [_canonical(g) for g in expected_game_ids]
        if not expected or not all(expected) or len(set(expected)) != len(expected):
            raise ValueError("expected_game_ids must contain distinct nonempty identities")
    passes = _count(benchmark.get("n_passes", 1))
    if passes is None or passes == 0:
        errors.append("invalid benchmark n_passes")
        passes = 1

    games = []
    seen_passes = Counter()
    for run_index, raw in enumerate(benchmark["game_runs"]):
        if not isinstance(raw, Mapping):
            errors.append(f"run {run_index}: not a mapping")
            continue
        gid = str(raw.get("game_id") or "").strip()
        canonical = _canonical(gid)
        prefix = f"run {run_index} ({gid or 'missing game_id'})"
        run_errors = []
        if not canonical:
            run_errors.append("missing game_id")
        pass_index = seen_passes[canonical]
        seen_passes[canonical] += 1
        state = _state(raw.get("state"))
        if state not in _STATES:
            run_errors.append(f"unknown state {state!r}")
        n = _count(raw.get("number_of_levels"))
        completed = _count(raw.get("levels_completed"))
        if n is None or n == 0 or completed is None or completed > n:
            errors.append(f"{prefix}: invalid level counts")
            # Do not manufacture missing level counts or a coverage denominator.
            continue
        counters = raw.get("actions_per_level")
        if not isinstance(counters, list) or len(counters) != n or any(_count(v) is None for v in counters):
            run_errors.append("actions_per_level must contain one nonnegative integer per level")
            counters = [None] * n
        history = raw.get("history")
        if not isinstance(history, list):
            run_errors.append("missing action history")
            history = []
        if all(v is not None for v in counters) and sum(counters) != len(history):
            run_errors.append("sum(actions_per_level) differs from history length")
        records = []
        prior_time = 0.0
        for action_index, record in enumerate(history):
            if not isinstance(record, Mapping):
                run_errors.append(f"action {action_index}: not a mapping")
                records.append({"tokens": None, "seconds": None, "kind": "unknown"})
                continue
            tokens = _count(record.get("generated_tokens"))
            seconds = _seconds(record.get("wallclock_seconds"))
            if tokens is None:
                run_errors.append(f"action {action_index}: invalid generated_tokens")
            if seconds is None or seconds < prior_time:
                run_errors.append(f"action {action_index}: invalid/nonmonotonic wallclock_seconds")
            if seconds is not None:
                prior_time = seconds
            records.append({"tokens": tokens, "seconds": seconds, "kind": _action_kind(record)})
        tail_tokens = _count(raw.get("final_generated_tokens", 0))
        final_seconds = _seconds(raw.get("final_wallclock_seconds", 0))
        if tail_tokens is None:
            run_errors.append("invalid final_generated_tokens")
        if final_seconds is None or (state not in {"not_started", "playing"} and final_seconds < prior_time):
            run_errors.append("invalid final_wallclock_seconds")
        if state == "won" and completed != n:
            run_errors.append("won state without all levels completed")
        if any((v or 0) > 0 for v in counters[completed + 1:]):
            run_errors.append("actions on a level beyond the reached frontier")
        started = bool(raw.get("started_at")) or bool(records) or completed > 0 or (tail_tokens or 0) > 0 or state in {"playing", "won", "gave_up", "cancelled"}
        if state == "not_started" and started:
            run_errors.append("not_started state contradicts startup/action/completion evidence")
        if completed > 0 and any(v == 0 for v in counters[:completed]):
            warnings.append(f"{prefix}: an engine-completed level has no recorded actions; do not claim it was played")

        known_total_tokens = (
            sum(r["tokens"] for r in records) + tail_tokens
            if tail_tokens is not None and all(r["tokens"] is not None for r in records) else None
        )
        attribution_ok = levels_are_monotonic and not run_errors
        cursor = 0
        levels = []
        for index in range(n):
            level_number = index + 1
            action_count = counters[index]
            is_completed = index < completed
            is_reached = is_completed or (started and index == completed)
            status = "completed" if is_completed else "reached" if is_reached else "unobserved"
            assigned = records[cursor:cursor + action_count] if attribution_ok else None
            if attribution_ok:
                cursor += action_count
            action_tokens = sum(r["tokens"] for r in assigned) if assigned is not None else None
            allocated_tail = tail_tokens if attribution_ok and is_reached and not is_completed else 0 if attribution_ok else None
            non_reset = sum(r["kind"] == "non_reset" for r in assigned) if assigned is not None else None
            reset = sum(r["kind"] == "reset" for r in assigned) if assigned is not None else None
            unknown = sum(r["kind"] == "unknown" for r in assigned) if assigned is not None else None
            levels.append({
                "level": level_number,
                "status": status,
                "completed_by_engine": is_completed,
                "reached_by_engine": is_reached,
                "has_executed_actions": action_count is not None and action_count > 0,
                "executed_actions": action_count,
                "non_reset_actions": non_reset,
                "reset_actions": reset,
                "unclassified_actions": unknown,
                "has_non_reset_action_evidence": non_reset > 0 if non_reset is not None else None,
                "action_generated_tokens": action_tokens,
                "final_no_action_generated_tokens": allocated_tail,
                "observed_generated_tokens": action_tokens + allocated_tail if action_tokens is not None else None,
                "first_action_wallclock_seconds": assigned[0]["seconds"] if assigned else None,
                "last_action_wallclock_seconds": assigned[-1]["seconds"] if assigned else None,
                "unobserved_reason": "preceding_level_not_completed" if status == "unobserved" and started else "game_not_started" if status == "unobserved" else None,
            })
        attributed_tokens = sum(l["observed_generated_tokens"] or 0 for l in levels) if attribution_ok else 0
        games.append({
            "run_index": run_index, "pass_index": pass_index,
            "game_id": gid, "canonical_game_id": canonical, "state": state,
            "game_started": started, "number_of_levels": n,
            "levels_completed": completed,
            "reached_levels": sum(l["reached_by_engine"] for l in levels),
            "levels_with_actions": sum(l["has_executed_actions"] for l in levels),
            "levels_with_non_reset_actions": sum(bool(l["has_non_reset_action_evidence"]) for l in levels) if attribution_ok else None,
            "total_executed_actions": sum(counters) if all(v is not None for v in counters) else None,
            "generated_tokens": known_total_tokens,
            "has_inference_evidence": known_total_tokens is not None and known_total_tokens > 0,
            "final_no_action_generated_tokens": tail_tokens,
            "unattributed_generated_tokens": known_total_tokens - attributed_tokens if known_total_tokens is not None else None,
            "final_wallclock_seconds": final_seconds,
            "valid_evidence": not run_errors,
            "invalid_reasons": run_errors,
            "per_level_cost_attribution": "chronological_counter_blocks_assuming_monotonic_levels" if attribution_ok else "unavailable",
            "levels": levels,
        })
        errors.extend(f"{prefix}: {e}" for e in run_errors)

    counts = Counter(g["canonical_game_id"] for g in games)
    if any(count > passes for count in counts.values()):
        errors.append("duplicate game runs beyond configured n_passes")
    missing = sorted(set(expected or []) - set(counts))
    unexpected = sorted(set(counts) - set(expected)) if expected is not None else []
    expected_counts_ok = (
        not missing and not unexpected and all(counts[g] == passes for g in expected)
        if expected is not None else None
    )
    if expected is not None and not expected_counts_ok:
        errors.append("observed game/pass identities do not match expected games and n_passes")
    flattened = [level for game in games for level in game["levels"]]
    n_total = len(flattened)
    n_completed = sum(l["completed_by_engine"] for l in flattened)
    n_reached = sum(l["reached_by_engine"] for l in flattened)
    n_actions = sum(l["has_executed_actions"] for l in flattened)
    non_reset_available = all(g["per_level_cost_attribution"] != "unavailable" for g in games)
    n_non_reset = sum(bool(l["has_non_reset_action_evidence"]) for l in flattened) if non_reset_available else None
    valid = not errors
    all_started = bool(games) and all(g["game_started"] for g in games)
    all_completed = bool(games) and n_completed == n_total
    return {
        "schema_version": 1,
        "scope": "Observed engine counters/actions only; no hidden-game score or exhaustive exploration claim",
        "valid_evidence": valid, "invalid_reasons": errors, "warnings": warnings,
        "semantics": {
            "completed": "Engine levels_completed includes this level; does not prove every mechanic was understood",
            "reached": "Starting/current board reached; may have zero actions",
            "unobserved": "No board/action evidence; later levels remain behind the uncompleted frontier",
            "action_coverage": "Executed actions include RESET; non-reset coverage is separate",
            "tokens": "Solver-reported generated tokens; final no-action tokens retained",
            "wallclock": "Elapsed since game start; includes queue/CPU time, not exclusive GPU time",
        },
        "levels_are_monotonic_assumption": levels_are_monotonic,
        "configured_limits": dict(configured_limits) if configured_limits is not None else None,
        "configured_limits_scope": "Caller-provided run policy, not inferred allocations or guarantees for individual levels",
        "expected_canonical_game_ids": sorted(expected) if expected is not None else None,
        "expected_passes": passes,
        "missing_game_ids": missing, "unexpected_game_ids": unexpected,
        "summary": {
            "game_runs": len(games), "unique_games": len(counts),
            "expected_game_pass_coverage_complete": expected_counts_ok,
            "started_game_runs": sum(g["game_started"] for g in games),
            "game_runs_with_inference": sum(g["has_inference_evidence"] for g in games),
            "won_game_runs": sum(g["state"] == "won" for g in games),
            "total_levels": n_total, "completed_levels": n_completed,
            "reached_uncompleted_levels": n_reached - n_completed,
            "unobserved_levels": n_total - n_reached,
            "levels_with_executed_actions": n_actions,
            "levels_with_non_reset_actions": n_non_reset,
            "level_completion_fraction": n_completed / n_total if n_total else None,
            "level_action_coverage_fraction": n_actions / n_total if n_total else None,
            "level_non_reset_action_coverage_fraction": n_non_reset / n_total if n_total and n_non_reset is not None else None,
            "all_observed_games_started": valid and all_started,
            "all_expected_games_started": valid and expected_counts_ok and all_started if expected is not None else None,
            "all_observed_levels_completed": valid and all_completed,
            "all_expected_games_and_levels_completed": valid and expected_counts_ok and all_completed if expected is not None else None,
        },
        "games": games,
    }
