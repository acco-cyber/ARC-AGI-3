"""Conservative public ARC3 evaluation reporting; never submits or packages.

The local bootstrap describes only the supplied public-game sample.  It is not
an estimate or guarantee of a Kaggle hidden-game score.  Full evaluation and
smoke testing deliberately share no automatic submission authorization.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral, Real


def canonical_game_id(value: object) -> str:
    """Ignore engine version/run suffixes while retaining the actual game ID."""
    return str(value or "").strip().lower().split("-", 1)[0]


def normalize_state(value: object) -> str:
    """Accept both plain TAAF states and enum-like representations."""
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raw = getattr(value, "name", raw)
    state = str(raw or "").strip().rsplit(".", 1)[-1].lower()
    return {"win": "won", "canceled": "cancelled"}.get(state, state)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    number = int(value)
    return number if number >= 0 else None


def _lower_bootstrap_mean(scores: list[float], samples: int, seed: int) -> float:
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(samples))
    rank = 0.05 * (samples - 1)
    lo = int(rank)
    hi = min(lo + 1, samples - 1)
    return means[lo] + (rank - lo) * (means[hi] - means[lo])


def assess_public_run(
    rows: Iterable[Mapping],
    expected_game_ids: Iterable[str],
    *,
    target: float = 40.0,
    evaluation_kind: str = "smoke",
    elapsed_seconds: float | None = None,
    budget_seconds: float | None = None,
    split_seed: str = "urad-arc3-target40-v1",
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20261016,
) -> dict:
    """Return a JSON-safe report using strict ``> target`` public-only checks.

    Expected identities define the split before any score is consulted.  A
    scoreless, absent, duplicate, crashed, unknown-state, still-playing or
    inference-free game invalidates full evaluation.  Zero-score games with
    actual inference remain in the mean.  Deadline cancellation is a valid
    terminal state; it never permits omitting that game's score or inference.

    A smoke run is diagnostic regardless of its scores.  Caller must provide
    whole-run elapsed and permitted budget evidence to pass full evaluation.
    """
    numeric_target = _finite_number(target)
    if numeric_target is None or not 0 <= numeric_target < 100:
        raise ValueError("target must be a finite number in [0, 100)")
    if evaluation_kind not in {"smoke", "full_public"}:
        raise ValueError("evaluation_kind must be 'smoke' or 'full_public'")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer of at least 100")

    expected = [canonical_game_id(gid) for gid in expected_game_ids]
    if len(expected) < 2 or not all(expected) or len(set(expected)) != len(expected):
        raise ValueError("expected_game_ids must contain at least two distinct valid game identities")
    order = sorted(expected, key=lambda gid: hashlib.sha256(
        (split_seed + "/" + gid).encode()).hexdigest())
    splits = {gid: ("heldout" if i % 2 else "development") for i, gid in enumerate(order)}

    games = []
    reasons = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            reasons.append(f"row {index}: not a mapping")
            continue
        gid = canonical_game_id(raw.get("game_id"))
        state = normalize_state(raw.get("state"))
        score = _finite_number(raw.get("score"))
        tokens = _count(raw.get("generated_tokens"))
        actions = _count(raw.get("actions"))
        completed = _count(raw.get("levels_completed"))
        levels = _count(raw.get("total_levels"))
        if score is None or not 0 <= score <= 100:
            reasons.append(f"{gid or index}: invalid score")
            score = None
        if state not in {"won", "gave_up", "cancelled"}:
            reasons.append(f"{gid or index}: invalid terminal state {state!r}")
        if tokens is None or tokens == 0:
            reasons.append(f"{gid or index}: no per-game inference evidence")
        if actions is None:
            reasons.append(f"{gid or index}: invalid action count")
        if completed is None or levels is None or levels == 0 or completed > levels:
            reasons.append(f"{gid or index}: invalid level counts")
        else:
            completion_cap = completed * (completed + 1) / (levels * (levels + 1)) * 100
            if score is not None and score > completion_cap + 1e-7:
                reasons.append(f"{gid or index}: score exceeds completed-level cap")
            if completed > 0 and actions == 0:
                reasons.append(f"{gid or index}: completed levels without counted actions")
            if state == "won" and completed != levels:
                reasons.append(f"{gid or index}: won state without all levels completed")
        games.append({
            "game_id": str(raw.get("game_id") or ""), "canonical_game_id": gid,
            "state": state, "score": score, "actions": actions,
            "generated_tokens": tokens, "levels_completed": completed,
            "total_levels": levels, "split": splits.get(gid),
        })

    counts = Counter(g["canonical_game_id"] for g in games)
    missing = sorted(set(expected) - counts.keys())
    unexpected = sorted(counts.keys() - set(expected))
    duplicate = sorted(gid for gid, count in counts.items() if count != 1)
    if missing:
        reasons.append("missing public games: " + ", ".join(missing))
    if unexpected:
        reasons.append("unexpected public games: " + ", ".join(unexpected))
    if duplicate:
        reasons.append("duplicate public game identities: " + ", ".join(duplicate))
    if len(games) != len(expected):
        reasons.append(f"expected {len(expected)} games; received {len(games)}")

    elapsed = _finite_number(elapsed_seconds)
    budget = _finite_number(budget_seconds)
    budget_ok = elapsed is not None and budget is not None and 0 <= elapsed <= budget and budget > 0
    if not budget_ok:
        reasons.append("missing, invalid, or exceeded whole-run runtime budget")

    complete = not missing and not unexpected and not duplicate and len(games) == len(expected)
    scores_ok = complete and all(g["score"] is not None for g in games)
    mean = heldout_mean = lower = None
    if scores_ok:
        scores = [g["score"] for g in sorted(games, key=lambda row: row["canonical_game_id"])]
        heldout = [g["score"] for g in games if g["split"] == "heldout"]
        mean = sum(scores) / len(scores)
        heldout_mean = sum(heldout) / len(heldout)
        lower = _lower_bootstrap_mean(scores, bootstrap_samples, bootstrap_seed)
    checks = {
        "all_expected_games_present_once": complete,
        "all_game_scores_valid": scores_ok,
        "all_games_have_inference": complete and all((g["generated_tokens"] or 0) > 0 for g in games),
        "all_games_terminal_without_crash": complete and all(g["state"] in {"won", "gave_up", "cancelled"} for g in games),
        "whole_run_within_budget": budget_ok,
        "mean_strictly_above_target": mean is not None and mean > numeric_target,
        "heldout_mean_strictly_above_target": heldout_mean is not None and heldout_mean > numeric_target,
        "public_bootstrap_lower_strictly_above_target": lower is not None and lower > numeric_target,
    }
    valid = not reasons
    passed = evaluation_kind == "full_public" and valid and all(checks.values())
    return {
        "scope": "Public offline games only; official Kaggle hidden-game score is unknown",
        "evaluation_kind": evaluation_kind, "target": numeric_target,
        "comparison": "strictly greater than target", "valid_evaluation": valid,
        "local_validation_passed": passed, "official_score": None,
        "official_score_guaranteed": False, "submission_authorized": False,
        "packaging_authorized": False,
        "mean_public_score": mean, "heldout_mean_public_score": heldout_mean,
        "public_game_bootstrap_lower_95": lower,
        "bootstrap_interval": "one-sided 95% percentile lower bound",
        "bootstrap_scope": "Variation within these public games only; not hidden-set uncertainty",
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
        "split_seed": split_seed, "expected_games": len(expected), "evaluated_games": len(games),
        "expected_canonical_game_ids": sorted(expected),
        "missing_game_ids": missing, "unexpected_game_ids": unexpected,
        "duplicate_game_ids": duplicate, "checks": checks, "invalid_reasons": reasons,
        "elapsed_seconds": elapsed, "budget_seconds": budget,
        "games": sorted(games, key=lambda g: (g["canonical_game_id"], g["game_id"])),
    }
