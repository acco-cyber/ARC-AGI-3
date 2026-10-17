import copy
import math
import unittest

from arc_level_coverage import assess_level_coverage


def run(*, actions=(2, 1, 0), completed=1, state="cancelled", tail=7):
    records = []
    for i in range(sum(actions)):
        records.append({"action": {"id": "RESET" if i == 0 else "ACTION1", "data": {}},
                        "generated_tokens": (i + 1) * 10, "wallclock_seconds": float(i + 1)})
    return {"game_id": "example-v1", "number_of_levels": len(actions),
            "levels_completed": completed, "state": state,
            "actions_per_level": list(actions), "history": records,
            "started_at": "2026-10-17T00:00:00", "final_generated_tokens": tail,
            "final_wallclock_seconds": float(sum(actions) + 5)}


def assess(game, **kwargs):
    return assess_level_coverage({"n_passes": 1, "game_runs": [game]}, **kwargs)


class CoverageEvidenceTests(unittest.TestCase):
    def test_partial_run_does_not_claim_every_level_played(self):
        r = assess(run(), expected_game_ids=["example-v1"], levels_are_monotonic=True)
        self.assertTrue(r["valid_evidence"])
        self.assertEqual([l["status"] for l in r["games"][0]["levels"]], ["completed", "reached", "unobserved"])
        self.assertEqual(r["summary"]["completed_levels"], 1)
        self.assertEqual(r["summary"]["reached_uncompleted_levels"], 1)
        self.assertEqual(r["summary"]["unobserved_levels"], 1)
        self.assertTrue(r["summary"]["all_expected_games_started"])
        self.assertFalse(r["summary"]["all_expected_games_and_levels_completed"])

    def test_reached_level_with_zero_actions_is_not_played(self):
        r = assess(run(actions=(2, 0, 0), completed=1), levels_are_monotonic=True)
        level = r["games"][0]["levels"][1]
        self.assertEqual(level["status"], "reached")
        self.assertFalse(level["has_executed_actions"])
        self.assertFalse(level["has_non_reset_action_evidence"])
        self.assertEqual(level["final_no_action_generated_tokens"], 7)

    def test_reset_only_is_not_non_reset_gameplay(self):
        r = assess(run(actions=(1, 0), completed=0, tail=0), levels_are_monotonic=True)
        self.assertEqual(r["summary"]["levels_with_executed_actions"], 1)
        self.assertEqual(r["summary"]["levels_with_non_reset_actions"], 0)
        self.assertFalse(r["games"][0]["has_inference_evidence"] is False)  # fixture has 10 tokens
        self.assertEqual(r["games"][0]["levels"][0]["reset_actions"], 1)

    def test_token_totals_reconcile_including_no_action_tail(self):
        r = assess(run(), levels_are_monotonic=True)
        g = r["games"][0]
        self.assertEqual([l["observed_generated_tokens"] for l in g["levels"]], [30, 37, 0])
        self.assertEqual(g["generated_tokens"], 67)
        self.assertEqual(g["unattributed_generated_tokens"], 0)

    def test_default_does_not_invent_per_level_costs(self):
        r = assess(run())
        g = r["games"][0]
        self.assertTrue(r["valid_evidence"])
        self.assertIsNone(g["levels"][0]["observed_generated_tokens"])
        self.assertIsNone(r["summary"]["levels_with_non_reset_actions"])
        self.assertEqual(g["unattributed_generated_tokens"], 67)

    def test_won_completion_and_post_win_tail_not_assigned_to_level(self):
        r = assess(run(actions=(2, 1, 1), completed=3, state="won", tail=9),
                   expected_game_ids=["example"], levels_are_monotonic=True)
        self.assertTrue(r["summary"]["all_expected_games_and_levels_completed"])
        self.assertEqual(r["games"][0]["unattributed_generated_tokens"], 9)
        self.assertEqual([l["final_no_action_generated_tokens"] for l in r["games"][0]["levels"]], [0, 0, 0])

    def test_engine_completed_level_without_action_not_claimed_played(self):
        r = assess(run(actions=(1, 0, 1), completed=3, state="won", tail=0), levels_are_monotonic=True)
        self.assertTrue(r["valid_evidence"])
        self.assertEqual(r["games"][0]["levels"][1]["status"], "completed")
        self.assertFalse(r["games"][0]["levels"][1]["has_executed_actions"])
        self.assertTrue(r["warnings"])

    def test_missing_expected_game_invalidates_full_coverage(self):
        r = assess(run(actions=(1,), completed=1, state="won", tail=0), expected_game_ids=["example", "missing"])
        self.assertFalse(r["valid_evidence"])
        self.assertFalse(r["summary"]["all_expected_games_and_levels_completed"])
        self.assertEqual(r["missing_game_ids"], ["missing"])

    def test_counter_mismatch_cannot_attribute_tokens_or_pass(self):
        game = run()
        game["actions_per_level"][0] += 1
        r = assess(game, levels_are_monotonic=True)
        self.assertFalse(r["valid_evidence"])
        self.assertIsNone(r["games"][0]["levels"][0]["observed_generated_tokens"])

    def test_nonfinite_or_negative_costs_invalid(self):
        for field, value in (("generated_tokens", -1), ("wallclock_seconds", math.nan),
                             ("generated_tokens", True)):
            game = run()
            game["history"][1][field] = value
            r = assess(game, levels_are_monotonic=True)
            self.assertFalse(r["valid_evidence"])

    def test_nonmonotonic_wallclock_invalid(self):
        game = run()
        game["history"][2]["wallclock_seconds"] = 0.1
        self.assertFalse(assess(game)["valid_evidence"])

    def test_not_started_level_is_unobserved(self):
        game = run(actions=(0, 0), completed=0, state="not_started", tail=0)
        game["started_at"] = None
        r = assess(game, expected_game_ids=["example"], levels_are_monotonic=True)
        self.assertTrue(r["valid_evidence"])
        self.assertFalse(r["summary"]["all_expected_games_started"])
        self.assertEqual([l["status"] for l in r["games"][0]["levels"]], ["unobserved", "unobserved"])

    def test_duplicate_passes_are_not_silently_full_game_coverage(self):
        game = run()
        r = assess_level_coverage({"game_runs": [game, copy.deepcopy(game)], "n_passes": 1}, expected_game_ids=["example"])
        self.assertFalse(r["valid_evidence"])
        r = assess_level_coverage({"game_runs": [game, copy.deepcopy(game)], "n_passes": 2}, expected_game_ids=["example"])
        self.assertTrue(r["valid_evidence"])
        self.assertEqual([g["pass_index"] for g in r["games"]], [0, 1])

    def test_won_inconsistent_with_engine_count_invalid(self):
        self.assertFalse(assess(run(state="won"))["valid_evidence"])

    def test_invalid_counter_denominator_does_not_disappear_into_success(self):
        game = run()
        game["number_of_levels"] = None
        r = assess(game, expected_game_ids=["example"])
        self.assertFalse(r["valid_evidence"])
        self.assertFalse(r["summary"]["all_expected_games_and_levels_completed"])
        self.assertIsNone(r["summary"]["level_completion_fraction"])


if __name__ == "__main__":
    unittest.main()
