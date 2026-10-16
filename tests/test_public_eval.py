import enum
import json
import math
import unittest

from arc40_evaluation import assess_public_run, normalize_state


EXPECTED = [f"g{i:02}-v1" for i in range(25)]


def rows(score=65):
    return [dict(game_id=f"g{i:02}-fresh-version", state="won", score=score,
                 generated_tokens=500, actions=10, levels_completed=6,
                 total_levels=6) for i in range(25)]


def assess(data, **kwargs):
    return assess_public_run(data, EXPECTED, evaluation_kind="full_public",
                             elapsed_seconds=60, budget_seconds=100,
                             bootstrap_samples=500, **kwargs)


class PublicEvaluationTests(unittest.TestCase):
    def test_full_public_pass_is_still_not_an_official_guarantee(self):
        report = assess(rows())
        self.assertTrue(report["local_validation_passed"])
        self.assertIsNone(report["official_score"])
        self.assertFalse(report["official_score_guaranteed"])
        self.assertFalse(report["submission_authorized"])
        self.assertFalse(report["packaging_authorized"])
        json.dumps(report, allow_nan=False)

    def test_exactly_40_does_not_exceed_40(self):
        report = assess(rows(40))
        self.assertTrue(report["valid_evaluation"])
        self.assertFalse(report["local_validation_passed"])
        self.assertFalse(report["checks"]["mean_strictly_above_target"])

    def test_smoke_never_passes_or_packages_even_at_100(self):
        report = assess_public_run(rows(100), EXPECTED, elapsed_seconds=1,
                                   budget_seconds=100, bootstrap_samples=100)
        self.assertEqual(report["evaluation_kind"], "smoke")
        self.assertFalse(report["local_validation_passed"])
        self.assertFalse(report["packaging_authorized"])

    def test_enum_and_prefixed_states_are_normalized(self):
        class State(enum.Enum):
            CRASHED = "crashed"
        self.assertEqual(normalize_state(State.CRASHED), "crashed")
        self.assertEqual(normalize_state("GameState.CRASHED"), "crashed")
        data = rows()
        data[0]["state"] = "GameState.CRASHED"
        self.assertFalse(assess(data)["local_validation_passed"])

    def test_playing_unknown_and_inference_free_games_invalidate(self):
        for state in ["playing", "GameState.PLAYING", "not_started", "mystery"]:
            with self.subTest(state=state):
                data = rows()
                data[0]["state"] = state
                self.assertFalse(assess(data)["valid_evaluation"])
        data = rows()
        data[0]["generated_tokens"] = 0
        report = assess(data)
        self.assertFalse(report["local_validation_passed"])
        self.assertFalse(report["checks"]["all_games_have_inference"])

    def test_wrong_identity_with_same_count_is_rejected(self):
        data = rows()
        data[0]["game_id"] = "wrong-game"
        report = assess(data)
        self.assertEqual(report["missing_game_ids"], ["g00"])
        self.assertEqual(report["unexpected_game_ids"], ["wrong"])
        self.assertFalse(report["local_validation_passed"])

    def test_duplicate_version_of_same_game_is_rejected(self):
        data = rows()
        data[0]["game_id"] = data[1]["game_id"] + "-other"
        report = assess(data)
        self.assertIn("g01", report["duplicate_game_ids"])
        self.assertFalse(report["local_validation_passed"])

    def test_zero_score_game_is_retained_in_mean(self):
        data = rows(50)
        data[0]["score"] = 0
        report = assess(data)
        self.assertEqual(report["evaluated_games"], 25)
        self.assertEqual(report["mean_public_score"], 48)

    def test_nonfinite_boolean_and_out_of_range_scores_rejected_json_safe(self):
        for score in [math.nan, math.inf, -math.inf, True, -1, 101, None, "70"]:
            with self.subTest(score=score):
                data = rows()
                data[0]["score"] = score
                report = assess(data)
                self.assertFalse(report["local_validation_passed"])
                json.dumps(report, allow_nan=False)

    def test_split_is_stable_under_row_reordering_and_suffixes(self):
        data = rows()
        for i, row in enumerate(data):
            row["score"] = 41 + i
        a = assess(data)
        b = assess(list(reversed(data)))
        split_a = {g["canonical_game_id"]: g["split"] for g in a["games"]}
        split_b = {g["canonical_game_id"]: g["split"] for g in b["games"]}
        self.assertEqual(split_a, split_b)
        self.assertEqual(a["public_game_bootstrap_lower_95"], b["public_game_bootstrap_lower_95"])

    def test_cancellation_still_requires_score_and_inference(self):
        data = rows()
        data[0]["state"] = "cancelled"
        self.assertTrue(assess(data)["valid_evaluation"])
        data[0]["generated_tokens"] = 0
        self.assertFalse(assess(data)["valid_evaluation"])

    def test_runtime_and_level_evidence_is_required(self):
        report = assess_public_run(rows(), EXPECTED, evaluation_kind="full_public", bootstrap_samples=100)
        self.assertFalse(report["valid_evaluation"])
        data = rows()
        data[0]["levels_completed"] = 7
        self.assertFalse(assess(data)["valid_evaluation"])

    def test_impossible_completed_level_score_is_rejected(self):
        data = rows(100)
        data[0]["state"] = "gave_up"
        data[0]["levels_completed"] = 1
        report = assess(data)
        self.assertFalse(report["valid_evaluation"])
        self.assertTrue(any("completed-level cap" in reason for reason in report["invalid_reasons"]))


if __name__ == "__main__":
    unittest.main()
