"""Milestone I (Code/AGENT_INTERFACE_PLAN.md): the registry and the
evaluation harness (pairing matrix, duplicate evaluation, Bradley-Terry
ratings, SPRT).
"""

import unittest

from bots.registry import AgentEntry, available_agents, make_agent, privilege_of, register
from keyforge.enums import PrivilegeLevel
from sim.evaluate import duplicate_evaluate, pairing_matrix_evaluate, summarize
from sim.rate import Rating, bradley_terry, games_needed_for_gap, non_transitivity_report, sprt, standard_error


class TestRegistry(unittest.TestCase):
    def test_random_and_heuristic_are_registered_by_default(self):
        agents = available_agents()
        self.assertIn("random", agents)
        self.assertIn("heuristic", agents)
        self.assertIsInstance(agents["random"], AgentEntry)

    def test_make_agent_builds_a_working_controller(self):
        bot = make_agent("heuristic", seed=1)
        self.assertTrue(hasattr(bot, "decide"))

    def test_unknown_agent_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            make_agent("definitely-not-registered")

    def test_checkpoint_suffix_is_parsed_and_forwarded(self):
        seen = {}

        def make(seed=None, checkpoint=None, **kw):
            seen["checkpoint"] = checkpoint
            from bots.random_bot import RandomBot
            return RandomBot(seed=seed)

        register("_test_checkpointed", make)
        try:
            make_agent("_test_checkpointed@v3", seed=1)
            self.assertEqual(seen["checkpoint"], "v3")
        finally:
            from bots import registry as registry_module
            del registry_module._REGISTRY["_test_checkpointed"]

    def test_default_privilege_is_observation(self):
        self.assertEqual(privilege_of("random"), PrivilegeLevel.OBSERVATION)
        self.assertEqual(privilege_of("heuristic"), PrivilegeLevel.OBSERVATION)

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            register("random", lambda seed=None, **kw: None)


class TestDuplicateEvaluationAndPairingMatrix(unittest.TestCase):
    def test_heuristic_beats_random_regardless_of_seat_or_deck(self):
        result = duplicate_evaluate("heuristic", "random", "fignor", "igor", seed=100, max_turns=150)
        self.assertEqual(result.agent_a_wins, 2)
        self.assertEqual(result.agent_b_wins, 0)

    def test_pairing_matrix_covers_all_four_seat_deck_combinations(self):
        results = pairing_matrix_evaluate("heuristic", "random", "fignor", "igor", seeds=[1, 2], max_turns=150)
        self.assertEqual(len(results), 4)  # 2 seeds x 2 deck assignments, each a duplicate (2 games)
        deck_pairs = {(r.deck_a, r.deck_b) for r in results}
        self.assertEqual(deck_pairs, {("fignor", "igor"), ("igor", "fignor")})
        summary = summarize(results)
        self.assertEqual(summary["games"], 8)
        self.assertGreater(summary["agent_a_win_rate"], 0.9)


class TestBradleyTerry(unittest.TestCase):
    def test_a_clear_hierarchy_produces_ratings_in_order(self):
        # A beats B beats C, consistently.
        results = []
        for _ in range(50):
            results.append(("A", "B", 1.0))
            results.append(("B", "C", 1.0))
            results.append(("A", "C", 1.0))
        ratings = bradley_terry(results)
        self.assertGreater(ratings["A"].rating, ratings["B"].rating)
        self.assertGreater(ratings["B"].rating, ratings["C"].rating)

    def test_rating_gaps_are_invariant_to_anchor_choice(self):
        results = [("A", "B", 1.0)] * 30 + [("B", "C", 1.0)] * 30 + [("A", "C", 1.0)] * 30
        r1 = bradley_terry(results, anchor="A")
        r2 = bradley_terry(results, anchor="C")
        gap1 = r1["A"].rating - r1["C"].rating
        gap2 = r2["A"].rating - r2["C"].rating
        self.assertAlmostEqual(gap1, gap2, places=6)

    def test_even_record_gives_equal_ratings(self):
        results = [("A", "B", 1.0)] * 20 + [("A", "B", 0.0)] * 20
        ratings = bradley_terry(results)
        self.assertAlmostEqual(ratings["A"].rating, ratings["B"].rating, places=6)

    def test_standard_error_and_games_needed_match_the_plans_guidance(self):
        self.assertAlmostEqual(standard_error(1000), 0.0158, places=3)
        # "about 1,000 games to resolve a 3-point gap and 10,000 for 1
        # point" -- approximate figures, so check within a generous band.
        self.assertTrue(800 <= games_needed_for_gap(0.03) <= 1300, games_needed_for_gap(0.03))
        self.assertEqual(games_needed_for_gap(0.01), 10000)

    def test_non_transitivity_is_flagged(self):
        # A rates higher overall but lost its own head-to-head against B
        # decisively -- a direct check of the report's own logic, rather
        # than trying to reverse-engineer game results that fit it.
        ratings = {
            "A": Rating(name="A", rating=1.0, games=100, wins=60),
            "B": Rating(name="B", rating=0.0, games=100, wins=50),
        }
        pairwise_win_rate = {("A", "B"): 0.2, ("B", "A"): 0.8}
        notes = non_transitivity_report(ratings, pairwise_win_rate)
        self.assertTrue(notes, "expected a non-transitivity note when A rates above B but lost their head-to-head")


class TestSPRT(unittest.TestCase):
    def test_lopsided_results_resolve_to_h1(self):
        decision = None
        wins = losses = draws = 0
        for _ in range(2000):
            wins += 1
            decision = sprt(wins, losses, draws, elo0=0, elo1=20)
            if decision is not None:
                break
        self.assertEqual(decision, "H1")

    def test_even_results_never_resolve_to_h1(self):
        decision = None
        wins = losses = draws = 0
        for i in range(400):
            if i % 2 == 0:
                wins += 1
            else:
                losses += 1
            decision = sprt(wins, losses, draws, elo0=0, elo1=20)
        self.assertNotEqual(decision, "H1")


if __name__ == "__main__":
    unittest.main()
