"""Milestone J (Code/AGENT_INTERFACE_PLAN.md): the no-network baseline and
the two diagnostics. (Data generation is `sim/bc_corpus.py` and
`agent/selfplay.py`, tested with the training milestones.)
"""

import unittest

from bots.baseline_evaluator import heuristic_value, uniform_policy
from bots.random_bot import RandomBot
from bots.search_bot import DeterminizedRolloutBot
from keyforge.config import GameConfig
from keyforge.decision import Decision
from keyforge.enums import DecisionKind, PrivilegeLevel
from sim.driver import derive_agent_seed, run_games
from tests.helpers import new_game


class TestBaselineEvaluator(unittest.TestCase):
    def test_uniform_policy_sums_to_one(self):
        game = new_game()
        d = game.pending_decision
        policy = uniform_policy(d)
        self.assertEqual(len(policy), len(d.options))
        self.assertAlmostEqual(sum(policy), 1.0, places=9)

    def test_uniform_policy_of_no_options_is_empty(self):
        d = Decision(player=1, kind=DecisionKind.CHOOSE_CARDS, prompt="x", options=[], min_n=0, max_n=0)
        self.assertEqual(uniform_policy(d), [])

    def test_heuristic_value_is_bounded_and_favors_the_leading_player(self):
        game = new_game()
        v = heuristic_value(game.view_for(1))
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)
        game.players[1].keys = 2
        v2 = heuristic_value(game.view_for(1))
        self.assertGreater(v2, v)


class TestSearchBot(unittest.TestCase):
    def test_falls_back_to_the_rollout_policy_without_a_capability(self):
        game = new_game()
        bot = DeterminizedRolloutBot(seed=1, n_samples=2)
        d = game.pending_decision
        choice = bot.decide(game.view_for(d.player), d, budget=None, capability=None)
        self.assertTrue(d.validate(choice))

    def test_beats_random_bot_with_search_privilege(self):
        results = run_games(
            4,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=60),
            lambda i: {
                1: DeterminizedRolloutBot(seed=derive_agent_seed(0, i, 1), n_samples=4),
                2: RandomBot(seed=derive_agent_seed(0, i, 2)),
            },
            privilege={1: PrivilegeLevel.SEARCH, 2: PrivilegeLevel.OBSERVATION},
        )
        for r in results:
            self.assertIsNone(r.forfeit, r.forfeit)
        wins = sum(1 for r in results if r.winner == 1)
        self.assertGreaterEqual(wins, 3)


class TestDiagnostics(unittest.TestCase):
    def test_hidden_info_diagnostic_runs_and_reports_every_condition(self):
        from tools.diag_hidden_info import run as run_hidden_info

        report = run_hidden_info(n_games=1, n_samples=1, max_turns=30, run_seed=0)
        self.assertEqual(len(report), 4)
        for label, wins, n, forfeits in report:
            self.assertEqual(n, 1)
            self.assertIn(wins, (0, 1))

    def test_search_curve_diagnostic_runs_and_reports_every_level(self):
        from tools.diag_search_curve import run as run_search_curve

        report = run_search_curve(n_games=1, sample_counts=[1, 2], max_turns=30, run_seed=0)
        self.assertEqual(len(report), 2)
        for n_samples, wins, n, forfeits in report:
            self.assertIn(n_samples, (1, 2))
            self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
