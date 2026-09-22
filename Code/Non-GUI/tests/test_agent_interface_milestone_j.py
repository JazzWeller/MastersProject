"""Milestone J (Code/AGENT_INTERFACE_PLAN.md): data generation, the
no-network baseline, and the two diagnostics.
"""

import os
import tempfile
import unittest

from bots.baseline_evaluator import heuristic_value, uniform_policy
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from bots.search_bot import DeterminizedRolloutBot
from keyforge.config import GameConfig
from keyforge.decision import Decision
from keyforge.enums import DecisionKind, PrivilegeLevel
from keyforge.game import Game
from sim.driver import derive_agent_seed, run_games
from sim.generate import play_and_record, read_shard, write_privileged_shard, write_shard
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


class TestTrajectoryGeneration(unittest.TestCase):
    def test_play_and_record_produces_a_valid_replayable_trajectory(self):
        config = GameConfig(decks=("fignor", "igor"), seed=5, max_turns=60)
        agents = {1: HeuristicBot(seed=5), 2: RandomBot(seed=6)}
        trajectory, privileged = play_and_record(config, agents)
        self.assertGreater(len(trajectory.decisions), 0)
        self.assertIn(trajectory.outcome["1"], (-1, 0, 1))
        self.assertEqual(trajectory.outcome["1"], -trajectory.outcome["2"])

        from keyforge.replay import config_from_dict, replay

        fork = replay(config_from_dict(trajectory.config), trajectory.choice_record)
        self.assertTrue(fork.is_over)
        self.assertEqual(fork.outcome_for(1), trajectory.outcome["1"])

        for pid_str in ("1", "2"):
            pid = int(pid_str)
            self.assertEqual(sorted(privileged.final_hand[pid_str]), sorted(c.instance_id for c in fork.players[pid].hand.cards()))

    def test_shards_round_trip_through_disk(self):
        config = GameConfig(decks=("fignor", "igor"), seed=9, max_turns=40)
        agents = {1: RandomBot(seed=9), 2: RandomBot(seed=10)}
        t1, p1 = play_and_record(config, agents)
        agents2 = {1: RandomBot(seed=11), 2: RandomBot(seed=12)}
        t2, p2 = play_and_record(GameConfig(decks=("fignor", "igor"), seed=11, max_turns=40), agents2)

        with tempfile.TemporaryDirectory() as d:
            shard_path = os.path.join(d, "shard.jsonl")
            priv_path = os.path.join(d, "shard.privileged.jsonl")
            write_shard(shard_path, [t1, t2])
            write_privileged_shard(priv_path, [p1, p2])
            loaded = read_shard(shard_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].choice_record, t1.choice_record)
            self.assertEqual(loaded[1].outcome, t2.outcome)
            # Appending more games to the same shard doesn't clobber it.
            write_shard(shard_path, [t1])
            self.assertEqual(len(read_shard(shard_path)), 3)


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
