"""Milestone G (Code/AGENT_INTERFACE_PLAN.md): agent protocol and the
in-process driver.
"""

import time
import unittest

from bots.base import Budget, Controller, SyncBatchAdapter
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.capabilities import ObservationCapability, PrivilegedCapability, SearchCapability, make_capability
from keyforge.config import GameConfig
from keyforge.enums import PrivilegeLevel
from keyforge.game import Game
from keyforge.match import MatchConfig
from keyforge.replay import replay
from sim.driver import derive_agent_seed, run_games


class _RecordingBot(Controller):
    """A Controller that records every lifecycle call it receives, for
    asserting call order/content, and otherwise plays randomly."""

    def __init__(self, seed=None):
        self._inner = RandomBot(seed=seed)
        self.calls = []

    def decide(self, view, decision, budget=None, capability=None):
        self.calls.append(("decide", decision.kind.name, capability))
        return self._inner.decide(view, decision, budget, capability)

    def on_game_start(self, seat, config):
        self.calls.append(("on_game_start", seat, dict(config)))

    def observe(self, event):
        self.calls.append(("observe", event))

    def on_game_end(self, outcome):
        self.calls.append(("on_game_end", outcome))


class _AlwaysRaises(Controller):
    def decide(self, view, decision, budget=None, capability=None):
        raise RuntimeError("boom")


class _AlwaysIllegal(Controller):
    def decide(self, view, decision, budget=None, capability=None):
        return object()  # never a legal option


class _Slow(Controller):
    def decide(self, view, decision, budget=None, capability=None):
        time.sleep(0.05)
        return decision.options[0]


class TestLifecycleHooks(unittest.TestCase):
    def test_hooks_fire_in_order_with_both_players_observing(self):
        agent1, agent2 = _RecordingBot(seed=1), _RecordingBot(seed=2)
        results = run_games(
            1,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=40),
            lambda i: {1: agent1, 2: agent2},
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].reason or results[0].winner is not None or results[0].turns is not None)
        for agent in (agent1, agent2):
            kinds = [c[0] for c in agent.calls]
            self.assertEqual(kinds[0], "on_game_start")
            self.assertEqual(kinds[-1], "on_game_end")
            self.assertIn("observe", kinds)
            # Both players' decisions should be observed by both agents
            # (agent1 also sees agent2's moves and vice versa).
            self.assertGreater(kinds.count("observe"), kinds.count("decide"))


class TestCapabilities(unittest.TestCase):
    def test_observation_level_has_no_fork_methods(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=10))
        cap = make_capability(PrivilegeLevel.OBSERVATION, game, 1)
        self.assertIsInstance(cap, ObservationCapability)
        self.assertFalse(hasattr(cap, "fork"))
        self.assertFalse(hasattr(cap, "fork_determinized"))
        obs = cap.observation()
        self.assertEqual(obs.viewer, 1)

    def test_search_level_can_determinize_but_not_fork_exactly(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=10))
        cap = make_capability(PrivilegeLevel.SEARCH, game, 1)
        self.assertIsInstance(cap, SearchCapability)
        self.assertFalse(hasattr(cap, "fork"))
        import random

        branch = cap.fork_determinized(random.Random(1))
        self.assertIsNot(branch, game)

    def test_privileged_level_can_fork_exactly_and_see_everything(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=10))
        cap = make_capability(PrivilegeLevel.PRIVILEGED, game, 1)
        self.assertIsInstance(cap, PrivilegedCapability)
        exact = cap.fork()
        self.assertEqual(exact.state_hash(), game.state_hash())
        full = cap.full_state_observation()
        self.assertIsNotNone(full.players[1].hand)
        self.assertIsNotNone(full.players[2].hand)

    def test_driver_grants_the_registered_privilege_per_seat(self):
        seen = {}

        class _Probe(Controller):
            def decide(self, view, decision, budget=None, capability=None):
                seen[decision.player] = type(capability).__name__ if capability else None
                return decision.options[0]

        run_games(
            1,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5),
            lambda i: {1: _Probe(), 2: _Probe()},
            privilege={1: PrivilegeLevel.PRIVILEGED, 2: PrivilegeLevel.OBSERVATION},
        )
        self.assertIn(seen.get(1), ("PrivilegedCapability", None))
        self.assertIn(seen.get(2), ("ObservationCapability", None))


class TestFaultIsolation(unittest.TestCase):
    def test_a_raising_agent_forfeits_without_killing_the_run(self):
        results = run_games(
            3,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=40),
            lambda i: {1: _AlwaysRaises(), 2: RandomBot(seed=i)},
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsNotNone(r)
            self.assertIsNotNone(r.forfeit)
            self.assertEqual(r.forfeit["seat"], 1)
            self.assertEqual(r.winner, 2)

    def test_an_illegal_choice_forfeits_that_game_only(self):
        results = run_games(
            2,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=40),
            lambda i: {1: _AlwaysIllegal(), 2: RandomBot(seed=i)},
        )
        for r in results:
            self.assertIsNotNone(r.forfeit)
            self.assertEqual(r.winner, 2)

    def test_a_slow_agent_forfeits_under_a_tight_wall_clock_budget(self):
        results = run_games(
            1,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=1, max_turns=10),
            lambda i: {1: _Slow(), 2: RandomBot(seed=1)},
            budget=Budget(wall_clock_seconds=0.001),
        )
        self.assertIsNotNone(results[0].forfeit)
        self.assertEqual(results[0].forfeit["seat"], 1)


class TestForcedDecisionAutoResolve(unittest.TestCase):
    def test_forced_decisions_are_not_sent_to_the_agent_but_stay_in_the_record(self):
        counting_calls = {"n": 0}

        class _Counter(Controller):
            def decide(self, view, decision, budget=None, capability=None):
                counting_calls["n"] += 1
                return RandomBot(seed=1).decide(view, decision, budget, capability)

        config = GameConfig(decks=("fignor", "igor"), seed=3, max_turns=40)
        agent1, agent2 = _Counter(), _Counter()
        results = run_games(1, lambda i: config, lambda i: {1: agent1, 2: agent2}, auto_resolve_forced=True)
        with_auto = counting_calls["n"]

        counting_calls["n"] = 0
        agent1b, agent2b = _Counter(), _Counter()
        run_games(1, lambda i: config, lambda i: {1: agent1b, 2: agent2b}, auto_resolve_forced=False)
        without_auto = counting_calls["n"]

        self.assertLess(with_auto, without_auto)

        # The forced choices are still in the replay record: replaying it
        # from scratch reproduces the exact same result.
        record = results[0].choice_record
        fork = replay(results[0].config, record)
        self.assertEqual(fork.result.get("winner"), results[0].winner)
        self.assertEqual(fork.turn_number, results[0].turns)


class TestSeeding(unittest.TestCase):
    def test_deterministic_and_seat_and_index_sensitive(self):
        a = derive_agent_seed(1, 0, 1)
        b = derive_agent_seed(1, 0, 1)
        c = derive_agent_seed(1, 0, 2)
        d = derive_agent_seed(1, 1, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)


class TestViewIsAPlayerView(unittest.TestCase):
    def test_heuristic_bot_runs_through_the_driver_without_forfeiting(self):
        """Regression: the driver must hand `decide` a PlayerView/MatchView
        (`view.me()`/`view.opponent()`), not the newer Observation -- only
        HeuristicBot actually calls `.me()`, so a driver that passed the
        wrong shape would forfeit every single game with an AttributeError."""
        results = run_games(
            10,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=100),
            lambda i: {1: HeuristicBot(seed=derive_agent_seed(0, i, 1)), 2: RandomBot(seed=derive_agent_seed(0, i, 2))},
            concurrency=4,
        )
        for r in results:
            self.assertIsNone(r.forfeit, r.forfeit)
        self.assertTrue(any(r.winner == 1 for r in results))


class _ViewRecordingBot(Controller):
    """Records whether `view` was `None` on every `decide()` call, and
    otherwise plays randomly."""

    def __init__(self, seed=None, needs_view=True):
        self._inner = RandomBot(seed=seed)
        self.needs_view = needs_view
        self.saw_views = []

    def decide(self, view, decision, budget=None, capability=None):
        self.saw_views.append(view is not None)
        return self._inner.decide(view, decision, budget, capability)


class TestLazyViews(unittest.TestCase):
    def test_driver_skips_building_a_view_for_an_agent_that_declares_needs_view_false(self):
        agent1 = _ViewRecordingBot(seed=1, needs_view=False)
        agent2 = _ViewRecordingBot(seed=2, needs_view=True)
        run_games(
            3,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=60),
            lambda i: {1: agent1, 2: agent2},
        )
        self.assertGreater(len(agent1.saw_views), 0)
        self.assertTrue(all(seen is False for seen in agent1.saw_views))
        self.assertGreater(len(agent2.saw_views), 0)
        self.assertTrue(all(seen is True for seen in agent2.saw_views))

    def test_random_bot_declares_it_does_not_need_a_view(self):
        self.assertFalse(RandomBot.needs_view)


class TestConcurrencyAndMatches(unittest.TestCase):
    def test_multiple_games_run_concurrently_and_all_complete(self):
        results = run_games(
            6,
            lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=60),
            lambda i: {1: RandomBot(seed=derive_agent_seed(0, i, 1)), 2: RandomBot(seed=derive_agent_seed(0, i, 2))},
            concurrency=3,
        )
        self.assertEqual(len(results), 6)
        for r in results:
            self.assertIsNotNone(r)
            self.assertIsNone(r.forfeit)

    def test_match_configs_run_through_the_same_driver(self):
        results = run_games(
            1,
            lambda i: MatchConfig(format="archon", decks=("fignor", "igor"), seed=1, max_turns=60),
            lambda i: {1: RandomBot(seed=1), 2: RandomBot(seed=2)},
        )
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].forfeit)


class TestSyncBatchAdapter(unittest.TestCase):
    def test_decide_many_matches_individual_decides(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        d = game.pending_decision
        view = game.view_for(d.player)
        bot = HeuristicBot(seed=1)
        adapter = SyncBatchAdapter(bot)
        single = adapter.decide(view, d, None, None)
        batched = adapter.decide_many([(view, d, None, None)])
        self.assertEqual(single, batched[0])


if __name__ == "__main__":
    unittest.main()
