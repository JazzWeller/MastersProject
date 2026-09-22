"""The agent conformance suite (Code/AGENT_INTERFACE_PLAN.md, Milestone K)
-- the plan's own definition of done: "An agent that passes this suite
runs in the harness, the GUI, the data generator, the self-play actors and
both diagnostics with no further change."

Parameterized over every agent in `AGENTS` below. Milestone I's
`bots/registry.py` will supply this list once it exists; for now it's the
two agents the plan itself measured a baseline against.
"""

import json
import os
import random
import subprocess
import sys
import unittest

from bots.base import Budget, Controller, SyncBatchAdapter
from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.capabilities import make_capability
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, PrivilegeLevel
from keyforge.game import Game
from keyforge.match import Match, MatchConfig
from sim.driver import run_games
from sim.simulate import _BOUNDARY_KINDS, check_invariants

AGENTS = [
    ("random", lambda seed: RandomBot(seed=seed)),
    ("heuristic", lambda seed: HeuristicBot(seed=seed)),
]

N_GAMES = 12
MAX_TURNS = 150
_NON_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_contract_worker.py")


def _play_game(factory, seed, max_turns=MAX_TURNS, decks=("fignor", "igor")):
    game = Game(GameConfig(decks=decks, seed=seed, max_turns=max_turns))
    bots = {1: factory(seed), 2: factory(seed + 500)}
    while not game.is_over:
        d = game.pending_decision
        choice = bots[d.player].decide(game.view_for(d.player), d)
        game.submit(choice)
    return game


class TestLegality(unittest.TestCase):
    def test_every_submission_satisfies_validate_in_game(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                for seed in range(N_GAMES):
                    game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=MAX_TURNS))
                    bots = {1: factory(seed), 2: factory(seed + 500)}
                    while not game.is_over:
                        d = game.pending_decision
                        choice = bots[d.player].decide(game.view_for(d.player), d)
                        self.assertTrue(d.validate(choice), f"{name}: invalid choice {choice!r} for {d}")
                        game.submit(choice)

    def test_every_submission_satisfies_validate_in_a_match(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                for seed in range(4):
                    match = Match(MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=seed, max_turns=MAX_TURNS))
                    bots = {1: factory(seed), 2: factory(seed + 500)}
                    steps = 0
                    while not match.is_over and steps < 20000:
                        d = match.pending_decision
                        choice = bots[d.player].decide(match.view_for(d.player), d)
                        self.assertTrue(d.validate(choice), f"{name}: invalid match choice {choice!r} for {d}")
                        match.submit(choice)
                        steps += 1
                    self.assertTrue(match.is_over, f"{name}: match never finished within budget")


class TestMultiSelectCorrectness(unittest.TestCase):
    def test_respects_min_n_max_n_and_order_effects_is_a_true_permutation(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                for seed in range(N_GAMES):
                    game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=MAX_TURNS))
                    bots = {1: factory(seed), 2: factory(seed + 500)}
                    while not game.is_over:
                        d = game.pending_decision
                        choice = bots[d.player].decide(game.view_for(d.player), d)
                        if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
                            self.assertIsInstance(choice, list)
                            self.assertGreaterEqual(len(choice), d.min_n)
                            self.assertLessEqual(len(choice), d.max_n)
                            if d.kind == DecisionKind.ORDER_EFFECTS:
                                self.assertEqual(sorted(id(c) for c in choice), sorted(id(o) for o in d.options))
                        game.submit(choice)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_a_byte_identical_record_in_process(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                g1 = _play_game(factory, seed=42)
                g2 = _play_game(factory, seed=42)
                self.assertEqual(g1.choice_record, g2.choice_record)
                self.assertEqual(g1.result, g2.result)

    def test_same_seed_replays_identically_in_a_freshly_spawned_worker(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                g1 = _play_game(factory, seed=43)
                proc = subprocess.run(
                    [sys.executable, _WORKER, name, "43"],
                    cwd=_NON_GUI_DIR, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                worker_out = json.loads(proc.stdout)
                self.assertEqual(worker_out["choice_record"], g1.choice_record)
                self.assertEqual(worker_out["result"], g1.result)


class TestNonMutation(unittest.TestCase):
    def test_state_hash_is_unchanged_across_decide_and_invariants_hold_at_boundaries(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                game = Game(GameConfig(decks=("fignor", "igor"), seed=7, max_turns=MAX_TURNS))
                bots = {1: factory(7), 2: factory(507)}
                while not game.is_over:
                    d = game.pending_decision
                    before = game.state_hash()
                    choice = bots[d.player].decide(game.view_for(d.player), d)
                    after = game.state_hash()
                    self.assertEqual(before, after, f"{name}: decide() mutated game state for {d.kind.name}")
                    game.submit(choice)
                    if game.is_over or game.pending_decision.kind in _BOUNDARY_KINDS:
                        check_invariants(game)


class TestTermination(unittest.TestCase):
    def test_game_finishes_within_max_turns(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                for seed in range(N_GAMES):
                    game = _play_game(factory, seed, max_turns=MAX_TURNS)
                    self.assertTrue(game.is_over, f"{name}: game did not finish")
                    self.assertLessEqual(game.turn_number, MAX_TURNS)


class TestBatchSyncAgreement(unittest.TestCase):
    def test_decide_many_matches_individual_decide_under_the_same_seed(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                game = Game(GameConfig(decks=("fignor", "igor"), seed=9, max_turns=40))
                bot_sync = factory(9)
                bot_batch = SyncBatchAdapter(factory(9))
                while not game.is_over:
                    d = game.pending_decision
                    view = game.view_for(d.player)
                    single = bot_sync.decide(view, d)
                    batched = bot_batch.decide_many([(view, d, None, None)])[0]
                    self.assertEqual(single, batched, f"{name}: decide_many disagreed with decide for {d.kind.name}")
                    game.submit(single)


class _HookRecorder(Controller):
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def decide(self, view, decision, budget=None, capability=None):
        return self.inner.decide(view, decision, budget, capability)

    def on_game_start(self, seat, config):
        self.calls.append(("start", seat))
        self.inner.on_game_start(seat, config)

    def observe(self, event):
        self.calls.append(("observe", event))
        self.inner.observe(event)

    def on_game_end(self, outcome):
        self.calls.append(("end", outcome))
        self.inner.on_game_end(outcome)


class TestHooks(unittest.TestCase):
    def test_hooks_fire_in_order_through_the_driver_including_opponent_moves(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                rec1, rec2 = _HookRecorder(factory(1)), _HookRecorder(factory(2))
                run_games(1, lambda i: GameConfig(decks=("fignor", "igor"), seed=1, max_turns=MAX_TURNS), lambda i: {1: rec1, 2: rec2})
                for who, rec in (("seat1", rec1), ("seat2", rec2)):
                    kinds = [c[0] for c in rec.calls]
                    self.assertEqual(kinds[0], "start", f"{name}/{who}")
                    self.assertEqual(kinds[-1], "end", f"{name}/{who}")
                    self.assertGreater(kinds.count("observe"), 0, f"{name}/{who}: never observed anything")
                    self.assertEqual(kinds.count("start"), 1, f"{name}/{who}: on_game_start fired more than once")
                    self.assertEqual(kinds.count("end"), 1, f"{name}/{who}: on_game_end fired more than once")


class TestBudget(unittest.TestCase):
    def test_agent_stays_within_a_generous_wall_clock_budget(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                results = run_games(
                    3,
                    lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=100),
                    lambda i: {1: factory(i), 2: factory(i + 500)},
                    budget=Budget(wall_clock_seconds=1.0),
                )
                for r in results:
                    self.assertIsNone(r.forfeit, f"{name}: forfeited under a 1s/decision budget: {r.forfeit}")


class TestPrivilege(unittest.TestCase):
    def test_observation_level_agent_is_never_handed_a_fork_capable_object(self):
        for name, factory in AGENTS:
            with self.subTest(agent=name):
                seen_caps = []

                class _CapProbe(Controller):
                    def __init__(self, inner):
                        self.inner = inner

                    def decide(self, view, decision, budget=None, capability=None):
                        seen_caps.append(capability)
                        return self.inner.decide(view, decision, budget, capability)

                run_games(
                    1,
                    lambda i: GameConfig(decks=("fignor", "igor"), seed=1, max_turns=MAX_TURNS),
                    lambda i: {1: _CapProbe(factory(1)), 2: factory(2)},
                    privilege={1: PrivilegeLevel.OBSERVATION},
                )
                self.assertTrue(seen_caps)
                for cap in seen_caps:
                    self.assertFalse(hasattr(cap, "fork"), f"{name}: observation-level capability exposed fork()")
                    self.assertFalse(hasattr(cap, "fork_determinized"), f"{name}: observation-level capability exposed fork_determinized()")

    def test_a_search_agent_cannot_recover_true_hidden_state_from_a_determinized_fork(self):
        factory = AGENTS[0][1]
        game = _play_game(factory, seed=17, max_turns=80)
        opponent_hand = game.players[2].hand.cards()
        if len(opponent_hand) < 2:
            self.skipTest("opponent hand too small at this point to measure a meaningful match rate")
        cap = make_capability(PrivilegeLevel.SEARCH, game, 1)
        true_ids = sorted(c.instance_id for c in opponent_hand)
        trials = 30
        matches = 0
        for i in range(trials):
            fork = cap.fork_determinized(random.Random(i))
            fork_ids = sorted(c.instance_id for c in fork.players[2].hand.cards())
            if fork_ids == true_ids:
                matches += 1
        self.assertLess(matches, trials * 0.5, "a determinized fork matched the true hand far too often")


if __name__ == "__main__":
    unittest.main()
