"""Agent Observation Plan, Milestone O4: the complete current state.

- The coverage registry (`agent/state_registry.py`, I4) classifies every
  attribute of every engine state object seen across fuzz games of every
  pool, and every BOOKKEEPING attribute holds its default.
- The v2 extract (`keyforge/infoset_v2.py`) is byte-identical across every
  world consistent with what the viewer knows (I1), resolution tokens
  included.
- Resolution tokens are identical after a replay, a copy and a snapshot
  round trip, at decisions of every kind.
"""

import random
import unittest

from agent.state_registry import check_game
from keyforge.enums import Resample
from keyforge.game import Game
from keyforge.infoset_v2 import build_infoset_v2, infoset_v2_bytes
from keyforge.replay import decode_choice
from tools.o1_acceptance import POOLS, bots_for, game_config


def _games(per_pool):
    for pool in POOLS:
        for i in range(per_pool):
            config = game_config(pool, i)
            yield pool, i, config, bots_for(i, config.seed)


class TestCoverage(unittest.TestCase):
    def test_every_attribute_is_classified(self):
        problems = set()
        for pool, i, config, bots in _games(6):
            game = Game(config)
            n = 0
            while not game.is_over:
                if n % 4 == 0:
                    check_game(game, problems)
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                n += 1
            check_game(game, problems)
        self.assertEqual(sorted(problems), [])


class TestInvariance(unittest.TestCase):
    def test_the_extract_is_the_same_in_every_world(self):
        checked = 0
        for pool, i, config, bots in _games(2):
            game = Game(config)
            n = 0
            while not game.is_over:
                if n % 11 == 0 and game.copy_anywhere:
                    for v in (1, 2):
                        base = infoset_v2_bytes(build_infoset_v2(game, v))
                        # (every consistent re-deal: `uniform` ignores what the viewer knows)
                        for sampler in ("constrained", "chance_exact"):
                            world = game.fork_determinized(v, random.Random(n), Resample.ALL, backend="copy",
                                                           sampler=sampler)
                            self.assertEqual(infoset_v2_bytes(build_infoset_v2(world, v)), base,
                                             f"{pool} game {i}, decision {n}, viewer {v}, {sampler}")
                            checked += 1
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                n += 1
        self.assertGreater(checked, 100)


class TestResolutionRoundTrips(unittest.TestCase):
    def test_tokens_survive_replay_copy_and_snapshot(self):
        kinds = set()
        for pool, i, config, bots in _games(2):
            game = Game(config, execution="compiled")  # (the explicit stack: compiled execution)
            rng = random.Random(config.seed)
            while not game.is_over:
                if rng.random() < 0.15:
                    kinds.add(game.pending_decision.kind)
                    for v in (1, 2):
                        tokens = build_infoset_v2(game, v)["resolution"]
                        replay = Game(config, execution="compiled")
                        for encoded in game.choice_record:
                            replay.submit(decode_choice(replay.pending_decision, encoded))
                        self.assertEqual(build_infoset_v2(replay, v)["resolution"], tokens)
                        self.assertEqual(build_infoset_v2(game.copy(), v)["resolution"], tokens)
                        self.assertEqual(build_infoset_v2(Game.restore(game.snapshot()), v)["resolution"], tokens)
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertGreaterEqual(len(kinds), 4)


if __name__ == "__main__":
    unittest.main()
